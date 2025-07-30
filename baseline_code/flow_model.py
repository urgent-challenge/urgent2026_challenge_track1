from typing import Any
import pytorch_lightning as L
from torch.optim.optimizer import Optimizer
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torchaudio
from baseline_code.models.bsrnn_flowse import BSRNN
from baseline_code.config import Config 
from espnet2.enh.loss.criterions.time_domain import SISNRLoss, MultiResL1SpecLoss
from torch_ema import ExponentialMovingAverage
from baseline_code.models.odes import FLOWMATCHING
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

from baseline_code.sampling import  get_white_box_solver

class FlowSEModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.sisnr_loss = SISNRLoss()

        self.save_hyperparameters()
        self.cfg = cfg
        self.ode = FLOWMATCHING(sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max)
        self.encoder = STFTEncoder(
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    use_builtin_complex=True,
                    default_fs=48000,
                    spec_transform_type=cfg.spec_transform_type,
                    spec_abs_exponent=cfg.spec_abs_exponent,
                    spec_factor=cfg.spec_factor
                )
        self.decoder = STFTDecoder(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            default_fs=48000,
            spec_transform_type='exponent',
            spec_abs_exponent=cfg.spec_abs_exponent,
            spec_factor=cfg.spec_factor
        )

        self.dnn = BSRNN(input_dim=self.encoder.output_dim, 
                         num_spk=1,
                         num_layer=cfg.num_layer,
                         target_fs=48000,
                         causal=False,
                         num_channel=cfg.bsrnn_hidden)
        
        self.lr = cfg.learning_rate
        self.ema_decay = cfg.ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = cfg.t_eps
        self.T_rev = cfg.T_rev
        self.ode.T_rev = cfg.T_rev
        self.loss_type = cfg.loss_type
        self.num_eval_files = 3
        self.loss_abs_exponent = cfg.loss_abs_exponent


    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        return

    def on_after_backward(self):
        return
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):

        # check if has grad of NaN
        grad_has_nan = any(
            torch.isnan(p.grad).any() 
            for p in self.parameters() 
            if p.grad is not None
        )
        if grad_has_nan:
            rank = torch.distributed.get_rank()
            print(f'RANK {rank}: NaN in grad has been decected, reset grad to zero')
            optimizer.zero_grad()
            
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res
    
    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x-x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield-condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield-condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    def speech_to_feature(self, speech, fs, speech_length):

        feature, f_lens = self.encoder(speech, speech_length, fs=fs) # B, T, F
        
        feature = feature.permute(0, 2, 1).unsqueeze(1)    
        return feature
    
    def feature_to_speech(self, feature, fs, speech_length):
        
        feature = feature.squeeze(1).permute(0, 2, 1)

        speech, _=  self.decoder(feature, speech_length, fs)
        return speech


    def forward_step(self, batch):

        clean_speech, noisy_speech, fs, speech_length = batch
        B, C, T = clean_speech.shape
        assert C == 1
        clean_speech = clean_speech.view(B, T).float()
        noisy_speech = noisy_speech.view(B, T).float()
        clean_speech = torch.nan_to_num(clean_speech, nan=0)
        noisy_speech = torch.nan_to_num(noisy_speech, nan=0)
        
        x0 = self.speech_to_feature(clean_speech, fs, speech_length)
        y = self.speech_to_feature(noisy_speech, fs, speech_length)

        rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.T_rev))
        mean, std = self.ode.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)  #
        sigmas = std[:, None, None, None]
        xt = mean + sigmas * z
        der_std = self.ode.der_std(t)
        der_mean = self.ode.der_mean(x0,t,y)
        condVF = der_std * z + der_mean
        vectorfield = self(xt, t, y)
        loss = self._loss(vectorfield, condVF)

        debug = False
        if debug:
            import matplotlib.pyplot as plt
            for tt in torch.linspace(0.99, 0.0, 30):
                tt = torch.ones_like(t) * tt
                mean, std = self.ode.marginal_prob(x0, tt, y)
                z = torch.randn_like(x0)  #
                sigmas = std[:, None, None, None]
                xt = mean + sigmas * z
                plt.imshow(abs(xt[0][0]).cpu().numpy())
                plt.savefig(f'/tmp/debug_foward.{tt.mean().item():.2f}.png')


        return loss
    
    def enhance(self, y, fs, speech_length, N=15):
        # y: B, T

        Y = self.speech_to_feature(y, fs, speech_length)
        
        sampler = get_white_box_solver("euler", self.ode, self, Y.cuda(), T_rev=self.T_rev, t_eps=self.t_eps, N=N)

        sample, _ = sampler()

        enhanced = self.feature_to_speech(sample, fs, speech_length)

        return enhanced


    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score
    def training_step(self, batch, batch_idx):

        loss = self.forward_step(batch)
        self.log('train_loss', loss, on_step=True,prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step(batch)
        self.log('val_loss', loss, on_step=False, prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            clean_speech, noisy_speech, fs, speech_length = batch
            B, C, T = clean_speech.shape
            assert C == 1
            clean_speech = clean_speech.view(B, T).float()
            noisy_speech = noisy_speech.view(B, T).float()

            predicted = self.enhance(noisy_speech, fs, speech_length, N=10)
            sisnr_loss = self.sisnr_loss(clean_speech, predicted).mean()
            self.log(f'sisnr', - sisnr_loss.detach().item(),
                 on_step=True, prog_bar=True, batch_size=B)

        return {'val_loss': loss.detach()}
    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)

        return [optimizer], [scheduler]
