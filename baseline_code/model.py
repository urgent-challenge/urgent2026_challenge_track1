from typing import Any
import lightning as L
from torch.optim.optimizer import Optimizer
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torchaudio
from baseline_code.bsrnn import BSRNN
from baseline_code.config import Config 
from espnet2.enh.loss.criterions.time_domain import SISNRLoss, MultiResL1SpecLoss


class SEModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.nn = torch.nn.Linear(100, 100)
        self.se_model = BSRNN()
        self.mr_l1_loss = MultiResL1SpecLoss(window_sz=[256, 512, 768, 1024], eps = 1.0e-6,normalize_variance=True, time_domain_weight=0.5)
        self.sisnr_loss = SISNRLoss()
        self.grad_has_nan = False

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:

        return

    def on_after_backward(self):
        # check if has grad of NaN
        self.grad_has_nan = any(
            torch.isnan(p.grad).any() 
            for p in self.parameters() 
            if p.grad is not None
        )

    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.grad_has_nan:
            print('NaN in grad has been decected, reset grad to zero')
            optimizer.zero_grad()
            self.grad_has_nan = False
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def forward_step(self, batch, stage='train'):

        clean_speech, noisy_speech, fs, speech_length = batch
        batch_size = len(clean_speech)
        B, C, T = clean_speech.shape
        assert C == 1
        clean_speech = clean_speech.view(B, T).float()
        noisy_speech = noisy_speech.view(B, T).float()

        se_speech, se_spec = self.se_model(noisy_speech, speech_length, fs)

        loss = self.mr_l1_loss(clean_speech, se_speech).mean()

        if torch.isnan(loss):
            print('NaN in loss has been decected, skip')
            return se_speech.mean() * 0  # Skip current step

        with torch.no_grad():
            sisnr_loss = self.sisnr_loss(clean_speech, se_speech).mean()

        # loss = loss + sisnr_loss
        loss = loss


        self.log(f'{stage}_loss', loss.detach().item(),
                 on_step=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}_sisnr', - sisnr_loss.detach().item(),
                 on_step=True, prog_bar=True, batch_size=batch_size)

        self.log(f'{stage}_sisnr_{fs}', - sisnr_loss.detach().item(),
                 on_step=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch):

        loss = self.forward_step(batch)

        return loss

    def validation_step(self, batch):
        loss = self.forward_step(batch, stage='val')

        return {'loss': loss.detach()}

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
