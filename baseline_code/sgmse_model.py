from typing import Any
import lightning as L
from torch.optim.optimizer import Optimizer
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torchaudio
from baseline_code.bsrnn_sgmse import SGMSE_BSRNN
from baseline_code.config import Config 
from espnet2.enh.loss.criterions.time_domain import SISNRLoss, MultiResL1SpecLoss


class SGMSEModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.se_model = SGMSE_BSRNN()


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
    
    def forward_step(self, batch, stage='train'):

        clean_speech, noisy_speech, fs, speech_length = batch
        batch_size = len(clean_speech)
        B, C, T = clean_speech.shape
        assert C == 1
        clean_speech = clean_speech.view(B, T).float()
        noisy_speech = noisy_speech.view(B, T).float()


        loss = self.se_model.forward(noisy_speech, clean_speech, speech_length, fs)

        self.log(f'{stage}_loss', loss.detach().item(),
                 on_step=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch):

        loss = self.forward_step(batch)

        return loss

    def validation_step(self, batch):
        loss = self.forward_step(batch, stage='val')

        return {'val_loss': loss.detach()}

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
