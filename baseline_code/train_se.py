#!/bin/env python

import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
import argparse
from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import  ModelCheckpoint,LearningRateMonitor
import torch.multiprocessing as mp
from baseline_code.model import SEModel
from baseline_code.config import Config
from baseline_code.dataset import AudioDataModule

def config_parser():
    cfg = Config(
        learning_rate = 1e-4,
        batch_size = 2,
        weight_decay = 1e-6,
        adam_epsilon = 1e-8,
        warmup_steps = 2,
        num_worker = 4,
        num_train_epochs = 150,
        gradient_accumulation_steps = 1,
        device = "cuda",
        num_gpu = 1,
        train_version = 0,
        train_tag = "debug",
        train_name = 'baseline_bsrnn',
        val_check_interval = 50000,
        save_top_k = 3,
        resume = True,
        seed = 1996,
        gradient_clip = 0.5,
        lr_step_size = 1,
        lr_gamma = 0.85,
        train_set_path = 'none',
        train_set_dynamic_mixing = True,
        valid_set_path = 'none',
        max_duration=192000,
    )
    parameters = vars(cfg)

    parser = ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    for par in parameters:
        default = parameters[par]
        parser.add_argument(f"--{par}", type=str2bool if isinstance(default, bool) else type(default), default=default)
    args = parser.parse_args()
    return args

    



def prepare_call_backs(cfg):

    best_metrics = [
        ('val_loss', 'min'),
        ('val_sisnr', 'max'),]
    call_backs = [LearningRateMonitor(logging_interval='epoch')]
    for i, (metric, min_or_max) in enumerate(best_metrics):
        call_back = ModelCheckpoint(
            filename="best_{epoch:02d}-{step:06d}-{"+ metric + ":.3f}",
            save_top_k=cfg.save_top_k,
            monitor=metric,
            every_n_train_steps=cfg.val_check_interval,
            mode=min_or_max,
            save_weights_only=(metric != "val_loss"),
            save_last=(metric == "val_loss"),
        )
        call_backs.append(call_back)


    return call_backs

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    args = config_parser()
    cfg = Config(**vars(args))
    print(cfg)
    L.seed_everything(seed=cfg.seed)


    model = SEModel(cfg=cfg)

    print(model)

    logger = TensorBoardLogger(save_dir=f"./exp/{cfg.train_tag}", version=cfg.train_version, name=cfg.train_name)
    call_backs = prepare_call_backs(cfg=cfg)

    last_ckpt = f"./exp/{cfg.train_tag}/{cfg.train_name}/version_{cfg.train_version}/checkpoints/last.ckpt"
    last_ckpt = last_ckpt if cfg.resume and os.path.exists(last_ckpt) else None


    trainer = L.Trainer(
        max_epochs=cfg.num_train_epochs,
        accelerator=cfg.device,
        devices=cfg.num_gpu,
        gradient_clip_val=cfg.gradient_clip,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        callbacks=call_backs,
        strategy='ddp_find_unused_parameters_true',
    )
    trainer.fit(model=model, datamodule=AudioDataModule(config=cfg), ckpt_path=last_ckpt,)
