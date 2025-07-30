#!/bin/env python

import os
import pytorch_lightning as L
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import  ModelCheckpoint,LearningRateMonitor
import torch.multiprocessing as mp
from baseline_code.d_model import SEModel
from baseline_code.flow_model import FlowSEModel
from baseline_code.config import Config
from baseline_code.dataset import AudioDataModule
import glob
from baseline_code.config import config_parser


def prepare_call_backs(cfg):

    best_metrics = [
        ('val_loss', 'min'),
        ]
    call_backs = [LearningRateMonitor(logging_interval='epoch')]
    for i, (metric, min_or_max) in enumerate(best_metrics):
        call_back = ModelCheckpoint(
            filename="best_{epoch:02d}-{step:06d}-{"+ metric + ":.3f}",
            save_top_k=cfg.save_top_k,
            monitor=metric,
            every_n_train_steps=cfg.val_check_interval,
            mode=min_or_max,
            save_weights_only=(metric != "val_loss"),
            save_last=False,
        )
        call_backs.append(call_back)

    return call_backs

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    
    args = config_parser()
    cfg = Config(**vars(args))
    cfg.read_yaml()
    print(cfg)
    L.seed_everything(seed=cfg.seed)

    if cfg.train_set_dynamic_mixing:
        os.environ['OMP_NUM_THREADS'] = "1"

    if cfg.model_type == "flowse":
        model = FlowSEModel(cfg=cfg)
    else:
        model = SEModel(cfg=cfg)

    if cfg.init_from != 'none':
        state_dict = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        print(f"Init param loaded from {cfg.init_from}")

    print(model)

    logger = TensorBoardLogger(save_dir=f"./exp/{cfg.train_tag}", version=cfg.train_version, name=cfg.train_name)
    call_backs = prepare_call_backs(cfg=cfg)

    ckpts = glob.glob(f"./exp/{cfg.train_tag}/{cfg.train_name}/version_{cfg.train_version}/checkpoints/*-val_loss*.ckpt")
    ckpts.sort(key=os.path.getmtime, reverse=True)
    last_ckpt = ckpts[0] if ckpts else None
    last_ckpt = last_ckpt if cfg.resume  else None
    if last_ckpt is not None:
        print(f"Resume form {last_ckpt}")

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
