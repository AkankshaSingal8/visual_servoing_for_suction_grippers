#!/usr/bin/env python3
"""! @file train.py

@brief Primary Goal: Neural Network to map monocular visual data to the ground truth grasp offsets.

"""


import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
import torch.nn as nn
from torchvision.transforms import v2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import tyro

import utils
import eval
from datastructs import EpochInfo
from trainutils import prepare_training, TrainingArtifacts

import loggers

def create_loggers(config):
    run_name = config['run_name']
    logger_list = []
    if 'wandb' in config['logging']:
        wandb_options = config['logging']['wandb']
        logger_list.append(loggers.WandbLogger(run_name, **wandb_options))
    if 'diagnostics' in config['logging']:
        logger_list.append(loggers.DiagnosticsLogger(run_name))
    if 'checkpoint' in config['logging']:
        checkpoint_options = config['logging']['checkpoint']
        logger_list.append(loggers.CheckpointLogger(run_name, **checkpoint_options))

    return logger_list

def run_epoch(model: nn.Module, criterion: nn.Module, optimizer, train_loader, train_transform: v2.Compose, epoch: int):
    train_losses = []
    model.train()
    assert len(train_loader) > 0, "Train loader is empty"
 
    device = utils.get_model_device(model)
    for batched_sample in train_loader:
        batched_sample.move_to(device)
        optimizer.zero_grad()
        batched_sample.transform(train_transform)
        pred_offsets = model(batched_sample)
        loss = criterion(pred_offsets, batched_sample.batched_offset)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

    return EpochInfo(
        epoch=epoch,
        avg_train_loss=torch.stack(train_losses).mean().cpu().item(),
    )

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()


def run_training(artifacts: TrainingArtifacts, loggers):
    checkpoint = artifacts.checkpoint
    infra = artifacts.infrastructure
    num_epochs = artifacts.config['training']['num_epochs']

    for logger in loggers:
        logger.on_train_start(checkpoint.model)

    for epoch in tqdm(range(num_epochs)):

        trainloader = tqdm(infra.trainloader, desc=f"Epoch {epoch} training", leave=False)
        epoch_info = run_epoch(checkpoint.model, infra.criterion, checkpoint.optimizer, trainloader, infra.train_transforms, epoch)

        valloader = tqdm(infra.valloader, desc=f"Epoch {epoch} evaling", leave=False)
        eval_info = eval.eval_model(checkpoint.model, infra.criterion, valloader, infra.val_transforms, epoch)
        checkpoint.scheduler.step(eval_info.avg_val_loss)

        for logger in loggers:
            logger.on_epoch_end(checkpoint, epoch_info, eval_info)

    return artifacts


def main(config_path: str = "experiment_configs/throwaway.yaml"):

    artifacts = prepare_training(config_path)
    loggers = create_loggers(artifacts.config)

    try:
        artifacts = run_training(
            artifacts=artifacts,
            loggers=loggers
        )
    finally:
        for logger in loggers:
            logger.close()

if __name__ == "__main__":
    tyro.cli(main)

"""
python train.py --config-path experiment_configs/new_train.yaml
"""
