import yaml
from dataclasses import asdict
import os

import cv2
import numpy as np
import torch

from datastructs import StereoSample, Checkpoint, Directories


def get_model_device(model):
    for p in model.parameters():
        return p.device

    for b in model.buffers():
        return b.device

    return "cpu"

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['run_name'] = os.path.basename(config_path)
    return config


def create_directories(config):
    run_name = config["run_name"]
    diagnostics_dir = config.setdefault("diagnostics", "diagnostics")
    checkpoint_dir = config["training"].get("checkpoint_dir", f"checkpoints/{run_name}")
    for d in (checkpoint_dir, diagnostics_dir):
        os.makedirs(d, exist_ok=True)
    return Directories(diagnostics_dir=diagnostics_dir, checkpoint_dir=checkpoint_dir)


def save_checkpoint(checkpoint: Checkpoint, path_to_dir, name):
    torch.save(checkpoint.todict(), f"{path_to_dir}/{name}.pth")

def train_val_split(dataset, config, seed):
    val_ratio = config["data"]["val_split"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    assert 0 < val_ratio < 1, f"val_split must be between 0 and 1, got {val_ratio}"

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [1 - val_ratio, val_ratio],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=StereoSample.collate,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=StereoSample.collate,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )

    return train_loader, val_loader, train_dataset, val_dataset

def unnormalize(img, mean, std):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    assert len(img.shape) >= 3 and img.shape[-3] == 3
    return img * std[..., None, None] + mean[..., None, None]


def img_torch2np(img):
    dims = (1, 2, 0) if len(img.shape) == 3 else (0, 2, 3, 1)
    return (
        cv2.normalize(
            img.permute(*dims).cpu().numpy(),
            None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
    )