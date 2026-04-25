from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from model import create_model, WeightedMSELoss
from dataset import MonocularImageDataset
from datastructs import Checkpoint, Infrastructure
import utils
import transforms as transforms_module


@dataclass
class TrainingArtifacts:
    config: Dict[str, Any]
    checkpoint: Checkpoint
    infrastructure: Infrastructure
    checkpoint_path: str


def prepare_infrastructure(config) -> Infrastructure:
    transforms_dict = transforms_module.create_transforms(config)
    offset_ranges = config["data"].get("offset_ranges", [0.30, 0.30, 0.13])
    criterion = WeightedMSELoss(offset_ranges).to(config["device"])
    dataset = MonocularImageDataset(config['data']['dataset_path'])

    train_loader, val_loader, train_dataset, val_dataset = utils.train_val_split(
        dataset, config, config["experiment"]["seed"]
    )
    if 'tqdm' in config['logging']:
        train_loader = tqdm(train_loader)
        val_loader = tqdm(val_loader)

    return Infrastructure(
        train_transforms=transforms_dict['train'],
        val_transforms=transforms_dict['eval'],
        criterion=criterion,
        valloader=val_loader,
        valset=val_dataset,
        trainloader=train_loader,
        trainset=train_dataset
    )


def prepare_checkpoint(config) -> Checkpoint:
    model = create_model(config)
    training_config = config['training']
    optimizer = getattr(torch.optim, training_config['optimizer']['name'])(model.parameters(), **training_config['optimizer']['args'])
    scheduler = getattr(torch.optim.lr_scheduler, training_config['scheduler']['name'])(optimizer, **training_config['scheduler']['args'])

    return Checkpoint(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
    )

def prepare_training(config_path: str) -> TrainingArtifacts:
    config = utils.load_config(config_path)
    dirs = utils.create_directories(config)

    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(config["experiment"]["seed"])

    checkpoint = prepare_checkpoint(config)
    infrastructure = prepare_infrastructure(config)

    return TrainingArtifacts(
        config=config,
        infrastructure=infrastructure,
        checkpoint=checkpoint,
        checkpoint_path=dirs.checkpoint_dir,
    )
