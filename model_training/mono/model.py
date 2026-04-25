import pdb
import sys
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from datastructs import BatchedMonocularSample

def _resolve_model_class(name: str):
    this_module = sys.modules[__name__]
    model_class = getattr(this_module, name, None)
    if model_class is None:
        raise NotImplementedError(f"Model architecture '{name}' not implemented!")
    return model_class


def create_model(config):
    model_cfg = config["model"]
    model_class = _resolve_model_class(model_cfg["architecture"])
    model_args = model_cfg.get("args", {})
    model = model_class(**model_args)
    return model.to(config["device"])

class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for handling different output ranges.
    
    Weights each dimension inversely proportional to the square of its range,
    so dimensions with smaller ranges contribute more to the loss.
    """
    
    def __init__(self, offset_ranges=[0.30, 0.30, 0.13]):
        """
        Args:
            offset_ranges: List of [x_range, y_range, z_range] representing
                the span of each offset dimension.
        """
        super().__init__()
        weights = [1 / (r ** 2) for r in offset_ranges]
        weights = torch.tensor(weights) / sum(weights)
        self.register_buffer('weights', weights.float())
        
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 3) predictions
            target: (batch, 3) ground truth
        """
        squared_errors = (pred - target) ** 2
        weighted_errors = squared_errors * self.weights
        return weighted_errors.mean()

class MonocularCNN(nn.Module):
    """! @brief Single-stream CNN for processing monocular images and predicting grasp offset.

    Monocular counterpart of DualStreamCNN — uses a single image (left camera)
    instead of a stereo pair.  The feature extractor is identical; the fusion
    layers now take 256 features (instead of 256*2).
    """

    def __init__(self, input_channels=3, output_dim=3):
        super(MonocularCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regression_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: BatchedMonocularSample):
        features = self.feature_extractor(batch.batched_img)
        features = features.view(features.size(0), -1)
        return self.regression_head(features)


class EfficientNet(nn.Module):
    """! @brief Monocular model using a pre-trained EfficientNet-B0 as the feature extractor."""
    def __init__(self, output_dim=3, freeze_early_layers=True):
        super(EfficientNet, self).__init__()

        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )

        if freeze_early_layers:
            for ct, child in enumerate(self.feature_extractor[0].children()):
                if ct < 4:
                    for param in child.parameters():
                        param.requires_grad = False

        # Single-stream: 1280 features from one image
        self.regression_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: BatchedMonocularSample):
        features = self.feature_extractor(batch.batched_img)
        features = features.view(features.size(0), -1)
        return self.regression_head(features)


class ResNet(nn.Module):
    """! @brief Monocular model using a pre-trained ResNet as the feature extractor."""

    def __init__(self, output_dim=3, resnet_variant='resnet18', freeze_early_layers=True):
        super(ResNet, self).__init__()

        if resnet_variant == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
        elif resnet_variant == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feature_dim = 512
        elif resnet_variant == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool
        )

        # Single-stream: feature_dim features from one image
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: BatchedMonocularSample):
        features = self.feature_extractor(batch.batched_img)
        features = features.view(features.size(0), -1)
        return self.regression_head(features)


class CFCNetwork(nn.Module):
    """! @brief Visual servoing with NCP/CfC as an enhanced FC head (monocular).

    Uses Neural Circuit Policies (NCP) with Closed-form Continuous-time cells (CfC)
    for efficient and interpretable visual servoing control.
    Single-stream: CNN encodes one image into 128-d, fed directly into the NCP head.
    """

    def __init__(self, input_channels=3, output_dim=3, dropout_rate=0.5,
                 sparsity_level=0.5, ncp_mode='default'):
        super(CFCNetwork, self).__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.sparsity_level = sparsity_level
        self.ncp_mode = ncp_mode

        # 5-layer CNN backbone (identical to stereo version)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 11 * 11, 128)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Single-stream: 128 features from one image (was 128*2 for stereo)
        feature_dim = 128

        from ncps.torch import CfC
        from ncps.wirings import AutoNCP

        self.wiring = AutoNCP(
            units=feature_dim,
            output_size=output_dim,
            sparsity_level=sparsity_level
        )

        self.ncp_head = CfC(
            input_size=feature_dim,
            units=self.wiring,
            batch_first=True,
            return_sequences=False,
            mode=ncp_mode
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: BatchedMonocularSample):
        features = self.feature_extractor(batch.batched_img)
        features = self.flatten(features)
        features = self.fc(features)
        features = self.dropout(features)

        features = features.unsqueeze(1)  # [B, 128] -> [B, 1, 128]

        offset, _ = self.ncp_head(features)
        offset = offset.squeeze(1)  # [B, 1, 3] -> [B, 3]

        return offset

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_cnn_parameters(self):
        cnn_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        fc_params = sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return cnn_params + fc_params

    def get_num_ncp_parameters(self):
        return sum(p.numel() for p in self.ncp_head.parameters() if p.requires_grad)


# ── Backward-compatible aliases ──
# If an existing config YAML references "DualStreamCNN", map it to the
# monocular equivalent so no config changes are needed.
DualStreamCNN = MonocularCNN
ResNetDualStream = ResNet
