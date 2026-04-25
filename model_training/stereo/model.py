import pdb
import sys
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from datastructs import BatchedStereoSample

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

class DualStreamCNN(nn.Module):
    """! @brief Dual-stream CNN for processing stereo images and predicting offset.
    
    Siamese-like network because it has two identical processing paths for left & right images
    Core Idea: 
        1. Same feature extractor can find important visual cues (edges, textures, shapes)
        2. By using the same weights (for both images) -> network is forced to learn a consistent representation
    """
    
    def __init__(self, input_channels=3, output_dim=3):
        super(DualStreamCNN, self).__init__()
        
        #! @note Shared feature extractor for both streams
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3), # Convolutional Layer
            nn.BatchNorm2d(64), # Batch Normalization
            nn.ReLU(inplace=True), # Rectified Linear Unit
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Pooling -> Down-sampling Layer
            
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
            nn.AdaptiveAvgPool2d((1, 1)) # Final Pooling Layer -> Average the feature map to 1x1
        )
        
        """! @note Fusion and regression layers
        Once we have feature vectors for the left and right images, we need to combine them and predict the offset.
        """
        self.fusion_layers = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim) #! @note `output_dim` -> (x, y, z offsets)
        )
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: BatchedStereoSample):
        left_features = self.feature_extractor(batch.batched_left_img)
        right_features = self.feature_extractor(batch.batched_right_img)

        left_features = left_features.view(left_features.size(0), -1)
        right_features = right_features.view(right_features.size(0), -1)

        combined_features = torch.cat([left_features, right_features], dim=1)
        return self.fusion_layers(combined_features)

class EfficientNet(nn.Module):
    """! @brief A Dual-Stream model using a pre-trained EfficientNet-B0 as the feature extractor."""
    def __init__(self, output_dim=3, freeze_early_layers=True):
        super(EfficientNet, self).__init__()

        """! @note 1. Load a pre-trained EfficientNet-B0 model with default weights."""
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        """! @note 2. Define the feature extractor.
            .features contains the convolutional blocks.
            .avgpool performs the final pooling to create a feature vector.
        """
        self.feature_extractor = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )
        
        """! @note 3. Freeze the weights of the early convolutional blocks."""
        if freeze_early_layers:
            for ct, child in enumerate(self.feature_extractor[0].children()):
                if ct < 4:
                    for param in child.parameters():
                        param.requires_grad = False
        
        """! @note 4. Define the fusion and regression head.
            The output of EfficientNet-B0's feature extractor is 1280.
            We use a fusion strategy of concatenating left, right, and their difference.
        """
        fusion_input_features = 1280 * 3
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_features, 512),
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

    def forward(self, batch: BatchedStereoSample):
        left_features = self.feature_extractor(batch.batched_left_img)
        right_features = self.feature_extractor(batch.batched_right_img)

        left_features = left_features.view(left_features.size(0), -1)
        right_features = right_features.view(right_features.size(0), -1)

        diff_features = right_features - left_features
        combined_features = torch.cat([left_features, right_features, diff_features], dim=1)
        return self.fusion_layers(combined_features)

class ResNetDualStream(nn.Module):
   
    def __init__(self, output_dim=3, resnet_variant='resnet18', freeze_early_layers=True):
        super(ResNetDualStream, self).__init__()
        
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
        
        fusion_input_features = feature_dim * 3
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_features, 512),
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

    def forward(self, batch: BatchedStereoSample):
        left_features = self.feature_extractor(batch.batched_left_img)
        right_features = self.feature_extractor(batch.batched_right_img)

        left_features = left_features.view(left_features.size(0), -1)
        right_features = right_features.view(right_features.size(0), -1)

        diff_features = right_features - left_features
        combined_features = torch.cat([left_features, right_features, diff_features], dim=1)
        return self.fusion_layers(combined_features)


class CFCNetwork(nn.Module):
    """! @brief Visual servoing with NCP/CfC as an enhanced FC head
    
    Uses Neural Circuit Policies (NCP) with Closed-form Continuous-time cells (CfC)
    for efficient and interpretable visual servoing control.
    """
    
    def __init__(self, input_channels=3, output_dim=3, dropout_rate=0.5, 
                 sparsity_level=0.5, ncp_mode='default'):
        super(CFCNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.sparsity_level = sparsity_level
        self.ncp_mode = ncp_mode
        
        # 5-layer CNN backbone
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
        
        combined_dim = 128 * 2  # 256
        
        # Lazy import to avoid hard dependency when not using this model
        from ncps.torch import CfC
        from ncps.wirings import AutoNCP
        
        self.wiring = AutoNCP(
            units=combined_dim,
            output_size=output_dim,
            sparsity_level=sparsity_level
        )
        
        self.ncp_head = CfC(
            input_size=combined_dim,
            units=self.wiring,
            batch_first=True,
            return_sequences=False,
            mode=ncp_mode
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: BatchedStereoSample):
        left_features = self.feature_extractor(batch.batched_left_img)
        left_features = self.flatten(left_features)
        left_features = self.fc(left_features)
        left_features = self.dropout(left_features)
        
        right_features = self.feature_extractor(batch.batched_right_img)
        right_features = self.flatten(right_features)
        right_features = self.fc(right_features)
        right_features = self.dropout(right_features)
        
        combined = torch.cat([left_features, right_features], dim=1)
        combined = combined.unsqueeze(1)  # [B, 256] -> [B, 1, 256]
        
        offset, _ = self.ncp_head(combined)
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
