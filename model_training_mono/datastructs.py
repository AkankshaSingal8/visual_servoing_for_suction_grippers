from dataclasses import dataclass, fields, field, asdict
import os
import pdb
from typing import List, Optional, Callable, Any, Dict, Tuple
from pathlib import Path

import numpy as np
from torchvision import tv_tensors
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TVF
import cv2


@dataclass
class Directories:
    diagnostics_dir: str
    checkpoint_dir: str

@dataclass
class Checkpoint:
    model: torch.nn.Module
    scheduler: torch.optim.lr_scheduler.LRScheduler
    optimizer: torch.optim.Optimizer
    epoch: int = 0
    train_loss: float = float('inf')
    val_loss: float = float('inf')

    def todict(self):
        return {field.name : getattr(self, field.name) for field in fields(self)} | dict(model=self.model.state_dict(), scheduler=self.scheduler.state_dict(), optimizer=self.optimizer.state_dict())

@dataclass
class Infrastructure:
    train_transforms: v2.Compose
    val_transforms: v2.Compose
    criterion: torch.nn.Module
    valloader: torch.utils.data.DataLoader
    valset: torch.utils.data.Dataset
    trainloader: torch.utils.data.DataLoader
    trainset: torch.utils.data.Dataset

@dataclass
class MonocularSample:
    img: tv_tensors.Image
    offset: torch.Tensor
    original_image_path: str

    fields_are_tv_tensors: tuple[str, ...] = ('img',)

    def __post_init__(self):
        for member in self.fields_are_tv_tensors:
            assert isinstance(getattr(self, member), torch.Tensor)
            setattr(self, member, tv_tensors.Image(getattr(self, member)))

    @staticmethod
    def collate(samples: List["MonocularSample"]):
        return BatchedMonocularSample(*samples)

    def transform(self, transforms: v2.Compose | Callable[[torch.Tensor], torch.Tensor]):
        """
        transforms MUST take one image as parameter, AND take arbitrary batch dimensions.
        """
        with tv_tensors.set_return_type("TVTensor"):
            self.img = transforms(self.img)
        return self

    def move_to(self, device: str):
        for f in fields(self):
            if isinstance(getattr(self, f.name), torch.Tensor):
                setattr(self, f.name, getattr(self, f.name).to(device))
        return self


class BatchedMonocularSample:
    def __init__(self, *monocular_samples: MonocularSample):
        for f in fields(MonocularSample):
            values = [getattr(s, f.name) for s in monocular_samples]
            if f.type is tv_tensors.Image:
                setattr(self, f"batched_{f.name}", tv_tensors.Image(torch.stack(values)))
            elif f.type is torch.Tensor:
                setattr(self, f"batched_{f.name}", torch.stack(values))
            else:
                setattr(self, f"batched_{f.name}", values)

    def new_tensor(self, arr_like):
        tensor_member = next(v for v in self.__dict__.values() if isinstance(v, torch.Tensor))
        return tensor_member.new_tensor(arr_like)

    def transform(self, transform: v2.Compose):
        with tv_tensors.set_return_type("TVTensor"):
            self.batched_img = transform(self.batched_img)
        return self

    def move_to(self, device: str):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


# ── Keep StereoSample and BatchedStereoSample as aliases for backward compat ──
# (e.g. CaptureResult.stereo_sample() used at inference time)

@dataclass
class StereoSample:
    """Retained for inference-time compatibility (CaptureResult)."""
    left_img: tv_tensors.Image
    right_img: tv_tensors.Image
    offset: torch.Tensor
    original_image_path: tuple[str, str]

    fields_are_tv_tensors: tuple[str, str] = ('left_img', 'right_img')

    def __post_init__(self):
        for member in self.fields_are_tv_tensors:
            assert isinstance(getattr(self, member), torch.Tensor)
            setattr(self, member, tv_tensors.Image(getattr(self, member)))

    def to_monocular(self) -> MonocularSample:
        """Convert to MonocularSample using only the left image."""
        return MonocularSample(
            img=self.left_img,
            offset=self.offset,
            original_image_path=self.original_image_path[0]
        )

    def move_to(self, device: str):
        for f in fields(self):
            if isinstance(getattr(self, f.name), torch.Tensor):
                setattr(self, f.name, getattr(self, f.name).to(device))
        return self


@dataclass
class EpochInfo:
    epoch: int
    avg_train_loss: float

@dataclass
class GraphInfo:
    graph_buf: np.ndarray
    name: str

@dataclass
class LossInfoGpu:
    eval_idx: int
    distance: torch.Tensor
    deviation: torch.Tensor
    loss: torch.Tensor
    predicted: torch.Tensor
    gt: torch.Tensor
    original_image_path: str

@dataclass
class EvalInfo:
    epoch: int
    avg_val_loss: float
    val_losses: np.ndarray
    performance_graph: np.ndarray
    abnormals: List[LossInfoGpu]
    loss_infos: List[LossInfoGpu]
    mae_x: float = 0.0
    mae_y: float = 0.0
    mae_z: float = 0.0
    euclidean_distances: float = 0.0

@dataclass
class DLTSolution:
    homography: torch.Tensor
    solution_cost: float


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class CaptureResult:
    """Result of a single capture operation."""
    position_index: int
    timestamp: float
    image_left: np.ndarray | torch.Tensor
    image_right: np.ndarray | torch.Tensor
    depth_image: Optional[np.ndarray | torch.Tensor]
    depth_map: Optional[np.ndarray | torch.Tensor]
    success: bool
    error_message: Optional[str]

    def to(self, device: str):
        self.image_left = self.image_left.to(device) #type: ignore
        self.image_right = self.image_right.to(device) #type: ignore
        self.depth_image = self.depth_image.to(device) #type: ignore
        self.depth_map = self.depth_map.to(device) #type: ignore
        return self

    def torch(self):
        if self.image_left.shape[-1] == 4:
            self.image_left = cv2.cvtColor(self.image_left, cv2.COLOR_RGBA2RGB) # type: ignore
            self.image_right = cv2.cvtColor(self.image_right, cv2.COLOR_RGBA2RGB) # type: ignore

        self.image_left = TVF.to_image(self.image_left) #type: ignore
        self.image_right = TVF.to_image(self.image_right)#type: ignore
        self.depth_image = torch.from_numpy(self.depth_image)
        self.depth_map = torch.from_numpy(self.depth_map)
        return self

    def monocular_sample(self) -> MonocularSample:
        """Create a MonocularSample from the left image."""
        return MonocularSample(self.image_left, self.depth_map, "") #type: ignore

    def stereo_sample(self):
        return StereoSample(self.image_left, self.image_right, self.depth_map, ("", "")) #type: ignore

@dataclass
class CaptureSequence:
    """
    Tracks state of multi-capture sequence.
    Enables resume-on-failure without losing progress.
    """
    num_captures: int
    completed_captures: Dict[int, CaptureResult] = field(default_factory=dict)
    failed_captures: List[int] = field(default_factory=list)
    current_index: int = 0
    max_retries_per_capture: int = 3

    def is_complete(self) -> bool:
        """Check if sequence is complete."""
        return self.current_index >= self.num_captures

    def get_progress(self) -> Tuple[int, int]:
        """Return (completed, total) count."""
        return len(self.completed_captures), self.num_captures

    def mark_capture_complete(self, index: int, result: CaptureResult):
        """Mark a capture as successfully completed."""
        self.completed_captures[index] = result

    def mark_capture_failed(self, index: int):
        """Mark a capture as failed after max retries."""
        self.failed_captures.append(index)
