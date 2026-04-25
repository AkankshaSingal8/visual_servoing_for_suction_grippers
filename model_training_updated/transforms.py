from typing import Any

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from pathlib import Path
from tqdm import tqdm


class RandomBackgroundComposite(v2.Transform):
    def __init__(self, background_dir: str, preload: bool = True):
        super().__init__()
        self.background_dir = Path(background_dir)
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        self.background_paths = [
            p for p in self.background_dir.rglob("*")
            if p.suffix.lower() in valid_extensions
        ]
        if len(self.background_paths) == 0:
            raise ValueError(f"No background images found in {background_dir}")

        self.backgrounds_cache = None
        if preload:
            self.backgrounds_cache = []
            for bg_path in tqdm(self.background_paths, desc="Loading backgrounds"):
                self.backgrounds_cache.append(read_image(str(bg_path), mode=ImageReadMode.RGB))

    def make_params(self, flat_inputs):
        return {"bg_idx": torch.randint(0, len(self.background_paths), (1,)).item()}

    def transform(self, inpt, params):
        if not isinstance(inpt, torch.Tensor):
            return inpt
        if len(inpt.shape) < 4:
            return inpt

        bg_idx = params["bg_idx"]
        if self.backgrounds_cache is not None:
            background = self.backgrounds_cache[bg_idx].to(inpt)
        else:
            background = read_image(str(self.background_paths[bg_idx]), mode=ImageReadMode.RGB).to(inpt)

        if background.shape[-2:] != inpt.shape[-2:]:
            background = F.interpolate(
                background.unsqueeze(0).float(),
                size=inpt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(inpt.dtype)

        alpha = inpt[:, 3:4, ...].float() / 255.0
        rgb = inpt[:, :3, ...]
        composited = (rgb.float() * alpha + background.float() * (1 - alpha)).to(inpt.dtype)
        return tv_tensors.Image(composited)


class StripAlpha(v2.Transform):
    def transform(self, inpt, params):
        if not isinstance(inpt, torch.Tensor) or inpt.shape[-3] != 4:
            return inpt
        not_batched = len(inpt.shape) < 4
        if not_batched:
            inpt = inpt[None, ...]
        inpt = inpt[:, :3, ...]
        return tv_tensors.Image(inpt.squeeze(0) if not_batched else inpt)


def _coerce_dtype(value: Any):
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).replace("torch.", "")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid dtype '{value}'")
    return dtype


def _resolve_class(name: str):
    custom_transforms = {
        "RandomBackgroundComposite": RandomBackgroundComposite,
        "StripAlpha": StripAlpha,
    }
    return custom_transforms.get(name) or getattr(v2, name, None)


def _instantiate(spec: Any):
    name = spec["name"]
    args = spec.get("args", {})

    cls = _resolve_class(name)
    if cls is None:
        raise ValueError(f"Unknown transform '{name}'")

    if name == "ToDtype" and "dtype" in args:
        args["dtype"] = _coerce_dtype(args["dtype"])

    return cls(**args)


def create_transforms(config: dict[str, Any]) -> dict[str, v2.Compose]:
    tcfg = config["transforms"]
    train_specs = tcfg["train_pipeline"]
    eval_specs = tcfg["eval_pipeline"]
    return {
        "train": v2.Compose([_instantiate(spec) for spec in train_specs]),
        "eval": v2.Compose([_instantiate(spec) for spec in eval_specs]),
    }