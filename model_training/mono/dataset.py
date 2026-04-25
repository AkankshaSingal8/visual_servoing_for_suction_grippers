import os
import re
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from pathlib import Path

from datastructs import MonocularSample


class MonocularImageDataset(Dataset):
    """Monocular dataset returning MonocularSample with raw RGBA left images."""

    __slots__ = ['dataset_path', 'samples', 'total_size']

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.samples = []

        self._load_dataset_info()

        self.total_size = len(self.samples)

    def _load_dataset_info(self):
        index_dir = os.path.join(self.dataset_path, "index")
        render_dir = os.path.join(self.dataset_path, "render")
        json_files = sorted([f for f in os.listdir(index_dir)
                            if f.startswith('index') and f.endswith('.json')])

        for json_file in json_files:
            regex_match = re.search(r"index([0-9]+)", json_file)
            if not regex_match:
                raise ValueError(f"{json_file} does not match pattern index[0-9]+")
            dataset_num = regex_match.group(1)
            png_dir = os.path.join(render_dir, f'render{dataset_num}')

            if not os.path.exists(png_dir):
                raise FileNotFoundError(f"{png_dir} not found.")

            with open(os.path.join(index_dir, json_file), 'r') as f:
                data = json.load(f)

            for frame in data['key_frames']:
                idx = frame['index']
                offset = np.array(frame['offset'], dtype=np.float32)

                index_str = f"{idx:04d}"
                left_path = os.path.join(png_dir, 'gripper_left_rgb', f'rgb_{index_str}.png')

                if os.path.exists(left_path):
                    self.samples.append((left_path, offset))
                else:
                    raise FileNotFoundError(f"Missing: {left_path}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        left_path, offset = self.samples[idx]

        img = read_image(left_path, mode=ImageReadMode.RGB_ALPHA)

        return MonocularSample(
            img=img,
            offset=torch.from_numpy(offset),
            original_image_path=left_path
        )
