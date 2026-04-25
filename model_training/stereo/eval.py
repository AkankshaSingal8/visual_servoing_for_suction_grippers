"""! @file eval.py

@brief Evaluation script for analyzing the performance of trained visual servoing models.
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
import pdb
import os
import io
from typing import List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import tyro
from torchvision.io import read_image, ImageReadMode

from datastructs import EvalInfo, LossInfoGpu, StereoSample, BatchedStereoSample
import utils
import trainutils


def plot_results(ground_truths: np.ndarray, predictions: np.ndarray, output_filename="evaluation_plots.png", img_only=False):
    errors = predictions - ground_truths
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    fig.suptitle('Model Performance Analysis', fontsize=16)

    axes_labels = ['X', 'Y', 'Z']

    for i in range(3):
        ax = axes[0, i]
        ax.scatter(ground_truths[:, i], predictions[:, i], alpha=0.6, s=10)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')

        ax.set_xlabel(f"Ground Truth {axes_labels[i]} (m)")
        ax.set_ylabel(f"Predicted {axes_labels[i]} (m)")
        ax.set_title(f"Ground Truth vs. Prediction ({axes_labels[i]}-axis)")
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

    for i in range(3):
        ax = axes[1, i]
        N = len(errors[:, i])
        weights = np.full(N, 100.0 / N)
        ax.hist(errors[:, i], range=(-0.1, 0.1), bins=30, weights=weights, alpha=0.75)
        ax.axvline(0, linestyle='--', label='Zero Error', color=(0, 1.0, 0), linewidth=4)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlabel(f"Prediction Error (m)")
        ax.set_ylabel("Percentage of Samples (%)")
        ax.set_title(f"Error Distribution ({axes_labels[i]}-axis) (higher is overshoot)")
        ax.grid(True)
        ax.margins(x=0)
        ax.legend()
    
    for i in range(3):
        ax = axes[2, i]
        hist, edges = np.histogram(errors[:, i], bins=100, range=(-0.1, 0.1), density=False)
        xp = np.minimum(edges + (edges[1] - edges[0])/2, edges.max())[:-1]
        x = np.linspace(xp.min(), xp.max(), num=1920//2)
        hist_interp = np.interp(x, xp, hist)
        extent = (edges.min() - (edges[1] - edges[0])/2, edges.max() + (edges[1] - edges[0])/2, 0 ,1)
        ax.imshow(hist_interp[None, :], cmap="plasma", extent=extent, aspect=0.05)
        ax.set_xlabel(f"Prediction Error (m)")
        ax.set_yticks([])
        ax.axvline(0, linestyle='--', label='Zero Error', color=(0, 1.0, 0), linewidth=4)
        ax.set_title(f"Error heatmap.")
        ax.set_xlim(axes[1, i].get_xlim())


    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    if not img_only:
        plt.savefig(output_filename)
        # plt.show()
        print(f"\nSaved evaluation plots to '{output_filename}'")

    buf = io.BytesIO()
    x_dim, y_dim = list(map(int, fig.bbox.bounds))[-2:]
    fig.savefig(buf, dpi=100, format="raw")
    plt.close(fig)
    return np.frombuffer(
        buf.getvalue(),
        dtype=np.uint8
    ).reshape(y_dim, x_dim, 4)[..., :-1]


def calculate_loss_infos(batched_pred, batched_stereo_sample: BatchedStereoSample, criterion, batch_idx) -> List[LossInfoGpu]:
    loss = criterion(batched_pred, batched_stereo_sample.batched_offset)
    batched_deviation = torch.abs(batched_stereo_sample.batched_offset - batched_pred)
    batched_distance = torch.linalg.norm(batched_stereo_sample.batched_offset - batched_pred, dim=-1)

    return [
        LossInfoGpu(
            batch_idx,
            distance,
            deviation,
            loss,
            pred,
            offset,
            image_paths,
        )
        for distance, deviation, pred, offset, image_paths in zip(
            batched_distance, 
            batched_deviation, 
            batched_pred, 
            batched_stereo_sample.batched_offset, 
            batched_stereo_sample.batched_original_image_path
        )
    ]


def is_abnormal(loss_info: LossInfoGpu, thresholds=[0.3, 0.02, 0.02]):
    """Check if a prediction deviates abnormally from ground truth."""
    return torch.any(loss_info.deviation > loss_info.deviation.new_tensor(thresholds))


def eval_model(model, criterion, val_set, val_transforms, epoch):
    loss_infos = []

    assert len(val_set) > 0, "Val set empty??"
    model.eval()

    with torch.no_grad():
        for idx, batched_stereo_sample in enumerate(val_set):
            batched_stereo_sample.move_to(utils.get_model_device(model))
            batched_stereo_sample.transform(val_transforms)
            model_output = model(batched_stereo_sample)

            batch_loss_info = calculate_loss_infos(model_output, batched_stereo_sample, criterion, idx)

            loss_infos += batch_loss_info


    val_losses = np.array([li.loss.cpu().item() for li in loss_infos])
    avg_val_loss = float(np.mean(val_losses))

    abnormals = list(filter(is_abnormal, loss_infos))
    ground_truths = np.array([li.gt.cpu().numpy() for li in loss_infos])
    predictions = np.array([li.predicted.cpu().numpy() for li in loss_infos])
    performance_graph = plot_results(ground_truths, predictions, img_only=True)

    # Compute MAE metrics per axis and euclidean distance
    errors = np.abs(predictions - ground_truths)
    mae_x = float(np.mean(errors[:, 0]))
    mae_y = float(np.mean(errors[:, 1]))
    mae_z = float(np.mean(errors[:, 2]))
    euclidean_distances = float(np.mean(np.linalg.norm(predictions - ground_truths, axis=1)))

    return EvalInfo(
        epoch=epoch,
        avg_val_loss=avg_val_loss,
        val_losses=val_losses,
        performance_graph=performance_graph,
        abnormals=abnormals,
        loss_infos=loss_infos,
        mae_x=mae_x,
        mae_y=mae_y,
        mae_z=mae_z,
        euclidean_distances=euclidean_distances,
    )


@dataclass
class EvalConfig:
    config_path: str
    save_path: str
    checkpoint_path: Optional[str] = None

if __name__ == "__main__":

    eval_config = tyro.cli(EvalConfig)
    assert os.path.exists(eval_config.config_path)
    save_dir = os.path.dirname(eval_config.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    config = utils.load_config(eval_config.config_path)
    infra = trainutils.prepare_infrastructure(config)
    model = trainutils.create_model(config)
    if eval_config.checkpoint_path:
        checkpoint = torch.load(eval_config.checkpoint_path, map_location=config["device"])
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
    eval_info = eval_model(model, infra.criterion, infra.valloader, infra.val_transforms, 0)
    Image.fromarray(eval_info.performance_graph).save(eval_config.save_path)
