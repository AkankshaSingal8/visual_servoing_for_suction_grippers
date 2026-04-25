import io
from typing import List
import os
import pdb
from zoneinfo import ZoneInfo
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.v2.functional as tvf
from tqdm import tqdm

from datastructs import EvalInfo, GraphInfo

_file = None
_log_path = None
def start(root_path: str):
    global _file, _log_path
    _log_path = os.path.join(
            root_path,
            f"{datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d_%I-%M-%S %p')}"
        )
    os.makedirs(_log_path, exist_ok=True)
    _file = open(
        os.path.join(_log_path, "log.txt"),
        "w"
    )

def fprint(*msg):
    global _file
    assert _file, "Didn't call diagnostics.start()!"
    timestamp = datetime.now(ZoneInfo('America/New_York')).strftime('%I-%M-%S %p')
    print(f"[[ {timestamp} ]] : ", *msg, file=_file, flush=True)

def to_log_path(path: os.PathLike):
    assert _log_path
    return Path(_log_path) / path

def annotate_(img: np.ndarray, locn: tuple[int, int], color: tuple[int, int, int]):
    img[:] = cv2.circle(img, locn, radius=5, color=color, thickness=5)

def draw_corners(img: np.ndarray, corners: np.ndarray):
    assert corners.shape == (4,2)
    for corner in corners:
        corner = tuple(map(int, corner))
        annotate_(img, corner, (255, 0, 0)) # type: ignore

def save_eval_info(*val_infos: EvalInfo):
    global _file, _log_path
    assert _file and _log_path
    assert all(isinstance(arg, EvalInfo) for arg in val_infos)

    for val_info in val_infos:
        with open(
            to_log_path(f"validation_epoch_{val_info.epoch}.txt"), "w"
            ) as f:
            f.write(f"Average validation loss: {val_info.avg_val_loss}\n")
            f.write(f"Validation losses: {', '.join(str(vl) for vl in val_info.val_losses)}\n")
            f.write(f"MAE X: {val_info.mae_x:.6f}, MAE Y: {val_info.mae_y:.6f}, MAE Z: {val_info.mae_z:.6f}\n")
            f.write(f"Euclidean distance: {val_info.euclidean_distances:.6f}\n")
            np.savetxt(
                to_log_path(f"val_losses_npy.txt"),
                val_info.val_losses
            )

            image_pil = Image.fromarray(val_info.performance_graph)
            image_pil.save(
                to_log_path(f"validation_performance_epoch_{val_info.epoch}.png")
            )

            for idx, loss_info in enumerate(val_info.abnormals):
                fprint(f"Abnormal idx: {loss_info.eval_idx} had distance = {loss_info.distance.item():.4f}")


def graph_eval_info(*val_infos: EvalInfo):
    """Graphs the val losses on the same plot."""
    fig, axs = plt.subplots(figsize=(19.2, 10.8), dpi=100, ncols=min(len(val_infos), 2))
    if not hasattr(axs, '__iter__'):
        axs = [axs]

    def _graph_one_info(info: EvalInfo, ax):
        ax.hist(info.val_losses, label=f"Epoch {info.epoch}")
        ax.set_xlabel("Validation losses")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Validation loss histogram (epoch {info.epoch})")

    fig.suptitle(f"Validation losses")

    for info, ax in zip(val_infos, axs):
        _graph_one_info(info, ax)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, format='raw')
    x_dim, y_dim = list(map(int, fig.canvas.get_width_height()))
    plt.close(fig)
    img_np = (
            np.frombuffer(buf.getvalue(), dtype=np.uint8)
            .reshape(y_dim, x_dim, 4)
            [..., :-1]
    )

    return GraphInfo(img_np, "Validation loss distributions")

def save_image(fname: os.PathLike | str, image: np.ndarray):
    cv2.imwrite(str(to_log_path(fname)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def stop():
    global _file
    assert _file, "Didn't call diagnostics.start()!"
    _file.close()
