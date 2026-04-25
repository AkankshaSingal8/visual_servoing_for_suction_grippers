
import os
from typing import Mapping, Any

import diagnostics
import utils

class Logger:
    def on_train_start(self, model) -> None: pass
    def on_epoch_end(self, checkpoint, epoch_info: Any, eval_info: Any) -> None: pass
    def close(self) -> None: pass


class CheckpointLogger(Logger):
    def __init__(self, run_name, dir, save_every: int):
        self._run_name = run_name
        self._dir = dir
        self._save_every = save_every
        self._log_path = f"{self._dir}/{self._run_name}"
        os.makedirs(self._log_path, exist_ok=True)

    def on_epoch_end(self, checkpoint, epoch_info, eval_info):
        if epoch_info.epoch % self._save_every == 0:
            utils.save_checkpoint(checkpoint, f"{self._dir}/{self._run_name}", f"{epoch_info.epoch}")

class WandbLogger(Logger):
    def __init__(
        self,
        identifier: str,
        project: str,
        entity: str,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was requested, but wandb is not installed. "
                "Remove logging.wandb from the config or install wandb."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=identifier,
        )

    def on_train_start(self, model) -> None:
        self._run.watch(model)

    def on_epoch_end(self, checkpoint, epoch_info: Any, eval_info: Any) -> None:
        step = epoch_info.epoch + 1
        self._run.log(
            {
                "epoch": step,
                "train_loss": epoch_info.avg_train_loss,
                "val_loss": eval_info.avg_val_loss,
                "learning_rate" : checkpoint.optimizer.param_groups[0]['lr'],
                "mae_x": eval_info.mae_x,
                "mae_y": eval_info.mae_y,
                "mae_z": eval_info.mae_z,
                "euclidean_distances": eval_info.euclidean_distances,
                "eval/performance_graph" : self._wandb.Image(eval_info.performance_graph)
            },
            step=step,
        )

    def close(self) -> None:
        self._run.finish()


class DiagnosticsLogger(Logger):
    def __init__(self, identifier: str, root_dir: str = "diagnostics"):
        self._identifier = identifier
        diagnostics.start(os.path.join(root_dir, identifier))

    def on_epoch_end(self, checkpoint, epoch_info: Any, eval_info: Any) -> None:
        diagnostics.fprint(
            f"Epoch [{epoch_info.epoch}] - "
            f"Train Loss: {epoch_info.avg_train_loss:.6f}, "
            f"Val Loss: {eval_info.avg_val_loss:.6f}, "
            f"MAE(x={eval_info.mae_x:.4f}, y={eval_info.mae_y:.4f}, z={eval_info.mae_z:.4f}), "
            f"Euclidean: {eval_info.euclidean_distances:.4f}"
        )

    def close(self) -> None:
        diagnostics.stop()
