#!/usr/bin/env python3
"""
offline_detector_ablation.py — Run all detector backends on the same raw.mp4
and compare centroid trajectories.

Usage:
    python full_system_pipeline/evaluation/offline_detector_ablation.py \
        --video runs/v2_proteinbar_servo12/raw.mp4 \
        --ref-image assets/objects/protein_bar.jpeg \
        --out-dir runs/ablation_detector_comparison/

    # Skip re-running; load pre-existing frames.jsonl from existing ablation dirs:
    python full_system_pipeline/evaluation/offline_detector_ablation.py \
        --video runs/v2_proteinbar_servo12/raw.mp4 \
        --ref-image assets/objects/protein_bar.jpeg \
        --out-dir runs/ablation_detector_comparison/ \
        --existing-dirs runs/ablation_sam3,runs/ablation_gdino,runs/ablation_gdino_sam2

Runs: sam3, gdino+sam2, gdino — each gets its own frames.jsonl.
Outputs: detector_comparison.png + per-detector frames.jsonl + summary.json

Note: This script tests *perception* only (detector backend selection), not
closed-loop control.  Raw.mp4 files were recorded with the robot moving, but the
robot does not move during offline replay.  Results reflect centroid estimate
quality only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PYTHON = sys.executable
_EXT = _ROOT / "full_system_pipeline" / "pipeline" / "servo_lastmile_v2_ext.py"

DETECTORS = ["sam3", "gdino+sam2", "gdino"]
COLORS: Dict[str, str] = {
    "sam3": "mediumseagreen",
    "gdino+sam2": "steelblue",
    "gdino": "darkorange",
}
IMAGE_CENTER = (640.0, 360.0)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _det_dirname(detector: str) -> str:
    """Filesystem-safe directory name for a detector slug."""
    return detector.replace("+", "_plus_")


def _run_detector(
    detector: str,
    video: Path,
    ref_image: Path,
    out_dir: Path,
    max_frames: Optional[int],
) -> tuple[Path, bool, str]:
    """
    Run servo_lastmile_v2_ext.py for one detector.

    Returns (jsonl_path, success, error_message).
    On failure the jsonl_path is still returned (may not exist).
    """
    det_dir = out_dir / _det_dirname(detector)
    det_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = det_dir / "frames.jsonl"

    cmd = [
        _PYTHON, str(_EXT),
        "--detector", detector,
        "--ref-image", str(ref_image),
        "--input-video", str(video),
        "--output-dir", str(det_dir),
        "--no-overlays",
        "--no-robot",
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]

    print(f"  Running detector={detector} ...", flush=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            err = result.stderr[-800:].strip()
            print(f"  FAILED ({detector}): returncode={result.returncode}", file=sys.stderr)
            print(f"  stderr tail:\n{err}", file=sys.stderr)
            return jsonl_path, False, err
        print(f"  OK ({detector})")
        return jsonl_path, True, ""
    except FileNotFoundError as exc:
        msg = f"Script not found: {_EXT} — {exc}"
        print(f"  FAILED ({detector}): {msg}", file=sys.stderr)
        return jsonl_path, False, msg
    except subprocess.TimeoutExpired:
        msg = "subprocess timed out after 300 s"
        print(f"  FAILED ({detector}): {msg}", file=sys.stderr)
        return jsonl_path, False, msg


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    frames: List[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return frames


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _centroid_err(c, ref: tuple = IMAGE_CENTER) -> Optional[float]:
    if c is None:
        return None
    return math.hypot(c[0] - ref[0], c[1] - ref[1])


def _detector_stats(frames: List[dict]) -> dict:
    errs = [
        _centroid_err(f.get("best_centroid"))
        for f in frames
        if f.get("best_centroid") is not None
    ]
    n_valid = len(errs)
    return {
        "n_frames": len(frames),
        "n_frames_with_centroid": n_valid,
        "mean_centroid_err_px": float(np.mean(errs)) if errs else None,
        "std_centroid_err_px": float(np.std(errs)) if len(errs) > 1 else None,
        "median_centroid_err_px": float(np.median(errs)) if errs else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_comparison(
    results: Dict[str, List[dict]],
    statuses: Dict[str, bool],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for detector, frames in results.items():
        color = COLORS.get(detector, "gray")
        available = statuses.get(detector, True)

        idxs: List[int] = []
        errs: List[float] = []
        xs: List[float] = []
        ys: List[float] = []

        for i, f in enumerate(frames):
            bc = f.get("best_centroid")
            if bc is not None:
                idxs.append(i)
                e = _centroid_err(bc)
                if e is not None:
                    errs.append(e)
                    xs.append(float(bc[0]))
                    ys.append(float(bc[1]))

        label_suffix = "" if available else " [unavailable — stub]"
        mean_str = f"{np.mean(errs):.1f}px" if errs else "N/A"

        kw = dict(color=color, linewidth=1.5, alpha=0.85)
        axes[0].plot(
            idxs, errs,
            label=f"{detector}{label_suffix} (mean={mean_str})",
            **kw,
        )
        axes[1].plot(idxs, xs, label=detector + label_suffix, **kw)
        axes[2].plot(idxs, ys, label=detector + label_suffix, **kw)

    axes[0].set_ylabel("|centroid − center| (px)")
    axes[0].set_title("Centroid Error vs Image Center — Detector Ablation")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("centroid X (px)")
    axes[1].axhline(IMAGE_CENTER[0], color="gray", linewidth=0.8, linestyle="--")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("centroid Y (px)")
    axes[2].axhline(IMAGE_CENTER[1], color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("frame index")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "detector_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(results: Dict[str, List[dict]]) -> None:
    print("\n=== Detector Comparison Summary ===")
    print(f"{'Detector':<16} {'N frames':>9} {'N w/centroid':>13} {'Mean err (px)':>14} {'Std err (px)':>13}")
    print("-" * 70)
    for detector, frames in results.items():
        stats = _detector_stats(frames)
        mean_e = f"{stats['mean_centroid_err_px']:.1f}" if stats["mean_centroid_err_px"] is not None else "N/A"
        std_e = f"{stats['std_centroid_err_px']:.1f}" if stats["std_centroid_err_px"] is not None else "N/A"
        print(
            f"{detector:<16} {stats['n_frames']:>9} "
            f"{stats['n_frames_with_centroid']:>13} "
            f"{mean_e:>14} {std_e:>13}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Path to raw.mp4 to process")
    parser.add_argument("--ref-image", required=True,
                        help="Path to reference object image")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for plots and per-detector results")
    parser.add_argument("--detectors", default=",".join(DETECTORS),
                        help="Comma-separated detectors to compare (default: sam3,gdino+sam2,gdino)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames processed per detector run")
    parser.add_argument(
        "--existing-dirs", default=None,
        help=(
            "Comma-separated paths to existing ablation dirs (one per detector, "
            "same order as --detectors). If given, frames.jsonl is loaded from these "
            "instead of re-running the pipeline."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    ref = Path(args.ref_image)
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    # Validate inputs
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        sys.exit(1)
    if not ref.exists():
        print(f"ERROR: ref image not found: {ref}", file=sys.stderr)
        sys.exit(1)

    # Build optional existing-dirs map
    existing_map: Dict[str, Path] = {}
    if args.existing_dirs:
        parts = [Path(p.strip()) for p in args.existing_dirs.split(",") if p.strip()]
        if len(parts) != len(detectors):
            print(
                f"ERROR: --existing-dirs has {len(parts)} entries but --detectors has {len(detectors)}",
                file=sys.stderr,
            )
            sys.exit(1)
        existing_map = dict(zip(detectors, parts))

    print(f"Ablation: {len(detectors)} detectors on {video.name}")
    print(f"Output:   {out_dir}\n")

    results: Dict[str, List[dict]] = {}
    statuses: Dict[str, bool] = {}
    run_metadata: Dict[str, dict] = {}

    for detector in detectors:
        print(f"--- {detector} ---")

        # If a pre-existing dir was supplied, use it directly
        if detector in existing_map:
            existing_jsonl = existing_map[detector] / "frames.jsonl"
            print(f"  Using existing frames.jsonl: {existing_jsonl}")
            frames = _load_jsonl(existing_jsonl)
            ok = len(frames) > 0
            err_msg = "" if ok else "frames.jsonl empty or missing"
            statuses[detector] = ok
            results[detector] = frames
            run_metadata[detector] = {
                "source": "existing",
                "jsonl_path": str(existing_jsonl),
                "success": ok,
                "error": err_msg,
                **_detector_stats(frames),
            }
            continue

        # Dry-run: just print the command
        if args.dry_run:
            det_dir = out_dir / _det_dirname(detector)
            cmd = [
                _PYTHON, str(_EXT),
                "--detector", detector,
                "--ref-image", str(ref),
                "--input-video", str(video),
                "--output-dir", str(det_dir),
                "--no-overlays", "--no-robot",
            ]
            if args.max_frames is not None:
                cmd += ["--max-frames", str(args.max_frames)]
            print(f"  [dry-run] Would run: {' '.join(cmd)}")
            results[detector] = []
            statuses[detector] = False
            run_metadata[detector] = {"source": "dry_run", "success": False}
            continue

        # Actually run the detector
        jsonl_path, ok, err_msg = _run_detector(
            detector, video, ref, out_dir, args.max_frames
        )
        frames = _load_jsonl(jsonl_path)

        if not ok or not frames:
            # Graceful degradation: write stub entry
            stub = {
                "detector": detector,
                "status": "unavailable",
                "reason": err_msg or "subprocess returned no frames",
            }
            stub_path = out_dir / _det_dirname(detector) / "stub.json"
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            stub_path.write_text(json.dumps(stub, indent=2))
            print(f"  Wrote stub: {stub_path}")

        statuses[detector] = ok
        results[detector] = frames
        run_metadata[detector] = {
            "source": "subprocess",
            "success": ok,
            "error": err_msg,
            **_detector_stats(frames),
        }

    # Print summary table
    _print_summary(results)

    # Generate comparison plot
    _plot_comparison(results, statuses, out_dir)

    # Write JSON summary
    summary = {
        "video": str(video),
        "ref_image": str(ref),
        "detectors": detectors,
        "image_center": list(IMAGE_CENTER),
        "per_detector": run_metadata,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
