#!/usr/bin/env python3
"""
signal_ablation.py — Replay frames.jsonl and compute fused centroids under
different signal subsets. Answers: "does fusion outperform any individual signal?"

Each ablation condition specifies which signals are INCLUDED (the rest are treated
as null before fusion). The fusion logic is re-applied from logged per-signal
values using the same weighted_geometric_median as the live pipeline.

Produces a bar chart of mean centroid error per condition across all NEAR frames.

Usage:
    python full_system_pipeline/evaluation/signal_ablation.py \
        --run-dir runs/v2_proteinbar_servo12 \
        --output-dir /tmp/signal_ablation_test

    # With GT centroid for error computation:
    python full_system_pipeline/evaluation/signal_ablation.py \
        --run-dir runs/v2_proteinbar_servo12 \
        --output-dir /tmp/signal_ablation_test \
        --gt-centroid 660 430
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the actual fusion function so we use the real implementation
from foundation_model.servo_lastmile import weighted_geometric_median  # noqa: E402

IMAGE_CENTER = (640.0, 360.0)

# Ablation conditions: name → set of signal keys to include, or None for canonical
ABLATION_CONDITIONS: dict[str, set[str] | None] = {
    "canonical":   None,        # use logged best_centroid unchanged
    "all_signals": {"c_A", "c_B", "c_D", "c_E"},
    "A_only":      {"c_A"},
    "B_only":      {"c_B"},
    "D_only":      {"c_D"},
    "E_only":      {"c_E"},
    "A_plus_B":    {"c_A", "c_B"},
    "A_plus_D":    {"c_A", "c_D"},
    "B_plus_D":    {"c_B", "c_D"},
    "no_A":        {"c_B", "c_D", "c_E"},
    "no_B":        {"c_A", "c_D", "c_E"},
}

CONDITION_ORDER = [
    "canonical",
    "all_signals",
    "A_only",
    "B_only",
    "D_only",
    "E_only",
    "A_plus_B",
    "A_plus_D",
    "B_plus_D",
    "no_A",
    "no_B",
]

COLORS = {
    "all_signals": "black",
    "A_only":      "mediumseagreen",
    "B_only":      "cyan",
    "D_only":      "gold",
    "E_only":      "darkorange",
    "A_plus_B":    "steelblue",
    "A_plus_D":    "limegreen",
    "B_plus_D":    "teal",
    "no_A":        "firebrick",
    "no_B":        "mediumpurple",
    "canonical":   "red",
}

DEFAULT_WEIGHT = 1.0


def _centroid_err(c: tuple[float, float] | list | None,
                  ref: tuple[float, float]) -> float | None:
    if c is None:
        return None
    return math.hypot(c[0] - ref[0], c[1] - ref[1])


def _load_frames(run_dir: Path) -> list[dict]:
    path = run_dir / "frames.jsonl"
    frames = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return frames


def _fuse_signals(frame: dict, signal_keys: set[str]) -> tuple[float, float] | None:
    """Re-compute fused centroid from a subset of logged signals with equal weights."""
    pts: list[tuple[float, float]] = []
    weights: list[float] = []
    for key in sorted(signal_keys):  # sorted for determinism
        val = frame.get(key)
        if val is not None:
            pts.append((float(val[0]), float(val[1])))
            weights.append(DEFAULT_WEIGHT)

    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]

    fused = weighted_geometric_median(pts, weights)
    if fused is None:
        return pts[0]
    return (float(fused[0]), float(fused[1]))


def compute_near_errors(frames: list[dict],
                        signal_keys: set[str] | None,
                        ref: tuple[float, float]) -> list[float]:
    """
    For each NEAR frame, compute centroid error from ref.
    Returns list of errors (non-null only — frames with no valid signal are skipped).
    """
    errors: list[float] = []
    for frame in frames:
        if frame.get("state") != "NEAR":
            continue

        if signal_keys is None:
            # canonical: use logged best_centroid
            c = frame.get("best_centroid")
        else:
            c = _fuse_signals(frame, signal_keys)

        err = _centroid_err(c, ref)
        if err is not None:
            errors.append(err)
    return errors


def _find_signal_availability(frames: list[dict]) -> dict[str, int]:
    """Count how many NEAR frames have each signal populated."""
    counts: dict[str, int] = {k: 0 for k in ("c_A", "c_B", "c_C", "c_D", "c_E")}
    near_total = 0
    for f in frames:
        if f.get("state") != "NEAR":
            continue
        near_total += 1
        for k in counts:
            if f.get(k) is not None:
                counts[k] += 1
    counts["_near_total"] = near_total
    return counts


def _plot_bar(condition_errors: dict[str, list[float]], out_dir: Path) -> None:
    """Bar chart: mean centroid error (px) per ablation condition."""
    ordered = [c for c in CONDITION_ORDER if c in condition_errors]

    means = []
    stds = []
    labels = []
    colors = []
    for cond in ordered:
        errs = condition_errors[cond]
        if errs:
            means.append(float(np.mean(errs)))
            stds.append(float(np.std(errs)))
        else:
            means.append(float("nan"))
            stds.append(0.0)
        labels.append(cond)
        colors.append(COLORS.get(cond, "gray"))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=4, linewidth=1.2)

    # Annotate bars with mean value
    for bar, mean in zip(bars, means):
        if not math.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2.0, mean + 0.5,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean centroid error from IMAGE_CENTER (px)", fontsize=11)
    ax.set_title("Signal Ablation: Mean Centroid Error by Fusion Condition\n"
                 "(NEAR frames only; error bars = 1 std dev)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "signal_ablation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _print_summary(condition_errors: dict[str, list[float]]) -> None:
    print("\n=== Signal Ablation Summary (NEAR frames only) ===")
    print(f"{'Condition':<16} {'N frames':>8} {'Mean err (px)':>14} {'Std':>8} {'Min':>8}")
    print("-" * 60)
    rows = []
    for cond in CONDITION_ORDER:
        errs = condition_errors.get(cond, [])
        if errs:
            rows.append((cond, len(errs), np.mean(errs), np.std(errs), np.min(errs)))
        else:
            rows.append((cond, 0, float("nan"), 0.0, float("nan")))
    rows.sort(key=lambda r: (math.isnan(r[2]), r[2]))
    for cond, n, mean, std, mn in rows:
        mean_s = f"{mean:>14.1f}" if not math.isnan(mean) else f"{'N/A':>14}"
        mn_s = f"{mn:>8.1f}" if not math.isnan(mn) else f"{'N/A':>8}"
        print(f"{cond:<16} {n:>8} {mean_s} {std:>8.1f} {mn_s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to a run directory containing frames.jsonl")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots and summary JSON")
    parser.add_argument("--gt-centroid", nargs=2, type=float, metavar=("X", "Y"),
                        help="Ground-truth centroid to measure error against "
                             "(default: IMAGE_CENTER = 640 360)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "signal_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref: tuple[float, float] = (
        (float(args.gt_centroid[0]), float(args.gt_centroid[1]))
        if args.gt_centroid else IMAGE_CENTER
    )

    print(f"Loading frames from: {run_dir / 'frames.jsonl'}")
    frames = _load_frames(run_dir)
    print(f"Total frames loaded: {len(frames)}")

    avail = _find_signal_availability(frames)
    print(f"\nNEAR frames: {avail['_near_total']}")
    for sig in ("c_A", "c_B", "c_C", "c_D", "c_E"):
        print(f"  {sig} populated: {avail[sig]} / {avail['_near_total']}")

    condition_errors: dict[str, list[float]] = {}
    for cond, signal_keys in ABLATION_CONDITIONS.items():
        errs = compute_near_errors(frames, signal_keys, ref)
        condition_errors[cond] = errs

    _print_summary(condition_errors)
    _plot_bar(condition_errors, out_dir)

    # Save JSON summary
    summary: dict[str, dict] = {}
    for cond, errs in condition_errors.items():
        summary[cond] = {
            "n_valid_frames": len(errs),
            "mean_err_px": float(np.mean(errs)) if errs else None,
            "std_err_px":  float(np.std(errs))  if errs else None,
            "min_err_px":  float(np.min(errs))  if errs else None,
        }
    json_path = out_dir / "signal_ablation_summary.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
