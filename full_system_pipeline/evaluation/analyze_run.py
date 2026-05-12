"""
analyze_run.py — Offline analysis and plotting for visual servoing runs.

Usage:
    python analyze_run.py runs/v2_servo10/ [runs/v2_servo11/ ...] \\
        [--out-dir plots/] \\
        [--gt-centroid X Y] \\
        [--gt-pose-json PATH]

Each run directory must contain a frames.jsonl file (one JSON object per line).
Outputs are written to --out-dir (default: <first_run_dir>/analysis/).

Outputs:
    error_over_distance.png   centroid error vs z_mm, colour-coded by state
    signal_centroids.png      X/Y per signal over frame index
    tilt_timeline.png         tilt/roll/pitch vs z_mm
    phase_summary.png         stacked bar of frame counts per phase
    signal_dominance.png      stacked area: which signals are active per frame
    summary.json              key statistics (printed to stdout as well)
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


IMAGE_W, IMAGE_H = 1280, 720
IMAGE_CENTER = (IMAGE_W / 2, IMAGE_H / 2)

STATE_COLORS = {
    "FAR": "steelblue",
    "NEAR-sam3": "mediumseagreen",
    "NEAR-track": "darkorange",
    "TERMINAL": "crimson",
}

SIGNAL_COLORS = {
    "A": "mediumseagreen",
    "B": "cyan",
    "C": "magenta",
    "D": "gold",
    "E": "darkorange",
    "best": "red",
}


def _phase_label(frame):
    state = frame.get("state", "FAR")
    if state != "NEAR":
        return state
    near_phase = frame.get("_near_phase") or ""
    if near_phase == "track":
        return "NEAR-track"
    return "NEAR-sam3"


def _centroid_err(centroid, ref):
    if centroid is None:
        return None
    return math.hypot(centroid[0] - ref[0], centroid[1] - ref[1])


def load_frames(run_dir):
    path = os.path.join(run_dir, "frames.jsonl")
    frames = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_over_distance(all_frames_list, out_dir, ref_centroid, gt_centroid=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    for frames in all_frames_list:
        for frame in frames:
            bc = frame.get("best_centroid")
            z = frame.get("z_mm")
            if bc is None or z is None:
                continue
            err = _centroid_err(bc, ref_centroid)
            label = _phase_label(frame)
            color = STATE_COLORS.get(label, "gray")
            ax.scatter(z, err, color=color, alpha=0.6, s=18, zorder=3)

    if gt_centroid is not None:
        gt_err = _centroid_err(gt_centroid, ref_centroid)
        ax.axhline(gt_err, color="purple", linestyle="--", linewidth=1.2,
                   label=f"GT centroid err ({gt_err:.1f} px)")

    legend_patches = [
        mpatches.Patch(color=c, label=l) for l, c in STATE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("z_mm (approach: right → left)")
    ax.set_ylabel("|best_centroid − image_center| (px)")
    ax.set_title("Centroid Error vs Distance")
    ax.grid(True, alpha=0.3)

    _save(fig, out_dir, "error_over_distance.png")


def plot_signal_centroids(all_frames_list, out_dir):
    signals = ["A", "B", "C", "D", "E", "best"]
    fig, axes = plt.subplots(2, len(signals), figsize=(18, 6), sharey=False)

    for frames in all_frames_list:
        for col, sig in enumerate(signals):
            key = f"c_{sig}" if sig != "best" else "best_centroid"
            color = SIGNAL_COLORS[sig]
            xs, ys, idxs = [], [], []
            for i, frame in enumerate(frames):
                val = frame.get(key)
                if val is not None:
                    xs.append(val[0])
                    ys.append(val[1])
                    idxs.append(i)
            if idxs:
                axes[0, col].plot(idxs, xs, color=color, linewidth=1.2, alpha=0.8)
                axes[1, col].plot(idxs, ys, color=color, linewidth=1.2, alpha=0.8)

        for col, sig in enumerate(signals):
            axes[0, col].set_title(sig, color=SIGNAL_COLORS[sig])
            axes[0, col].set_ylabel("X (px)" if col == 0 else "")
            axes[1, col].set_ylabel("Y (px)" if col == 0 else "")
            axes[1, col].set_xlabel("frame")
            for row in range(2):
                axes[row, col].grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("X (px)")
    axes[1, 0].set_ylabel("Y (px)")
    fig.suptitle("Signal Centroids over Time", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "signal_centroids.png")


def plot_tilt_timeline(all_frames_list, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for frames in all_frames_list:
        zs, tilts, rolls, pitchs = [], [], [], []
        for frame in frames:
            z = frame.get("z_mm")
            t = frame.get("tilt_deg")
            r = frame.get("roll_deg")
            p = frame.get("pitch_deg")
            if z is not None:
                if t is not None:
                    zs.append(z); tilts.append(t)
                if r is not None:
                    rolls.append((z, r))
                if p is not None:
                    pitchs.append((z, p))

        if tilts:
            ax.plot(zs, tilts, color="royalblue", linewidth=1.5, alpha=0.8,
                    linestyle="-", label="tilt_deg")
        if rolls:
            rzs, rvs = zip(*rolls)
            ax.plot(rzs, rvs, color="orangered", linewidth=1.5, alpha=0.8,
                    linestyle="--", label="roll_deg")
        if pitchs:
            pzs, pvs = zip(*pitchs)
            ax.plot(pzs, pvs, color="mediumorchid", linewidth=1.5, alpha=0.8,
                    linestyle=":", label="pitch_deg")

    for threshold, color in [(2, "green"), (5, "orange")]:
        ax.axhline(threshold, color=color, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(-threshold, color=color, linewidth=0.8, linestyle="--", alpha=0.7)

    ax.invert_xaxis()
    ax.set_xlabel("z_mm (approach: right → left)")
    ax.set_ylabel("degrees")
    ax.set_title("Tilt / Roll / Pitch Timeline")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "tilt_timeline.png")


def plot_phase_summary(all_frames_list, out_dir):
    phase_order = ["FAR", "NEAR-sam3", "NEAR-track", "TERMINAL"]
    counts = {p: 0 for p in phase_order}
    for frames in all_frames_list:
        for frame in frames:
            label = _phase_label(frame)
            if label in counts:
                counts[label] += 1

    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = 0
    for phase in phase_order:
        count = counts[phase]
        ax.bar("frames", count, bottom=bottom,
               color=STATE_COLORS[phase], label=f"{phase} ({count})")
        bottom += count

    ax.set_ylabel("frame count")
    ax.set_title("Phase Summary")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir, "phase_summary.png")


def plot_signal_dominance(all_frames_list, out_dir):
    signals = ["A", "B", "C", "D", "E"]
    keys = [f"c_{s}" for s in signals]

    all_presence = {s: [] for s in signals}
    max_len = max(len(frames) for frames in all_frames_list)

    combined = []
    for frames in all_frames_list:
        combined.extend(frames)

    frame_indices = list(range(len(combined)))
    presence = {s: [] for s in signals}
    for frame in combined:
        for sig, key in zip(signals, keys):
            presence[sig].append(1 if frame.get(key) is not None else 0)

    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(combined))
    for sig in signals:
        vals = np.array(presence[sig], dtype=float)
        ax.fill_between(frame_indices, bottom, bottom + vals,
                        color=SIGNAL_COLORS[sig], alpha=0.75, label=sig)
        bottom += vals

    ax.set_xlabel("frame index")
    ax.set_ylabel("active signals (stacked)")
    ax.set_title("Signal Dominance over Time")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, len(signals))
    ax.grid(True, alpha=0.25)
    _save(fig, out_dir, "signal_dominance.png")


def _angular_err(r1, r2):
    def to_mat(rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx
    R1 = to_mat(*[math.radians(v) for v in r1])
    R2 = to_mat(*[math.radians(v) for v in r2])
    R_rel = R1.T @ R2
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return math.degrees(math.acos(trace))


def compute_summary(all_frames_list, ref_centroid, gt_centroid=None, gt_pose_json=None):
    all_frames = [f for frames in all_frames_list for f in frames]
    n_frames = len(all_frames)

    phase_counts = {"FAR": 0, "NEAR-sam3": 0, "NEAR-track": 0, "TERMINAL": 0}
    for frame in all_frames:
        label = _phase_label(frame)
        if label in phase_counts:
            phase_counts[label] += 1

    terminal_frames = [f for f in all_frames if f.get("state") == "TERMINAL"]
    near_track_frames = [f for f in all_frames if _phase_label(f) == "NEAR-track"]

    def safe_mean(vals):
        valid = [v for v in vals if v is not None]
        return float(np.mean(valid)) if valid else None

    mean_z_terminal = safe_mean([f.get("z_mm") for f in terminal_frames])
    mean_tilt_terminal = safe_mean([f.get("tilt_deg") for f in terminal_frames])
    mean_centroid_err_terminal = safe_mean(
        [_centroid_err(f.get("best_centroid"), ref_centroid) for f in terminal_frames]
    )

    final_centroid_err = None
    if near_track_frames:
        last = near_track_frames[-1]
        bc = last.get("best_centroid")
        if bc is not None:
            final_centroid_err = _centroid_err(bc, ref_centroid)

    watchdog_count = sum(1 for f in all_frames if f.get("watchdog_alarm"))

    summary = {
        "n_frames": n_frames,
        "n_far": phase_counts["FAR"],
        "n_near_sam3": phase_counts["NEAR-sam3"],
        "n_near_track": phase_counts["NEAR-track"],
        "n_terminal": phase_counts["TERMINAL"],
        "mean_z_at_terminal": mean_z_terminal,
        "mean_tilt_at_terminal": mean_tilt_terminal,
        "mean_centroid_err_at_terminal": mean_centroid_err_terminal,
        "final_centroid_err_px": final_centroid_err,
        "watchdog_alarms_count": watchdog_count,
    }

    if gt_centroid is not None:
        gt_errs = []
        for f in all_frames:
            bc = f.get("best_centroid")
            if bc is not None:
                gt_errs.append(_centroid_err(bc, gt_centroid))
        summary["gt_centroid_err_px"] = float(np.mean(gt_errs)) if gt_errs else None

    if gt_pose_json is not None:
        with open(gt_pose_json) as fh:
            pose_data = json.load(fh)
        gt_pose = pose_data.get("gt_pose")

        # final_pose: prefer explicit value in JSON, else take last robot_pos
        # logged in frames.jsonl (written by live pipeline since robot_pos patch)
        final_pose = pose_data.get("final_pose")
        if final_pose is None:
            for f in reversed(all_frames):
                rp = f.get("robot_pos")
                if rp is not None:
                    final_pose = rp
                    break

        if gt_pose and final_pose:
            lateral_err = math.hypot(
                final_pose[0] - gt_pose[0], final_pose[1] - gt_pose[1]
            )
            depth_err = abs(final_pose[2] - gt_pose[2])
            euclidean_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(final_pose[:3], gt_pose[:3])))
            angular_err = _angular_err(gt_pose[3:6], final_pose[3:6])
            tilt_err = mean_tilt_terminal if mean_tilt_terminal is not None else angular_err

            summary["final_robot_pos"] = final_pose
            summary["lateral_err_mm"] = lateral_err
            summary["depth_err_mm"] = depth_err
            summary["euclidean_err_mm"] = euclidean_err
            summary["angular_err_deg"] = angular_err
            summary["success"] = (
                lateral_err < 5.0 and depth_err < 8.0 and tilt_err < 3.0
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="One or more run directories")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <first_run>/analysis/)")
    parser.add_argument("--gt-centroid", nargs=2, type=float, metavar=("X", "Y"),
                        help="Ground-truth grasp centroid in image coordinates")
    parser.add_argument("--gt-pose-json", default=None,
                        help="Path to JSON with gt_pose and final_pose arrays")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dirs[0], "analysis")
    os.makedirs(out_dir, exist_ok=True)

    all_frames_list = []
    for run_dir in args.run_dirs:
        frames = load_frames(run_dir)
        all_frames_list.append(frames)
        print(f"Loaded {len(frames)} frames from {run_dir}")

    gt_centroid = tuple(args.gt_centroid) if args.gt_centroid else None
    ref_centroid = gt_centroid if gt_centroid is not None else IMAGE_CENTER

    print("Generating plots...")
    plot_error_over_distance(all_frames_list, out_dir, ref_centroid, gt_centroid)
    plot_signal_centroids(all_frames_list, out_dir)
    plot_tilt_timeline(all_frames_list, out_dir)
    plot_phase_summary(all_frames_list, out_dir)
    plot_signal_dominance(all_frames_list, out_dir)

    summary = compute_summary(
        all_frames_list, ref_centroid, gt_centroid, args.gt_pose_json
    )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n--- Summary ---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
