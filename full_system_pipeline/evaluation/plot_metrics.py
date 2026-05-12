#!/usr/bin/env python3
"""
Plot servo pipeline metrics from a run's frames.jsonl.

Without GT:   errors are relative to image centre (servo target).
With GT:      errors are relative to the annotated/recorded GT grasp point.

Usage:
    # Basic (errors vs image centre)
    python full_system_pipeline/evaluation/plot_metrics.py runs/trial_01/

    # With 3D GT pose (robot-space error over time)
    python full_system_pipeline/evaluation/plot_metrics.py runs/trial_01/ \\
        --gt-pose-json runs/trial_01/gt_pose.json

    # With image-space GT centroid (annotate first with annotate_gt.py)
    python full_system_pipeline/evaluation/plot_metrics.py runs/trial_01/ \\
        --gt-centroid 658 341

    # Multiple runs compared side by side
    python full_system_pipeline/evaluation/plot_metrics.py runs/trial_01/ runs/trial_02/ \\
        --gt-pose-json runs/trial_01/gt_pose.json
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PHASE_COLORS = {
    "FAR":        "#aec6e8",
    "NEAR-sam3":  "#98d67f",
    "NEAR-track": "#ffb347",
    "TERMINAL":   "#ff6b6b",
}
IMAGE_CENTER = (640, 360)  # 1280×720 default

# ── helpers ──────────────────────────────────────────────────────────────────

def load_frames(run_dir):
    path = os.path.join(run_dir, "frames.jsonl")
    if not os.path.exists(path):
        sys.exit(f"frames.jsonl not found in {run_dir}")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def phase_label(fr):
    s = fr.get("state", "FAR")
    if s == "NEAR":
        return "NEAR-" + (fr.get("_near_phase") or "sam3")
    return s

def centroid_err(c, ref):
    if c is None:
        return None
    return math.hypot(c[0] - ref[0], c[1] - ref[1])

def phase_colors_for(frames):
    return [PHASE_COLORS.get(phase_label(f), "#cccccc") for f in frames]

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")

def _ref_label(gt_centroid):
    if gt_centroid == IMAGE_CENTER:
        return "image centre"
    return f"GT ({gt_centroid[0]:.0f},{gt_centroid[1]:.0f})"

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_phase_timeline(frames, out_dir, run_label=""):
    fig, ax = plt.subplots(figsize=(14, 1.5))
    for i, f in enumerate(frames):
        ax.barh(0, 1, left=i, color=PHASE_COLORS.get(phase_label(f), "#ccc"),
                height=0.6, align="center")
    ax.set_xlim(0, len(frames))
    ax.set_yticks([])
    ax.set_xlabel("Frame index")
    ax.set_title(f"Phase timeline{' — ' + run_label if run_label else ''}")
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9, ncol=4)
    save(fig, os.path.join(out_dir, "phase_timeline.png"))


def plot_centroid_error_over_frames(frames, out_dir, run_label="", gt_centroid=IMAGE_CENTER):
    """Pixel error of best_centroid vs frame index, coloured by phase."""
    errs = [centroid_err(f.get("best_centroid"), gt_centroid) for f in frames]
    idxs = list(range(len(frames)))
    colors = phase_colors_for(frames)

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(len(idxs)):
        if errs[i] is not None:
            ax.bar(idxs[i], errs[i], color=colors[i], width=1.0, align="edge")

    ref = _ref_label(gt_centroid)
    ax.axhline(8, color="black", ls="--", lw=1, label="8 px threshold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel(f"Centroid error vs {ref} (px)")
    ax.set_title(f"Centroid error over time{' — ' + run_label if run_label else ''}")
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, os.path.join(out_dir, "centroid_error_over_frames.png"))


def plot_centroid_error_vs_depth(frames, out_dir, run_label="", gt_centroid=IMAGE_CENTER):
    """Error vs ZED depth — does error reduce as robot approaches?"""
    data = [(f.get("z_mm"), centroid_err(f.get("best_centroid"), gt_centroid), phase_label(f))
            for f in frames
            if f.get("z_mm") is not None and f.get("best_centroid") is not None]
    if not data:
        print("  skip centroid_error_vs_depth — no z_mm data")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for phase, color in PHASE_COLORS.items():
        pts = [(z, e) for z, e, p in data if p == phase]
        if pts:
            zs, es = zip(*pts)
            ax.scatter(zs, es, c=color, label=phase, s=20, alpha=0.8)

    # trend line
    all_z = [d[0] for d in data]
    all_e = [d[1] for d in data]
    if len(all_z) > 3:
        coeffs = np.polyfit(all_z, all_e, 1)
        zline = np.linspace(min(all_z), max(all_z), 100)
        ax.plot(zline, np.polyval(coeffs, zline), "k--", lw=1.2, label="trend")

    ref = _ref_label(gt_centroid)
    ax.set_xlabel("Depth z (mm)  [left = closer]")
    ax.set_ylabel(f"Error vs {ref} (px)")
    ax.set_title(f"Error vs depth{' — ' + run_label if run_label else ''}\n"
                 f"(ideal: error decreases as depth decreases)")
    ax.invert_xaxis()
    ax.axhline(8, color="gray", ls=":", lw=1, label="8 px threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, os.path.join(out_dir, "centroid_error_vs_depth.png"))


def plot_xy_trajectory(frames, out_dir, run_label="", gt_centroid=IMAGE_CENTER):
    """Centroid X and Y over frames, with GT and image-centre reference lines."""
    xs = [f["best_centroid"][0] if f.get("best_centroid") else None for f in frames]
    ys = [f["best_centroid"][1] if f.get("best_centroid") else None for f in frames]
    idxs = list(range(len(frames)))
    colors = phase_colors_for(frames)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x is not None:
            ax1.scatter(idxs[i], x, c=colors[i], s=8)
        if y is not None:
            ax2.scatter(idxs[i], y, c=colors[i], s=8)

    ic_x, ic_y = IMAGE_CENTER
    gt_x, gt_y = gt_centroid
    ax1.axhline(ic_x, color="gray", ls=":", lw=1, label=f"image centre x={ic_x}")
    ax2.axhline(ic_y, color="gray", ls=":", lw=1, label=f"image centre y={ic_y}")
    if gt_centroid != IMAGE_CENTER:
        ax1.axhline(gt_x, color="red", ls="--", lw=1.2, label=f"GT x={gt_x:.0f}")
        ax2.axhline(gt_y, color="red", ls="--", lw=1.2, label=f"GT y={gt_y:.0f}")

    ax1.set_ylabel("Centroid X (px)"); ax1.grid(alpha=0.3)
    ax2.set_ylabel("Centroid Y (px)"); ax2.grid(alpha=0.3)
    ax2.set_xlabel("Frame index")

    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax1.legend(handles=patches, fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Centroid X/Y trajectory{' — ' + run_label if run_label else ''}")
    save(fig, os.path.join(out_dir, "centroid_xy_trajectory.png"))


def plot_gt_error_components(frames, out_dir, run_label="", gt_centroid=IMAGE_CENTER):
    """X-error and Y-error to GT separately — shows which axis is drifting."""
    if gt_centroid == IMAGE_CENTER:
        return  # only meaningful when GT is provided

    near = [(i, f) for i, f in enumerate(frames) if f.get("state") == "NEAR"]
    if not near:
        return

    idxs  = [i for i, _ in near]
    ex    = [f["best_centroid"][0] - gt_centroid[0] if f.get("best_centroid") else None
             for _, f in near]
    ey    = [f["best_centroid"][1] - gt_centroid[1] if f.get("best_centroid") else None
             for _, f in near]
    colors = [PHASE_COLORS.get(phase_label(f), "#ccc") for _, f in near]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, (x, y, c) in enumerate(zip(ex, ey, colors)):
        if x is not None:
            ax1.bar(idxs[i], x, color=c, width=1.0, align="edge")
        if y is not None:
            ax2.bar(idxs[i], y, color=c, width=1.0, align="edge")

    for ax, label in [(ax1, "X error (px)  [+ = right of GT]"),
                      (ax2, "Y error (px)  [+ = below GT]")]:
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(8,  color="red", ls="--", lw=1)
        ax.axhline(-8, color="red", ls="--", lw=1)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)

    ax2.set_xlabel("Frame index")
    fig.suptitle(f"GT-relative X/Y error (NEAR phase){' — ' + run_label if run_label else ''}")
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax1.legend(handles=patches, fontsize=9)
    save(fig, os.path.join(out_dir, "gt_xy_error_components.png"))


def plot_3d_robot_error_vs_gt(frames, out_dir, gt_pose, run_label=""):
    """3 stacked plots: X error, Y error, Z error vs frame index (mm)."""
    data = [(i, f["robot_pos"]) for i, f in enumerate(frames) if f.get("robot_pos")]
    if not data:
        print("  skip xyz_error_vs_gt — no robot_pos (offline run?)")
        return

    idxs   = [i for i, _ in data]
    ex     = [p[0] - gt_pose[0] for _, p in data]   # signed X error
    ey     = [p[1] - gt_pose[1] for _, p in data]   # signed Y error
    ez     = [p[2] - gt_pose[2] for _, p in data]   # signed Z error
    colors = [PHASE_COLORS.get(phase_label(frames[i]), "#ccc") for i in idxs]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for ax, errs, label, color, thresh in [
        (ax1, ex, "X error (mm)  [+ = further right than GT]",  "#e74c3c", 0.0),
        (ax2, ey, "Y error (mm)  [+ = further forward than GT]","#3498db", 0.0),
        (ax3, ez, "Z error (mm)  [+ = higher than GT]",         "#2ecc71", 2.0),
    ]:
        ax.scatter(idxs, errs, c=colors, s=14, zorder=3)
        ax.plot(idxs, errs, color=color, lw=1, alpha=0.6)
        ax.axhline(0, color="black", lw=0.8)
        if thresh > 0:
            ax.axhline( thresh, color="red", ls="--", lw=1, label=f"±{thresh} mm threshold")
            ax.axhline(-thresh, color="red", ls="--", lw=1)
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    ax3.set_xlabel("Frame index  (≈ time)")
    fig.suptitle(f"X / Y / Z error vs GT over time{' — ' + run_label if run_label else ''}\n"
                 f"GT: x={gt_pose[0]:.1f}  y={gt_pose[1]:.1f}  z={gt_pose[2]:.1f} mm",
                 fontsize=11)
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax1.legend(handles=patches, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "xyz_error_vs_gt.png"))

    # also keep the old combined-magnitude version
    lat  = [math.hypot(p[0]-gt_pose[0], p[1]-gt_pose[1]) for _, p in data]
    dep  = [abs(p[2]-gt_pose[2]) for _, p in data]
    euc  = [math.sqrt(sum((a-b)**2 for a,b in zip(p[:3], gt_pose[:3]))) for _, p in data]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, vals, label, thresh, unit in zip(
            axes,
            [lat, dep, euc],
            ["Lateral error (XY)", "Depth error (Z)", "Euclidean 3D error"],
            [None, 2.0, None],
            ["mm", "mm", "mm"]):
        ax.scatter(idxs, vals, c=colors, s=10, zorder=3)
        if thresh:
            ax.axhline(thresh, color="red", ls="--", lw=1,
                       label=f"success threshold ({thresh} {unit})")
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"Robot position error vs GT{' — ' + run_label if run_label else ''}")
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    axes[0].legend(handles=patches, fontsize=8, loc="upper right")
    save(fig, os.path.join(out_dir, "robot_error_vs_gt.png"))


def plot_tilt_over_frames(frames, out_dir, run_label=""):
    near = [(i, f) for i, f in enumerate(frames) if f.get("tilt_deg") is not None]
    if not near:
        print("  skip tilt_over_frames — no tilt data")
        return

    idxs    = [i for i, _ in near]
    tilts   = [f["tilt_deg"]   for _, f in near]
    rolls   = [f.get("roll_deg",  0) for _, f in near]
    pitches = [f.get("pitch_deg", 0) for _, f in near]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idxs, tilts,   color="#e74c3c", lw=1.5, label="tilt_deg (total)")
    ax.plot(idxs, rolls,   color="#3498db", lw=1,   label="roll_deg",  ls="--")
    ax.plot(idxs, pitches, color="#2ecc71", lw=1,   label="pitch_deg", ls=":")
    ax.axhline( 3, color="gray", ls="--", lw=1, label="±3° threshold")
    ax.axhline(-3, color="gray", ls="--", lw=1)
    ax.set_xlabel("Frame index"); ax.set_ylabel("Angle (deg)")
    ax.set_title(f"Surface tilt over time{' — ' + run_label if run_label else ''}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save(fig, os.path.join(out_dir, "tilt_over_frames.png"))


def plot_tilt_vs_depth(frames, out_dir, run_label=""):
    data = [(f["z_mm"], f["tilt_deg"], phase_label(f))
            for f in frames
            if f.get("z_mm") is not None and f.get("tilt_deg") is not None]
    if not data:
        print("  skip tilt_vs_depth — no data")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for phase, color in PHASE_COLORS.items():
        pts = [(z, t) for z, t, p in data if p == phase]
        if pts:
            zs, ts = zip(*pts)
            ax.scatter(zs, ts, c=color, label=phase, s=20, alpha=0.8)

    ax.set_xlabel("Depth z (mm)  [left = closer]")
    ax.set_ylabel("Tilt (deg)")
    ax.set_title(f"Tilt vs depth{' — ' + run_label if run_label else ''}")
    ax.invert_xaxis()
    ax.axhline(3, color="gray", ls="--", lw=1, label="3° threshold")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, os.path.join(out_dir, "tilt_vs_depth.png"))


def plot_signal_centroids(frames, out_dir, run_label="", gt_centroid=IMAGE_CENTER):
    near = [(i, f) for i, f in enumerate(frames) if f.get("state") == "NEAR"]
    if not near:
        print("  skip signal_centroids — no NEAR frames")
        return

    idxs = [i for i, _ in near]
    signals = {
        "best_centroid": ("#000000", "best (servo target)", 2.5),
        "c_B":           ("#ff7f0e", "B CoTracker3",        1.5),
        "c_E":           ("#2ca02c", "E SAM3 anchor",       1.5),
        "c_A":           ("#9467bd", "A locked-3D",         1.0),
    }

    for axis, axis_idx, fname in [("X", 0, "signal_centroids_x.png"),
                                   ("Y", 1, "signal_centroids_y.png")]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ic = IMAGE_CENTER[axis_idx]
        gt = gt_centroid[axis_idx]
        ax.axhline(ic, color="gray", ls=":", lw=1, label=f"image centre {axis}={ic}")
        if gt_centroid != IMAGE_CENTER:
            ax.axhline(gt, color="red", ls="--", lw=1.5, label=f"GT {axis}={gt:.0f}")

        for key, (color, label, lw) in signals.items():
            vals = [f.get(key) for _, f in near]
            pts  = [c[axis_idx] if c is not None else None for c in vals]
            vi   = [idxs[j] for j, v in enumerate(pts) if v is not None]
            vv   = [v for v in pts if v is not None]
            if vv:
                ax.plot(vi, vv, color=color, lw=lw, label=label, marker=".", ms=3)

        ax.set_xlabel("Frame index")
        ax.set_ylabel(f"Centroid {axis} (px)")
        ax.set_title(f"Signal centroids {axis} — NEAR phase{' — ' + run_label if run_label else ''}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        save(fig, os.path.join(out_dir, fname))


def plot_depth_over_frames(frames, out_dir, run_label=""):
    data = [(i, f["z_mm"]) for i, f in enumerate(frames) if f.get("z_mm") is not None]
    if not data:
        print("  skip depth_over_frames — no z_mm data")
        return

    idxs, zs = zip(*data)
    colors = [PHASE_COLORS.get(phase_label(frames[i]), "#cccccc") for i in idxs]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(idxs, zs, c=colors, s=12)
    ax.axhline(30, color="red", ls="--", lw=1, label="TERMINAL trigger (30 mm)")
    ax.set_xlabel("Frame index"); ax.set_ylabel("Depth z (mm)")
    ax.set_title(f"ZED depth over time{' — ' + run_label if run_label else ''}")
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax.legend(handles=patches + [mpatches.Patch(color="red", label="TERMINAL 30mm")],
              fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, os.path.join(out_dir, "depth_over_frames.png"))


def plot_robot_trajectory(frames, out_dir, run_label="", gt_pose=None):
    data = [(i, f["robot_pos"]) for i, f in enumerate(frames) if f.get("robot_pos")]
    if not data:
        print("  skip robot_trajectory — no robot_pos (offline run?)")
        return

    idxs  = [i for i, _ in data]
    xs    = [p[0] for _, p in data]
    ys    = [p[1] for _, p in data]
    zs    = [p[2] for _, p in data]
    colors = [PHASE_COLORS.get(phase_label(frames[i]), "#cccccc") for i in idxs]

    # ── Plot 1: raw trajectory with GT as dashed reference ───────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, vals, label, color, axis_name in zip(
            axes,
            [xs, ys, zs],
            ["X (mm) — approach", "Y (mm) — lateral", "Z (mm) — height"],
            ["#e74c3c", "#3498db", "#2ecc71"],
            ["X", "Y", "Z"]):
        ax.scatter(idxs, vals, c=colors, s=10, zorder=3)
        ax.plot(idxs, vals, color=color, lw=1.2, alpha=0.7)
        if gt_pose is not None:
            gt_val = gt_pose[["X","Y","Z"].index(axis_name)]
            ax.axhline(gt_val, color="black", ls="--", lw=1.5,
                       label=f"GT {axis_name} = {gt_val:.1f} mm")
            ax.legend(fontsize=9, loc="upper right")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Frame index  (≈ time)")
    fig.suptitle(
        f"Robot EE trajectory{' — ' + run_label if run_label else ''}"
        + ("\n(black dashed = GT grasp position)" if gt_pose else ""),
        fontsize=11,
    )
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    axes[0].legend(handles=patches + (
        [mpatches.Patch(color="black", label=f"GT X={gt_pose[0]:.1f}mm")]
        if gt_pose else []), fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "robot_xyz_trajectory.png"))

    # ── Plot 2: absolute error vs GT — only meaningful when GT is provided ────
    if gt_pose is None:
        return

    ex = [abs(x - gt_pose[0]) for x in xs]
    ey = [abs(y - gt_pose[1]) for y in ys]
    ez = [abs(z - gt_pose[2]) for z in zs]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, errs, label, color, thresh in [
        (ax1, ex, "|X error| (mm)", "#e74c3c", 0.0),
        (ax2, ey, "|Y error| (mm)", "#3498db", 0.0),
        (ax3, ez, "|Z error| (mm)", "#2ecc71", 2.0),
    ]:
        ax.scatter(idxs, errs, c=colors, s=10, zorder=3)
        ax.plot(idxs, errs, color=color, lw=1.2, alpha=0.7)
        if thresh > 0:
            ax.axhline(thresh, color="red", ls="--", lw=1.2,
                       label=f"threshold = {thresh} mm")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(label)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

    ax3.set_xlabel("Frame index  (≈ time)")
    fig.suptitle(
        f"Robot EE absolute error vs GT{' — ' + run_label if run_label else ''}\n"
        f"GT: x={gt_pose[0]:.1f}  y={gt_pose[1]:.1f}  z={gt_pose[2]:.1f} mm",
        fontsize=11,
    )
    patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
    ax1.legend(handles=patches, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "robot_xyz_error_vs_gt.png"))


def plot_inliers_over_frames(frames, out_dir, run_label=""):
    data = [(i, f["near_plane_n_inliers"]) for i, f in enumerate(frames)
            if f.get("near_plane_n_inliers") is not None]
    if not data:
        print("  skip inliers_over_frames — no inlier data")
        return

    idxs, inliers = zip(*data)
    colors = [PHASE_COLORS.get(phase_label(frames[i]), "#cccccc") for i in idxs]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(idxs, inliers, color=colors, width=1.0, align="edge")
    ax.axhline(500, color="red", ls="--", lw=1, label="tilt-correction gate (500)")
    ax.set_xlabel("Frame index"); ax.set_ylabel("RANSAC inliers")
    ax.set_title(f"Surface normal confidence{' — ' + run_label if run_label else ''}")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    save(fig, os.path.join(out_dir, "ransac_inliers.png"))


def print_summary(frames, gt_centroid=IMAGE_CENTER, gt_pose=None, run_label=""):
    print(f"\n{'─'*60}")
    if run_label:
        print(f"  Run: {run_label}")

    phase_counts = {}
    for f in frames:
        p = phase_label(f)
        phase_counts[p] = phase_counts.get(p, 0) + 1
    for p in ["FAR", "NEAR-sam3", "NEAR-track", "TERMINAL"]:
        print(f"  {p:<14}: {phase_counts.get(p, 0):4d} frames")

    ref_str = _ref_label(gt_centroid)
    term = [f for f in frames if f.get("state") == "TERMINAL"]
    if term:
        errs  = [centroid_err(f.get("best_centroid"), gt_centroid)
                 for f in term if f.get("best_centroid")]
        tilts = [f["tilt_deg"] for f in term if f.get("tilt_deg") is not None]
        zs    = [f["z_mm"]     for f in term if f.get("z_mm")     is not None]
        if errs:  print(f"  centroid err @ TERMINAL (vs {ref_str}): "
                        f"mean={np.mean(errs):.1f} px  std={np.std(errs):.1f} px")
        if tilts: print(f"  tilt @ TERMINAL:  mean={np.mean(tilts):.2f}°  std={np.std(tilts):.2f}°")
        if zs:    print(f"  depth @ TERMINAL: mean={np.mean(zs):.1f} mm")

    print(f"  watchdog alarms: {sum(1 for f in frames if f.get('watchdog_alarm'))}")

    if gt_pose is not None:
        final_pose = None
        for f in reversed(frames):
            if f.get("robot_pos"):
                final_pose = f["robot_pos"]
                break
        if final_pose:
            lat = math.hypot(final_pose[0]-gt_pose[0], final_pose[1]-gt_pose[1])
            dep = abs(final_pose[2]-gt_pose[2])
            euc = math.sqrt(sum((a-b)**2 for a,b in zip(final_pose[:3], gt_pose[:3])))
            tilt_err = np.mean([f["tilt_deg"] for f in term if f.get("tilt_deg")]) if term else 99
            success = dep < 2.0 and tilt_err < 3.0
            print(f"  ── 3D error vs GT ──────────────────────")
            print(f"  lateral error:    {lat:.2f} mm")
            print(f"  depth error:      {dep:.2f} mm  (threshold 2 mm)")
            print(f"  euclidean error:  {euc:.2f} mm")
            print(f"  SUCCESS: {'YES ✓' if success else 'NO  ✗'}"
                  f"  (dep<2={dep<2:.0f}  tilt<3°={tilt_err<3:.0f})")
        else:
            print("  (no robot_pos in frames — run was offline or robot_pos not logged)")

    print(f"{'─'*60}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+",
                   help="Run directories containing frames.jsonl")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <run_dir>/plots/)")
    p.add_argument("--gt-pose-json", default=None,
                   help="gt_pose.json recorded by record_grasp_point.py")
    p.add_argument("--gt-centroid", nargs=2, type=float, metavar=("X", "Y"),
                   default=None,
                   help="Image-space GT centroid in pixels, e.g. --gt-centroid 658 341 "
                        "(annotate with annotate_gt.py first)")
    args = p.parse_args()

    # Resolve GT centroid reference
    gt_centroid = IMAGE_CENTER
    if args.gt_centroid:
        gt_centroid = tuple(args.gt_centroid)
        print(f"GT centroid: {gt_centroid} px  (all errors relative to this point)")
    else:
        print("No --gt-centroid supplied — errors are relative to image centre (640,360).")
        print("For GT-relative plots: first run annotate_gt.py then pass --gt-centroid X Y")

    # Resolve GT pose
    gt_pose = None
    if args.gt_pose_json and os.path.exists(args.gt_pose_json):
        with open(args.gt_pose_json) as fh:
            gt_pose = json.load(fh).get("gt_pose")
        if gt_pose:
            print(f"GT pose loaded: x={gt_pose[0]:.1f}  y={gt_pose[1]:.1f}  z={gt_pose[2]:.1f} mm")

    for run_dir in args.run_dirs:
        # Auto-detect gt_pose.json inside run_dir if not supplied
        local_gt_pose = gt_pose
        if local_gt_pose is None:
            candidate = os.path.join(run_dir, "gt_pose.json")
            if os.path.exists(candidate):
                with open(candidate) as fh:
                    local_gt_pose = json.load(fh).get("gt_pose")
                if local_gt_pose:
                    print(f"  Auto-loaded GT pose from {candidate}")

        out_dir = args.out_dir or os.path.join(run_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        run_label = os.path.basename(run_dir.rstrip("/"))
        print(f"\nProcessing: {run_dir}  →  {out_dir}")
        frames = load_frames(run_dir)
        print(f"  {len(frames)} frames")

        plot_phase_timeline(frames, out_dir, run_label)
        plot_centroid_error_over_frames(frames, out_dir, run_label, gt_centroid)
        plot_centroid_error_vs_depth(frames, out_dir, run_label, gt_centroid)
        plot_xy_trajectory(frames, out_dir, run_label, gt_centroid)
        plot_gt_error_components(frames, out_dir, run_label, gt_centroid)
        plot_tilt_over_frames(frames, out_dir, run_label)
        plot_tilt_vs_depth(frames, out_dir, run_label)
        plot_signal_centroids(frames, out_dir, run_label, gt_centroid)
        plot_depth_over_frames(frames, out_dir, run_label)
        plot_robot_trajectory(frames, out_dir, run_label, local_gt_pose)
        plot_inliers_over_frames(frames, out_dir, run_label)
        if local_gt_pose:
            plot_3d_robot_error_vs_gt(frames, out_dir, local_gt_pose, run_label)

        print_summary(frames, gt_centroid, local_gt_pose, run_label)
        print(f"\n  Done → {out_dir}/")


if __name__ == "__main__":
    main()
