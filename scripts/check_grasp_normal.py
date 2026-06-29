#!/usr/bin/env python3
"""
check_grasp_normal.py
=====================
Standalone perception validation script.

Given one reference image (masked_objects/<name>.png) and one RGB-D frame
(saved numpy files OR live ZED), it:
  1. Runs SAM3 to segment the target object
  2. Cleans the mask → raw / clean / eroded
  3. Backprojects depth through the eroded mask → 3D point cloud
  4. Fits a RANSAC plane (fit_plane_ransac)
  5. Corrects normal sign + runs sanity checks
  6. Sets grasp point = plane inlier centroid
  7. Draws 2D overlay (OpenCV) and 3D cloud (Plotly offline / Open3D live)
  8. Saves overlay PNG + JSONL log

Usage (offline):
    python scripts/check_grasp_normal.py \\
        --ref masked_objects/protein_bar.png \\
        --rgb data/rgb/frame_0000.png \\
        --depth data/depth/frame_0000.npy \\
        --prompt "protein bar"

Usage (live ZED):
    python scripts/check_grasp_normal.py \\
        --ref masked_objects/cheez_it_box.png \\
        --live

Integration interface:
    from scripts.check_grasp_normal import GraspResult, compute_grasp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path: make sure the project root is importable when run directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from foundation_model.servo_lastmile import (
    DEFAULT_INTRINSICS,
    fit_plane_ransac,
    make_intrinsics_matrix,
    project_3d_to_pixel,
)

log = logging.getLogger("check_grasp")

# ---------------------------------------------------------------------------
# Tunable constants (all documented in grasp_selection_readme.md §25.5)
# ---------------------------------------------------------------------------
RANSAC_THRESH_MM      = 8.0    # inlier distance threshold
RANSAC_MAX_ITER       = 200
RANSAC_MIN_INLIER_R   = 0.35   # minimum inlier ratio to accept plane
RANSAC_MIN_INLIERS    = 50     # minimum absolute inlier count
ERODE_KERNEL          = 7      # morphological erosion kernel (px)
ARROW_LENGTH_MM       = 60.0   # normal arrow scale for 2D overlay
EMA_ALPHA             = 0.3    # temporal smoothing factor (live/video mode)
SIM_ACCEPT_THRESH     = 0.40   # minimum ResNet-18 similarity to accept mask
DEPTH_MIN_MM          = 100.0  # workspace near clip
DEPTH_MAX_MM          = 1500.0 # workspace far clip
TILT_WARN_DEG         = 45.0   # log STEEP_TILT warning above this
TILT_REJECT_DEG       = 80.0   # log LOW_SUCTION_CONFIDENCE above this

# ---------------------------------------------------------------------------
# Integration interface
# ---------------------------------------------------------------------------

@dataclass
class GraspResult:
    target_found: bool
    grasp_point_2d: tuple[int, int] | None      # (u, v) pixels
    grasp_point_cam: np.ndarray | None           # (3,) mm, camera frame
    normal_cam: np.ndarray | None                # (3,) unit vector
    plane_inlier_ratio: float = 0.0
    tilt_deg: float = 0.0
    reference_similarity: float | None = None
    status: str = "NO_TARGET"
    # debug extras
    raw_mask: np.ndarray | None = field(default=None, repr=False)
    eroded_mask: np.ndarray | None = field(default=None, repr=False)
    inlier_pts: np.ndarray | None = field(default=None, repr=False)


def compute_grasp(
    frame_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    sam3_runner,                           # callable: frame_bgr → dict
    arrow_length_mm: float = ARROW_LENGTH_MM,
    erode_kernel: int = ERODE_KERNEL,
    ransac_thresh_mm: float = RANSAC_THRESH_MM,
    _ema_state: dict | None = None,        # mutable dict for live EMA state
) -> GraspResult:
    """
    Run full grasp-point + surface-normal pipeline on one frame.

    Parameters
    ----------
    frame_bgr   : H×W×3 BGR image
    depth_mm    : H×W depth map in millimetres, aligned to frame_bgr
    K           : 3×3 camera intrinsics matrix
    sam3_runner : callable returned by make_default_sam3_runner()
    _ema_state  : pass a persistent dict for temporal EMA smoothing (live mode)

    Returns
    -------
    GraspResult
    """
    h, w = frame_bgr.shape[:2]
    fail = GraspResult(target_found=False, grasp_point_2d=None,
                       grasp_point_cam=None, normal_cam=None)

    # ------------------------------------------------------------------
    # 1. SAM3 → raw mask
    # ------------------------------------------------------------------
    res = sam3_runner(frame_bgr)
    raw_mask = res.get("mask_np")
    if raw_mask is None or raw_mask.sum() == 0:
        fail.status = "NO_TARGET"
        return fail

    raw_mask = (raw_mask > 0).astype(np.uint8) * 255
    similarity = res.get("similarity")

    if similarity is not None and similarity < SIM_ACCEPT_THRESH:
        log.warning("SAM3 similarity %.3f below threshold %.2f → LOW_REF_SIMILARITY",
                    similarity, SIM_ACCEPT_THRESH)
        fail.status = "LOW_REF_SIMILARITY"
        fail.reference_similarity = float(similarity)
        fail.raw_mask = raw_mask
        return fail

    # ------------------------------------------------------------------
    # 2. Mask cleanup → clean_mask + eroded_mask
    # ------------------------------------------------------------------
    clean_mask = _clean_mask(raw_mask)
    if clean_mask.sum() == 0:
        fail.status = "NO_TARGET"
        return fail

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (erode_kernel, erode_kernel))
    eroded_mask = cv2.erode(clean_mask, kernel, iterations=1)
    if eroded_mask.sum() == 0:
        # erosion wiped the mask; fall back to clean_mask
        log.warning("Erosion eliminated mask; falling back to clean_mask.")
        eroded_mask = clean_mask

    # ------------------------------------------------------------------
    # 3. Backproject eroded_mask pixels through depth → 3D point cloud
    # ------------------------------------------------------------------
    pts_xyz = _backproject(eroded_mask, depth_mm, K,
                           depth_min=DEPTH_MIN_MM, depth_max=DEPTH_MAX_MM)
    if pts_xyz.shape[0] < 3:
        fail.status = "BAD_DEPTH"
        fail.raw_mask = raw_mask
        fail.eroded_mask = eroded_mask
        return fail

    # ------------------------------------------------------------------
    # 4. RANSAC plane fit
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    fit = fit_plane_ransac(pts_xyz, thresh_mm=ransac_thresh_mm,
                           max_iter=RANSAC_MAX_ITER, rng=rng)
    if fit is None:
        # Retry with looser threshold
        fit = fit_plane_ransac(pts_xyz, thresh_mm=ransac_thresh_mm * 1.5,
                               max_iter=RANSAC_MAX_ITER, rng=rng)
        if fit is not None:
            log.info("RANSAC succeeded only with thresh_mm=%.1f (retry).",
                     ransac_thresh_mm * 1.5)

    if fit is None:
        fail.status = "BAD_PLANE"
        fail.raw_mask = raw_mask
        fail.eroded_mask = eroded_mask
        return fail

    normal_raw, centroid = fit

    # Measure inlier ratio
    d_val = -float(normal_raw @ centroid)
    dists = np.abs(pts_xyz @ normal_raw + d_val)
    inlier_mask = dists < ransac_thresh_mm
    inlier_ratio = float(inlier_mask.mean())
    inlier_pts = pts_xyz[inlier_mask]
    n_inliers = int(inlier_mask.sum())

    if inlier_ratio < RANSAC_MIN_INLIER_R or n_inliers < RANSAC_MIN_INLIERS:
        log.warning("RANSAC inlier_ratio=%.3f n=%d below thresholds → BAD_PLANE",
                    inlier_ratio, n_inliers)
        fail.status = "BAD_PLANE"
        fail.raw_mask = raw_mask
        fail.eroded_mask = eroded_mask
        return fail

    # ------------------------------------------------------------------
    # 5. Normal sign correction
    # fit_plane_ransac already orients normal_raw[2] < 0 (toward camera),
    # but we double-check with the dot-product convention.
    # ------------------------------------------------------------------
    normal = normal_raw.copy()
    if np.dot(normal, -centroid) < 0:
        normal = -normal

    # ------------------------------------------------------------------
    # 6. Normal sanity checks
    # ------------------------------------------------------------------
    status = "GOOD"

    # Z-component: normal must have a meaningful component toward camera
    if abs(normal[2]) < 0.1:
        log.warning("Normal nearly perpendicular to optical axis → BAD_PLANE")
        fail.status = "BAD_PLANE"
        fail.raw_mask = raw_mask
        fail.eroded_mask = eroded_mask
        return fail

    # Tilt angle from optical axis
    tilt_deg = float(np.degrees(np.arccos(
        np.clip(abs(np.dot(normal, np.array([0.0, 0.0, -1.0]))), 0.0, 1.0)
    )))
    if tilt_deg > TILT_REJECT_DEG:
        status = "LOW_SUCTION_CONFIDENCE"
        log.warning("Tilt=%.1f° exceeds %.0f° → LOW_SUCTION_CONFIDENCE",
                    tilt_deg, TILT_REJECT_DEG)
    elif tilt_deg > TILT_WARN_DEG:
        status = "STEEP_TILT"
        log.info("Tilt=%.1f° — STEEP_TILT warning.", tilt_deg)

    # Plane centroid must project inside raw mask bounding box
    uv_centroid = project_3d_to_pixel(centroid, K)
    if uv_centroid is not None:
        ys, xs = np.where(raw_mask > 0)
        if ys.size > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cu, cv_ = uv_centroid
            if not (x1 <= cu <= x2 and y1 <= cv_ <= y2):
                log.warning("Plane centroid projected outside mask bbox → BAD_PLANE")
                fail.status = "BAD_PLANE"
                fail.raw_mask = raw_mask
                fail.eroded_mask = eroded_mask
                return fail

    # ------------------------------------------------------------------
    # 7. Temporal EMA (live/video mode — only when _ema_state is provided)
    # ------------------------------------------------------------------
    if _ema_state is not None:
        prev = _ema_state.get("normal")
        if prev is not None:
            delta = float(np.degrees(np.arccos(
                np.clip(abs(np.dot(normal, prev)), 0.0, 1.0))))
            if delta > 30.0:
                log.info("EMA: normal_delta=%.1f° — outlier frame, reusing prev.", delta)
                normal = prev
            else:
                blended = EMA_ALPHA * normal + (1 - EMA_ALPHA) * prev
                norm_len = np.linalg.norm(blended)
                if norm_len > 1e-6:
                    normal = blended / norm_len
        _ema_state["normal"] = normal.copy()

    # ------------------------------------------------------------------
    # 8. Grasp point = plane inlier centroid
    # ------------------------------------------------------------------
    grasp_3d = centroid  # fit_plane_ransac returns mean of inliers
    uv = project_3d_to_pixel(grasp_3d, K)
    if uv is None:
        fail.status = "BAD_PLANE"
        return fail
    grasp_2d = (int(round(uv[0])), int(round(uv[1])))

    return GraspResult(
        target_found=True,
        grasp_point_2d=grasp_2d,
        grasp_point_cam=grasp_3d,
        normal_cam=normal,
        plane_inlier_ratio=inlier_ratio,
        tilt_deg=tilt_deg,
        reference_similarity=float(similarity) if similarity is not None else None,
        status=status,
        raw_mask=raw_mask,
        eroded_mask=eroded_mask,
        inlier_pts=inlier_pts,
    )


# ---------------------------------------------------------------------------
# Mask cleaning
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes, keep largest connected component."""
    # fill small holes
    filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # keep largest connected component
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(filled,
                                                                 connectivity=8)
    if n_comp <= 1:
        return filled
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth backprojection
# ---------------------------------------------------------------------------

def _backproject(mask: np.ndarray, depth_mm: np.ndarray, K: np.ndarray,
                 depth_min: float = 100.0,
                 depth_max: float = 1500.0) -> np.ndarray:
    """Return Nx3 array of valid 3D points inside mask (camera frame, mm)."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_mm[ys, xs].astype(np.float64)
    valid = (z > depth_min) & (z < depth_max) & np.isfinite(z)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2D overlay (always OpenCV — both offline and live)
# ---------------------------------------------------------------------------

def draw_overlay(frame_bgr: np.ndarray, result: GraspResult,
                 K: np.ndarray,
                 arrow_length_mm: float = ARROW_LENGTH_MM) -> np.ndarray:
    """
    Draw grasp point cross + normal arrow + mask outline + HUD text.
    Returns a copy of the frame with overlays.
    """
    out = frame_bgr.copy()

    if result.raw_mask is not None:
        contours, _ = cv2.findContours(result.raw_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)  # yellow outline

    if not result.target_found:
        _draw_hud(out, result)
        return out

    # Grasp point cross (red)
    gp = result.grasp_point_2d
    cv2.drawMarker(out, gp, (0, 0, 255), cv2.MARKER_CROSS, 20, 3)

    # Normal arrow (green)
    if result.normal_cam is not None and result.grasp_point_cam is not None:
        tip_3d = result.grasp_point_cam + result.normal_cam * arrow_length_mm
        tip_uv = project_3d_to_pixel(tip_3d, K)
        if tip_uv is not None:
            tip = (int(round(tip_uv[0])), int(round(tip_uv[1])))
            cv2.arrowedLine(out, gp, tip, (0, 255, 0), 3, tipLength=0.2)

    # Plane inlier pixels (cyan)
    if result.inlier_pts is not None and result.inlier_pts.shape[0] > 0:
        for pt in result.inlier_pts[::max(1, len(result.inlier_pts) // 500)]:
            uv = project_3d_to_pixel(pt, K)
            if uv is not None:
                cv2.circle(out, (int(uv[0]), int(uv[1])), 1, (255, 255, 0), -1)

    _draw_hud(out, result)
    return out


def _draw_hud(img: np.ndarray, r: GraspResult) -> None:
    lines = [
        f"target_found: {r.target_found}",
        f"status: {r.status}",
        f"inlier_ratio: {r.plane_inlier_ratio:.2f}",
        f"tilt_deg: {r.tilt_deg:.1f}",
    ]
    if r.reference_similarity is not None:
        lines.append(f"ref_sim: {r.reference_similarity:.3f}")
    if r.grasp_point_cam is not None:
        p = r.grasp_point_cam
        lines.append(f"grasp_cam: [{p[0]:.0f},{p[1]:.0f},{p[2]:.0f}]mm")
    if r.normal_cam is not None:
        n = r.normal_cam
        lines.append(f"normal: [{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}]")

    y0 = 25
    for line in lines:
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y0 += 22


def draw_mask_stages(raw: np.ndarray, clean: np.ndarray,
                     eroded: np.ndarray) -> np.ndarray:
    """Side-by-side: raw / clean / eroded masks on a white background."""
    def to_bgr(m):
        return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    panel = np.hstack([to_bgr(raw), to_bgr(clean), to_bgr(eroded)])
    h = panel.shape[0]
    for i, label in enumerate(["raw", "clean", "eroded"]):
        x = i * raw.shape[1] + 5
        cv2.putText(panel, label, (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)
    return panel


# ---------------------------------------------------------------------------
# 3D visualisation — Plotly (offline) or Open3D (live)
# ---------------------------------------------------------------------------

def show_3d_plotly(pts_xyz: np.ndarray, inlier_pts: np.ndarray | None,
                   grasp_3d: np.ndarray, normal: np.ndarray,
                   arrow_mm: float = ARROW_LENGTH_MM,
                   out_html: str | None = None) -> None:
    """Static interactive 3D scatter. Opens in browser or saves HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.warning("Plotly not installed; skipping 3D visualisation.")
        return

    traces = []
    if pts_xyz.shape[0] > 0:
        step = max(1, pts_xyz.shape[0] // 2000)
        sub = pts_xyz[::step]
        traces.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
            mode="markers",
            marker=dict(size=2, color=sub[:, 2],
                        colorscale="Viridis", opacity=0.5),
            name="point cloud",
        ))

    if inlier_pts is not None and inlier_pts.shape[0] > 0:
        step = max(1, inlier_pts.shape[0] // 1000)
        sub = inlier_pts[::step]
        traces.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
            mode="markers",
            marker=dict(size=3, color="cyan", opacity=0.8),
            name="plane inliers",
        ))

    # grasp point sphere
    traces.append(go.Scatter3d(
        x=[grasp_3d[0]], y=[grasp_3d[1]], z=[grasp_3d[2]],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="grasp point",
    ))

    # normal arrow as a line
    tip = grasp_3d + normal * arrow_mm
    traces.append(go.Scatter3d(
        x=[grasp_3d[0], tip[0]],
        y=[grasp_3d[1], tip[1]],
        z=[grasp_3d[2], tip[2]],
        mode="lines+markers",
        line=dict(color="green", width=6),
        marker=dict(size=4, color="green"),
        name="normal",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="X (mm)",
                   yaxis_title="Y (mm)",
                   zaxis_title="Z (mm)"),
        title="Grasp point + surface normal",
    )
    if out_html:
        fig.write_html(out_html)
        log.info("3D plot saved → %s", out_html)
    else:
        fig.show()


def show_3d_open3d(pts_xyz: np.ndarray, inlier_pts: np.ndarray | None,
                   grasp_3d: np.ndarray, normal: np.ndarray,
                   arrow_mm: float = ARROW_LENGTH_MM) -> None:
    """Live Open3D window for real-time verification."""
    try:
        import open3d as o3d
    except ImportError:
        log.warning("Open3D not installed; skipping 3D visualisation.")
        return

    geoms = []

    if pts_xyz.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz / 1000.0)  # m
        pcd.paint_uniform_color([0.5, 0.5, 0.8])
        geoms.append(pcd)

    if inlier_pts is not None and inlier_pts.shape[0] > 0:
        pcd_in = o3d.geometry.PointCloud()
        pcd_in.points = o3d.utility.Vector3dVector(inlier_pts / 1000.0)
        pcd_in.paint_uniform_color([0.0, 1.0, 1.0])
        geoms.append(pcd_in)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.translate(grasp_3d / 1000.0)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    geoms.append(sphere)

    tip = grasp_3d + normal * arrow_mm
    pts_line = np.array([grasp_3d / 1000.0, tip / 1000.0])
    lines_set = o3d.geometry.LineSet()
    lines_set.points = o3d.utility.Vector3dVector(pts_line)
    lines_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    lines_set.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])
    geoms.append(lines_set)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geoms.append(frame)

    o3d.visualization.draw_geometries(geoms, window_name="Grasp + Normal",
                                      width=800, height=600)


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

def _result_to_dict(r: GraspResult, frame_id: int) -> dict:
    def _arr(a):
        return a.tolist() if isinstance(a, np.ndarray) else a

    return dict(
        frame_id=frame_id,
        timestamp=time.time(),
        target_found=r.target_found,
        status=r.status,
        reference_similarity=r.reference_similarity,
        grasp_point_2d=list(r.grasp_point_2d) if r.grasp_point_2d else None,
        grasp_point_cam=_arr(r.grasp_point_cam),
        normal_cam=_arr(r.normal_cam),
        plane_inlier_ratio=r.plane_inlier_ratio,
        tilt_deg=r.tilt_deg,
    )


# ---------------------------------------------------------------------------
# Depth loading helpers
# ---------------------------------------------------------------------------

def _load_depth_mm(path: str) -> np.ndarray:
    """Load depth from .npy (mm), .npz key 'depth', or 16-bit PNG (mm)."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.endswith(".npz"):
        d = np.load(path)
        key = "depth" if "depth" in d else list(d.keys())[0]
        return d[key].astype(np.float32)
    # 16-bit PNG: assume mm
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Cannot load depth from {path}")
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Offline mode: process one saved RGB-D pair
# ---------------------------------------------------------------------------

def run_offline(args: argparse.Namespace) -> None:
    from foundation_model.servo_lastmile import make_default_sam3_runner

    out_dir = Path(args.output)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.jsonl"

    # Intrinsics
    if args.intrinsics:
        with open(args.intrinsics) as f:
            intr = json.load(f)
    else:
        intr = DEFAULT_INTRINSICS
        log.info("Using DEFAULT_INTRINSICS (fx=%.0f fy=%.0f cx=%.0f cy=%.0f)",
                 intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    K = make_intrinsics_matrix(intr)

    # SAM3 runner
    runner = make_default_sam3_runner(args.ref, prompt=args.prompt)

    # Load frame
    frame_bgr = cv2.imread(args.rgb)
    if frame_bgr is None:
        log.error("Cannot load RGB image: %s", args.rgb)
        return
    depth_mm = _load_depth_mm(args.depth)

    result = compute_grasp(frame_bgr, depth_mm, K, runner,
                           erode_kernel=args.erode_kernel,
                           ransac_thresh_mm=args.ransac_thresh)

    log.info("Result: found=%s status=%s tilt=%.1f° inlier_ratio=%.2f",
             result.target_found, result.status,
             result.tilt_deg, result.plane_inlier_ratio)

    # 2D overlay
    overlay = draw_overlay(frame_bgr, result, K)
    ov_path = str(out_dir / "overlays" / "frame_0000_overlay.png")
    cv2.imwrite(ov_path, overlay)
    log.info("Overlay saved → %s", ov_path)

    # Mask stages
    if result.raw_mask is not None and result.eroded_mask is not None:
        clean = _clean_mask(result.raw_mask)
        stages = draw_mask_stages(result.raw_mask, clean, result.eroded_mask)
        cv2.imwrite(str(out_dir / "masks" / "frame_0000_stages.png"), stages)

    # 3D visualisation (Plotly in offline mode)
    if result.target_found and not args.no_3d:
        pts = _backproject(result.eroded_mask if result.eroded_mask is not None
                           else np.zeros_like(depth_mm, dtype=np.uint8),
                           depth_mm, K)
        html_path = str(out_dir / "overlays" / "frame_0000_3d.html")
        show_3d_plotly(pts, result.inlier_pts,
                       result.grasp_point_cam, result.normal_cam,
                       out_html=html_path)

    # JSONL log
    with open(log_path, "a") as f:
        f.write(json.dumps(_result_to_dict(result, 0)) + "\n")

    # Show overlay
    if not args.no_display:
        cv2.imshow("Grasp + Normal", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Live ZED mode
# ---------------------------------------------------------------------------

def run_live(args: argparse.Namespace) -> None:
    try:
        import pyzed.sl as sl
    except ImportError:
        log.error("pyzed not installed. Use --rgb/--depth for offline mode.")
        return

    from foundation_model.servo_lastmile import make_default_sam3_runner

    out_dir = Path(args.output)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.jsonl"

    cam = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.camera_resolution = sl.RESOLUTION.HD720
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        log.error("ZED open failed: %s", err)
        return

    cam_info = cam.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    intr = dict(fx=calib.left_cam.fx, fy=calib.left_cam.fy,
                cx=calib.left_cam.cx, cy=calib.left_cam.cy)
    log.info("ZED intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
             intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    K = make_intrinsics_matrix(intr)

    runner = make_default_sam3_runner(args.ref, prompt=args.prompt)
    ema_state: dict = {}

    zed_image = sl.Mat()
    zed_depth = sl.Mat()
    runtime = sl.RuntimeParameters()

    frame_id = 0
    log.info("ZED live loop started. Press Q to quit.")
    try:
        while True:
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            cam.retrieve_image(zed_image, sl.VIEW.LEFT)
            cam.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

            frame_bgr = cv2.cvtColor(zed_image.get_data()[:, :, :3],
                                     cv2.COLOR_BGRA2BGR)
            depth_mm = zed_depth.get_data().astype(np.float32)

            result = compute_grasp(frame_bgr, depth_mm, K, runner,
                                   erode_kernel=args.erode_kernel,
                                   ransac_thresh_mm=args.ransac_thresh,
                                   _ema_state=ema_state)

            overlay = draw_overlay(frame_bgr, result, K)

            ov_path = str(out_dir / "overlays" / f"frame_{frame_id:04d}_overlay.png")
            cv2.imwrite(ov_path, overlay)
            with open(log_path, "a") as f:
                f.write(json.dumps(_result_to_dict(result, frame_id)) + "\n")

            cv2.imshow("check_grasp_normal [ZED live]", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if result.target_found and not args.no_3d and frame_id % 30 == 0:
                pts = _backproject(result.eroded_mask, depth_mm, K)
                show_3d_open3d(pts, result.inlier_pts,
                               result.grasp_point_cam, result.normal_cam)

            frame_id += 1
    finally:
        cam.close()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grasp point + surface normal validation script.")
    p.add_argument("--ref", required=True, metavar="PATH",
                   help="Reference image path (masked_objects/<name>.png)")
    p.add_argument("--prompt", default="box",
                   help="Grounding DINO text prompt (default: 'box')")
    p.add_argument("--rgb", metavar="PATH",
                   help="Offline RGB image (.png/.jpg)")
    p.add_argument("--depth", metavar="PATH",
                   help="Offline depth map (.npy/.npz or 16-bit PNG, mm)")
    p.add_argument("--live", action="store_true",
                   help="Use live ZED camera instead of saved files")
    p.add_argument("--intrinsics", metavar="JSON",
                   help="Camera intrinsics JSON {fx,fy,cx,cy}. "
                        "Default: ZED Mini HD720 defaults.")
    p.add_argument("--output", default="outputs",
                   help="Output directory for overlays, masks, logs.")
    p.add_argument("--erode-kernel", type=int, default=ERODE_KERNEL,
                   help=f"Erosion kernel size in px (default: {ERODE_KERNEL})")
    p.add_argument("--ransac-thresh", type=float, default=RANSAC_THRESH_MM,
                   help=f"RANSAC inlier threshold mm (default: {RANSAC_THRESH_MM})")
    p.add_argument("--no-3d", action="store_true",
                   help="Skip 3D visualisation (Plotly/Open3D)")
    p.add_argument("--no-display", action="store_true",
                   help="Skip cv2.imshow (for headless runs)")
    p.add_argument("--debug", action="store_true",
                   help="Set log level to DEBUG")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.live:
        run_live(args)
    elif args.rgb and args.depth:
        run_offline(args)
    else:
        print("ERROR: provide --rgb + --depth for offline mode, or --live for ZED.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
