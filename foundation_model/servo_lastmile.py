#!/usr/bin/env python3
"""
SemVS - Last-Mile Visual Servoing Pipeline (servo_lastmile)
============================================================

End-to-end pipeline that fixes the "object exits the frame as we approach"
failure of the SAM3-only pipeline. Implements the multi-source fused
estimator described in the design doc:

    FAR   -> SAM3 detection + ResNet disambig (existing pipeline)
    LOCK  -> capture 3D centroid in robot base frame, ref crop,
             CoTracker init points, DINOv2 reference features
    NEAR  -> 4 signals running in parallel, fused via geometric median:
               A) locked-3D centroid projected through EE pose
               B) CoTracker3 online (LK fallback) on uniform-sampled points
               C) SAM2 mask propagation (watchdog only)
               D) DINOv2 best-buddy correspondence to LOCK crop
    TERM  -> plane-fit on ZED stereo depth + normal alignment

The pipeline is designed for graceful degradation: heavy model deps
(SAM3/SAM2/CoTracker3/DINOv2) are lazily imported and each Signal* class
falls back to a deterministic stub if its model is unavailable. This
lets the math/state-machine path be unit-tested without GPU weights.

Hand-eye calibration:
    A 4x4 T_ee_cam (camera frame in EE frame) is required for Signal A.
    Provide it via --hand-eye PATH (npy or yaml). Default identity makes
    Signal A degenerate and forces fusion to lean on B/C/D only.

Usage (offline image):
    python foundation_model/servo_lastmile.py \
        --ref-image REF.png --input-image SCENE.png \
        --output-dir runs/lastmile_smoke

Usage (offline video):
    python foundation_model/servo_lastmile.py \
        --ref-image REF.png --input-video CLIP.mp4 \
        --output-dir runs/lastmile_video

Usage (live, perception only):
    python foundation_model/servo_lastmile.py \
        --ref-image REF.png --dry-run

Usage (live, robot enabled):
    python foundation_model/servo_lastmile.py \
        --ref-image REF.png --hand-eye config/hand_eye.npy

Tests:
    python -m pytest foundation_model/test_servo_lastmile.py -v
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

# macOS / multi-OpenMP workaround. Several deps (torch, opencv, numpy/MKL,
# transformers) ship their own libomp; importing two of them in the same
# process aborts with OMP error #15. Anthropic's own docs and Apple's
# clang-OpenMP guidance both endorse this knob as the pragmatic fix; it is
# safe for our workload (we are not relying on OpenMP thread affinity).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(name: str = "lastmile",
                  log_dir: str = "logs",
                  level: int = logging.DEBUG) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"servo_lastmile_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    # Bridge: attach the same handlers to 'semvs' (servo_pipeline_sam3's
    # logger) so the camera / SAM3 / robot log lines appear in our log file
    # and console when live mode imports that module. Without this, the
    # live driver looks dead because all interesting messages get dropped.
    semvs_logger = logging.getLogger("semvs")
    semvs_logger.setLevel(level)
    semvs_logger.handlers.clear()
    for h in logger.handlers:
        semvs_logger.addHandler(h)
    semvs_logger.propagate = False

    logger.info("Logging to %s", os.path.abspath(log_path))
    logger._semvs_log_path = os.path.abspath(log_path)
    return logger


log = logging.getLogger("lastmile")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# LOCK trigger: object must be safely fully framed AND within stereo depth
# range AND area-fraction in a sensible band, for N consecutive frames.
LOCK_BORDER_MARGIN_PX     = 8
LOCK_BORDER_MARGIN_FRAC   = 0.05
LOCK_AREA_MIN_FRAC        = 0.05
LOCK_AREA_MAX_FRAC        = 0.25
LOCK_DEPTH_MIN_MM         = 100.0
# Bumped from 400 -> 700 mm because the rig's home pose puts the camera
# ~450 mm above the table, well outside the original [150, 400] band, so
# the LOCK trigger never fired and the FSM sat in FAR forever.
LOCK_DEPTH_MAX_MM         = 700.0
LOCK_CONSEC_FRAMES        = 3

# NEAR trigger: any of border-touch / area>30% / Z<80mm flips state
NEAR_AREA_MAX_FRAC        = 0.30
NEAR_DEPTH_MM             = 80.0

# TERMINAL trigger
TERM_DEPTH_MM             = 30.0
TERM_FUSED_ERR_PX         = 8.0

# CoTracker3 init
COTRACKER_NUM_POINTS      = 80
COTRACKER_VIS_THRESH      = 0.5
COTRACKER_CONF_THRESH     = 0.5

# Fusion
FUSION_OVERRIDE_PX        = 80.0
FUSION_GEO_MED_ITERS      = 12
FUSION_GEO_MED_EPS        = 1e-3

# Watchdog
WATCHDOG_DISAGREE_PX      = 60.0
WATCHDOG_DISAGREE_FRAMES  = 5

# DINOv2 best-buddy
DINOV2_PATCH_SIZE         = 14
DINOV2_SIM_THRESH         = 0.55
DINOV2_MIN_MATCHES        = 8

# Camera intrinsics fallback (HD720 ZED). Real pipeline replaces these
# with values loaded from ZED settings .conf file at startup.
DEFAULT_INTRINSICS = dict(fx=700.0, fy=700.0, cx=640.0, cy=360.0,
                          width=1280, height=720)

# Heavy-model lazy loads (CoTracker3, SAM2, DINOv2) can be globally disabled
# via env var so unit tests / CI / dev environments without weights don't
# attempt torch.hub network calls or hit OpenMP-duplicate crashes on macOS.
HEAVY_MODELS_DISABLED = os.environ.get(
    "LASTMILE_DISABLE_HEAVY_MODELS", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Result dict schema
# ---------------------------------------------------------------------------

# Result keys carried across frames and dumped to JSONL summary, mirrored
# from servo_pipeline_sam3.py result-dict for downstream tool compat.
def _empty_result() -> dict:
    return dict(
        # SAM3 path (FAR state)
        mask_np=None, depth_np=None, gdino_box=None,
        sam_score=None, similarity=None, dets_all=None,
        detector_used=None, detection_was_skipped=False,

        # Last-mile additions
        state="FAR",
        c_A=None, c_B=None, c_C=None, c_D=None,
        c_fused=None,
        weights=None,
        watchdog_alarm=False,
        plane_normal=None,
        z_mm=None,
        in_frame_fraction=None,
        best_centroid=None,  # the value the controller consumes
    )


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def weighted_geometric_median(points: list[tuple[float, float]],
                              weights: list[float],
                              max_iter: int = FUSION_GEO_MED_ITERS,
                              eps: float = FUSION_GEO_MED_EPS
                              ) -> tuple[float, float] | None:
    """
    Weiszfeld iteration for the weighted geometric median in 2D.
    Robust to a single rogue input. Returns None if no valid input.
    """
    pts = [(float(x), float(y), float(w)) for (x, y), w in zip(points, weights)
           if w is not None and w > 0 and x is not None and y is not None
           and not (math.isnan(x) or math.isnan(y))]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][0], pts[0][1]

    sx = sum(p[0] * p[2] for p in pts)
    sy = sum(p[1] * p[2] for p in pts)
    sw = sum(p[2] for p in pts)
    cx, cy = sx / sw, sy / sw

    for _ in range(max_iter):
        nx = ny = nd = 0.0
        for x, y, w in pts:
            d = math.hypot(x - cx, y - cy)
            if d < 1e-6:
                # geometric median coincides with this point: return it
                return x, y
            nx += w * x / d
            ny += w * y / d
            nd += w / d
        if nd <= 0:
            break
        new_cx, new_cy = nx / nd, ny / nd
        if math.hypot(new_cx - cx, new_cy - cy) < eps:
            cx, cy = new_cx, new_cy
            break
        cx, cy = new_cx, new_cy
    return cx, cy


def project_3d_to_pixel(pt_cam: np.ndarray,
                        K: np.ndarray) -> tuple[float, float] | None:
    """
    Pinhole projection of a 3D point in camera frame (mm or m, units must
    match K) onto the image plane. Returns None if point is behind camera.
    """
    if pt_cam.shape != (3,):
        pt_cam = pt_cam.reshape(3)
    z = float(pt_cam[2])
    if z <= 1e-6:
        return None
    u = float(K[0, 0]) * pt_cam[0] / z + float(K[0, 2])
    v = float(K[1, 1]) * pt_cam[1] / z + float(K[1, 2])
    return u, v


def transform_point(T: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a 3D point."""
    pt_h = np.concatenate([pt.reshape(3), [1.0]])
    out = T @ pt_h
    return out[:3] / max(out[3], 1e-9)


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def uniform_mask_samples(mask: np.ndarray, n: int,
                         rng: np.random.Generator | None = None
                         ) -> np.ndarray:
    """
    Sample n points uniformly inside a binary mask. Returns Nx2 array of
    (x, y) image coordinates. If mask has fewer than n nonzero pixels,
    samples with replacement.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    replace = xs.size < n
    idx = rng.choice(xs.size, size=n, replace=replace)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def fit_plane_ransac(points_xyz: np.ndarray,
                     thresh_mm: float = 4.0,
                     max_iter: int = 100,
                     rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    RANSAC plane fit. Returns (normal, point_on_plane). Normal is the
    unit vector pointing toward the camera (positive z component).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = points_xyz.shape[0]
    if n < 3:
        return None

    best_inliers = -1
    best_normal = None
    best_centroid = None

    for _ in range(max_iter):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points_xyz[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        dist = np.abs(points_xyz @ normal + d)
        inliers = int((dist < thresh_mm).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_centroid = points_xyz[dist < thresh_mm].mean(axis=0)

    if best_normal is None:
        return None

    # orient so positive z points toward camera (toward origin)
    if best_normal[2] > 0:
        best_normal = -best_normal

    return best_normal, best_centroid


def bbox_touches_border(box: np.ndarray, w: int, h: int,
                        margin_px: int = LOCK_BORDER_MARGIN_PX,
                        margin_frac: float = LOCK_BORDER_MARGIN_FRAC) -> bool:
    """Is the bbox within margin px of any image border?"""
    x1, y1, x2, y2 = [float(v) for v in box]
    side = max(x2 - x1, y2 - y1)
    margin = max(margin_px, int(margin_frac * side))
    return (x1 < margin or y1 < margin
            or x2 > w - margin or y2 > h - margin)


def bbox_area_fraction(box: np.ndarray, w: int, h: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, w * h)


def in_frame_fraction(points_xy: np.ndarray, w: int, h: int) -> float:
    """Fraction of points lying within image bounds. Used for confidence."""
    if points_xy.size == 0:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = ((x >= 0) & (x < w) & (y >= 0) & (y < h)).sum()
    return float(inside) / float(points_xy.shape[0])


# ---------------------------------------------------------------------------
# Hand-eye / intrinsics IO
# ---------------------------------------------------------------------------

def load_hand_eye(path: str | None) -> np.ndarray:
    """
    Load 4x4 T_ee_cam from .npy or .yaml. Returns identity if path is None
    or file is missing (with a warning so signal A degrades visibly).
    """
    if path is None:
        log.warning("Hand-eye: no path supplied; using identity. "
                    "Signal A (locked-3D projection) will be unreliable. "
                    "Run a one-time calibrateHandEye to populate.")
        return np.eye(4)
    if not os.path.exists(path):
        log.warning("Hand-eye: file %s missing; using identity.", path)
        return np.eye(4)
    if path.endswith(".npy"):
        T = np.load(path).astype(np.float64)
    elif path.endswith((".yaml", ".yml")):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        T = fs.getNode("T_ee_cam").mat()
        fs.release()
    else:
        raise ValueError(f"Unsupported hand-eye format: {path}")
    if T.shape != (4, 4):
        raise ValueError(f"Hand-eye matrix must be 4x4, got {T.shape}")
    log.info("Hand-eye: loaded T_ee_cam from %s", path)
    return T


def make_intrinsics_matrix(intr: dict) -> np.ndarray:
    K = np.array([
        [intr["fx"],        0.0, intr["cx"]],
        [       0.0, intr["fy"], intr["cy"]],
        [       0.0,        0.0,        1.0],
    ], dtype=np.float64)
    return K


# ---------------------------------------------------------------------------
# LockState: snapshot at the FAR -> NEAR transition
# ---------------------------------------------------------------------------

@dataclass
class LockState:
    """
    Captured at the last fully-framed clean frame. Frozen for the rest of
    the approach. All four NEAR signals derive their reference from this.
    """
    # Image-space references
    mask_uint8:   np.ndarray         # (H, W) {0,255}
    centroid_px:  tuple[float, float]
    bbox_xyxy:    np.ndarray          # (4,)
    crop_bgr:     np.ndarray          # bbox + 10% pad
    crop_bbox:    np.ndarray          # (4,) padded crop in original coords

    # 3D anchor in robot base frame (the deterministic spine)
    centroid_base: np.ndarray         # (3,) mm
    z_at_lock_mm:  float

    # Robot pose at lock (for sanity / debug)
    T_base_ee:     np.ndarray         # (4, 4)

    # CoTracker init: uniform-sampled points inside the mask
    init_points:   np.ndarray         # (N, 2)

    # Optional cached features (populated by signal D)
    dinov2_ref_features: Any = None

    # Frame metadata
    frame_idx:     int = -1
    timestamp:     float = 0.0


# ---------------------------------------------------------------------------
# Signal A: locked-3D centroid projection
# ---------------------------------------------------------------------------

class SignalA:
    """
    Project the locked 3D centroid (in base frame) into the current image
    using the latest EE pose and known T_ee_cam + K.

    This is the deterministic spine of the fusion: it does not depend on
    the camera seeing the object at all. Reliability is bounded only by
    the accuracy of hand-eye calibration and EE pose feedback.
    """

    def __init__(self, T_ee_cam: np.ndarray, K: np.ndarray):
        self.T_ee_cam = T_ee_cam
        self.T_cam_ee = invert_transform(T_ee_cam)
        self.K = K

    def predict(self, lock: LockState, T_base_ee_now: np.ndarray
                ) -> tuple[float, float] | None:
        # cam-from-base = (base-from-ee * ee-from-cam)^-1
        T_base_cam = T_base_ee_now @ self.T_ee_cam
        T_cam_base = invert_transform(T_base_cam)
        pt_cam = transform_point(T_cam_base, lock.centroid_base)
        return project_3d_to_pixel(pt_cam, self.K)


# ---------------------------------------------------------------------------
# Signal B: CoTracker3 online (with LK fallback)
# ---------------------------------------------------------------------------

class SignalB:
    """
    Online point tracking from a uniform-sampled seed inside the LOCK mask.

    Primary: CoTracker3 in online mode (per-point visibility + confidence).
    Fallback: pyramidal LK optical flow (no visibility flag, so we infer
    visibility from frame-bound + flow-error heuristics).

    The visibility-weighted centroid of the *currently-visible* points is
    an unbiased estimator of the true mask centroid because the seed was
    sampled uniformly inside the mask.
    """

    def __init__(self, prefer_cotracker: bool = True):
        self.prefer_cotracker = prefer_cotracker
        self._cotracker = None
        self._cot_state = None
        self._cot_failed = False
        self._lk_prev_gray = None
        self._points = None     # (N, 2)
        self._visible = None    # (N,) bool
        self._mode = "uninit"

    def reset(self, lock: LockState, frame_bgr: np.ndarray):
        self._points = lock.init_points.copy()
        self._visible = np.ones(self._points.shape[0], dtype=bool)
        self._lk_prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._cot_state = None

        if self.prefer_cotracker and not self._cot_failed and not HEAVY_MODELS_DISABLED:
            try:
                self._cotracker = self._lazy_load_cotracker()
                self._mode = "cotracker"
                # CoTracker3 expects torch tensors; we keep state externally
                # and call its online predictor frame-by-frame.
                log.info("Signal B: CoTracker3 online initialized with %d points",
                         self._points.shape[0])
                return
            except Exception as e:
                log.warning("Signal B: CoTracker3 unavailable (%s); falling "
                            "back to LK optical flow.", e)
                self._cot_failed = True

        self._mode = "lk"
        log.info("Signal B: LK optical flow initialized with %d points",
                 self._points.shape[0])

    @staticmethod
    def _lazy_load_cotracker():
        # CoTracker3 ships via torch.hub at facebookresearch/co-tracker.
        # If unavailable (offline / no network / no torch), the caller
        # falls back to LK. We never crash the pipeline on this path.
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online")
        model = model.to(device).eval()
        return dict(model=model, device=device, torch=torch)

    def step(self, frame_bgr: np.ndarray
             ) -> tuple[tuple[float, float] | None, float, np.ndarray, np.ndarray]:
        """
        Returns (centroid_xy, confidence_in_[0,1], points_now, visible_mask).
        confidence is the visible fraction × mean per-point confidence.
        """
        if self._points is None or self._mode == "uninit":
            return None, 0.0, np.zeros((0, 2)), np.zeros(0, dtype=bool)

        if self._mode == "cotracker":
            return self._step_cotracker(frame_bgr)
        return self._step_lk(frame_bgr)

    def _step_lk(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._lk_prev_gray is None or self._points.shape[0] == 0:
            self._lk_prev_gray = gray
            return None, 0.0, self._points, self._visible

        prev_pts = self._points.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._lk_prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      30, 0.01))
        if next_pts is None:
            self._lk_prev_gray = gray
            return None, 0.0, self._points, self._visible

        h, w = gray.shape
        new_pts = next_pts.reshape(-1, 2)
        ok = status.reshape(-1).astype(bool)
        # frame-bound check
        in_bounds = ((new_pts[:, 0] >= 0) & (new_pts[:, 0] < w)
                     & (new_pts[:, 1] >= 0) & (new_pts[:, 1] < h))
        # error-bound check (LK reports per-point match error)
        err_flat = err.reshape(-1) if err is not None else np.zeros(len(new_pts))
        good_err = err_flat < 30.0
        # visibility: AND of all three
        visible = self._visible & ok & in_bounds & good_err
        self._points = new_pts
        self._visible = visible
        self._lk_prev_gray = gray

        if int(visible.sum()) < 4:
            return None, 0.0, self._points, self._visible

        vis_pts = self._points[visible]
        cx, cy = float(vis_pts[:, 0].mean()), float(vis_pts[:, 1].mean())
        conf = float(visible.sum()) / float(len(visible))
        return (cx, cy), conf, self._points, self._visible

    def _step_cotracker(self, frame_bgr: np.ndarray):
        # CoTracker3 online predictor consumes a 16-frame sliding window.
        # For a single-frame interface we maintain a rolling buffer; if it
        # has fewer than 2 frames we return no estimate and seed the buffer.
        # Implementation kept compact: heavy-lifting deferred to torch.hub
        # weights at runtime; fallback handles the degraded path.
        try:
            ct = self._cotracker
            torch = ct["torch"]
            model = ct["model"]
            device = ct["device"]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # (1, 1, 3, H, W), float in [0, 1]
            frame_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            frame_t = frame_t.to(device).float() / 255.0
            queries = torch.from_numpy(np.concatenate([
                np.zeros((self._points.shape[0], 1)),
                self._points
            ], axis=1)).to(device).float().unsqueeze(0)

            with torch.no_grad():
                pred = model(video_chunk=frame_t, queries=queries,
                             is_first_step=(self._cot_state is None))
            self._cot_state = pred  # opaque per-implementation state

            # CoTracker3 returns tracks (B, T, N, 2) and visibility (B, T, N).
            # Take the last timestep.
            tracks = pred[0] if isinstance(pred, tuple) else pred.tracks
            vis = pred[1] if isinstance(pred, tuple) else pred.visibility

            tracks_np = tracks.detach().cpu().numpy()[0, -1]   # (N, 2)
            vis_np = vis.detach().cpu().numpy()[0, -1]         # (N,)

            self._points = tracks_np.astype(np.float32)
            visible = vis_np > COTRACKER_VIS_THRESH
            self._visible = visible

            # CoTracker3 reports a predicted position for EVERY tracked point,
            # including occluded / off-frame ones (extrapolated via cross-track
            # attention). Using all positions, not just the visible subset,
            # gives an UNBIASED centroid estimator: the original seed was
            # uniform inside the mask, so the mean of all tracked positions
            # remains the true mask centroid even when half exit the frame.
            if self._points.shape[0] < 4:
                return None, 0.0, self._points, self._visible

            cx = float(self._points[:, 0].mean())
            cy = float(self._points[:, 1].mean())
            # Confidence still reflects how much the model is sure of:
            # mean per-point visibility (treated as soft confidence).
            conf = float(vis_np.mean())
            return (cx, cy), conf, self._points, self._visible
        except Exception as e:
            # Soft fall-back to LK on any runtime hiccup
            if not self._cot_failed:
                log.warning("Signal B: CoTracker3 step failed (%s); "
                            "downgrading to LK for the rest of approach.", e)
            self._cot_failed = True
            self._mode = "lk"
            return self._step_lk(frame_bgr)


# ---------------------------------------------------------------------------
# Signal C: SAM2 mask propagation (watchdog)
# ---------------------------------------------------------------------------

class SignalC:
    """
    SAM2 mask propagation seeded from the LOCK SAM3 mask. Its role here is
    a sanity-check watchdog, not a primary controller signal: if the
    propagated mask centroid disagrees with the fused estimate by more
    than WATCHDOG_DISAGREE_PX for WATCHDOG_DISAGREE_FRAMES consecutive
    frames, raise an alarm.
    """

    def __init__(self):
        self._predictor = None
        self._failed = False
        self._prev_logits = None
        self._prev_bbox = None

    def reset(self, lock: LockState, frame_bgr: np.ndarray):
        self._prev_logits = None
        self._prev_bbox = lock.bbox_xyxy.copy()
        # Bridge: convert the SAM3 binary mask into SAM2 logit hint.
        # We rescale {0,255} -> {-10,+10} and resize to 256x256.
        m = (lock.mask_uint8 > 127).astype(np.float32)
        m_resized = cv2.resize(m, (256, 256),
                               interpolation=cv2.INTER_LINEAR)
        # offset to logit space; SAM2's mask_input is pre-sigmoid logits
        self._prev_logits = (m_resized * 20.0 - 10.0)[None, ...]
        if HEAVY_MODELS_DISABLED:
            log.info("Signal C: heavy models disabled by env; skipping SAM2 load.")
            self._failed = True
            return
        try:
            self._predictor = self._lazy_load_sam2()
            log.info("Signal C: SAM2 propagation initialized.")
        except Exception as e:
            log.warning("Signal C: SAM2 unavailable (%s); watchdog disabled.", e)
            self._failed = True

    @staticmethod
    def _lazy_load_sam2():
        # Reuse the existing SAM2 setup from sam2_tracking_method.py if
        # available. This keeps a single source of truth for SAM2 config.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        third_party = os.path.join(os.path.dirname(__file__), "third-party")
        cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        ckpt = os.path.join(third_party, "sam2/checkpoints/sam2.1_hiera_large.pt")
        model = build_sam2(cfg, ckpt, device=device)
        return SAM2ImagePredictor(model)

    def step(self, frame_bgr: np.ndarray
             ) -> tuple[tuple[float, float] | None, float, np.ndarray | None]:
        if self._failed or self._predictor is None or self._prev_logits is None:
            return None, 0.0, None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(rgb)
            kwargs = dict(mask_input=self._prev_logits, multimask_output=False)
            if self._prev_bbox is not None:
                x1, y1, x2, y2 = self._prev_bbox
                bw, bh = x2 - x1, y2 - y1
                pad_x = int(bw * 0.10)
                pad_y = int(bh * 0.10)
                h, w = frame_bgr.shape[:2]
                kwargs["box"] = np.array([
                    max(0, x1 - pad_x), max(0, y1 - pad_y),
                    min(w, x2 + pad_x), min(h, y2 + pad_y)],
                    dtype=np.float32)

            masks, scores, logits = self._predictor.predict(**kwargs)
            best = int(np.argmax(scores))
            mask = (masks[best] > 0).astype(np.uint8) * 255
            self._prev_logits = logits[best:best + 1]

            ys, xs = np.where(mask > 0)
            if xs.size < 50:
                return None, 0.0, mask
            cx, cy = float(xs.mean()), float(ys.mean())
            # update bbox for next-frame loose hint
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
            self._prev_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            score = float(scores[best])
            # NOTE: this centroid is biased under partial framing; the
            # caller must downweight via (1 - boundary_fraction).
            return (cx, cy), score, mask
        except Exception as e:
            log.warning("Signal C: SAM2 step failed (%s); disabling watchdog.", e)
            self._failed = True
            return None, 0.0, None


# ---------------------------------------------------------------------------
# Signal D: DINOv2 best-buddy correspondence
# ---------------------------------------------------------------------------

class SignalD:
    """
    Dense DINOv2 patch-feature matching against the LOCK crop. Bidirectional
    best-buddy pairs give robust correspondences; centroid of matched points
    in the current frame is the centroid estimate.

    Robust to partial framing: patches that have moved off-frame produce no
    match; the patches that DO match still match correctly.
    """

    def __init__(self):
        self._model = None
        self._failed = False
        self._ref_feat = None     # (Hp, Wp, C) cached at lock
        self._ref_grid = None     # (Hp, Wp, 2) ref patch centers
        self._ref_size = None     # (W, H) of the cropped ref image
        self._ref_offset = None   # (x0, y0) crop offset in original frame

    def reset(self, lock: LockState):
        self._ref_feat = None
        self._ref_grid = None
        self._ref_offset = (float(lock.crop_bbox[0]), float(lock.crop_bbox[1]))
        if HEAVY_MODELS_DISABLED:
            log.info("Signal D: heavy models disabled by env; skipping DINOv2 load.")
            self._failed = True
            return
        try:
            self._model = self._lazy_load_dinov2()
            self._ref_feat, self._ref_grid, self._ref_size = (
                self._embed_image(lock.crop_bgr))
            log.info("Signal D: DINOv2 reference features cached "
                     "(%d x %d patches).",
                     self._ref_feat.shape[1], self._ref_feat.shape[0])
        except Exception as e:
            log.warning("Signal D: DINOv2 unavailable (%s); disabled.", e)
            self._failed = True

    @staticmethod
    def _lazy_load_dinov2():
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = model.to(device).eval()
        return dict(model=model, device=device, torch=torch)

    def _embed_image(self, image_bgr: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        torch = self._model["torch"]
        model = self._model["model"]
        device = self._model["device"]

        H, W = image_bgr.shape[:2]
        # round to multiple of patch size
        ph = (H // DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
        pw = (W // DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
        if ph == 0 or pw == 0:
            raise ValueError(f"image too small for DINOv2: {W}x{H}")
        img = cv2.resize(image_bgr, (pw, ph))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = ((t - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.forward_features(t)["x_norm_patchtokens"]
        Hp = ph // DINOV2_PATCH_SIZE
        Wp = pw // DINOV2_PATCH_SIZE
        feats = feats[0].reshape(Hp, Wp, -1).cpu().numpy()
        # Patch-center grid in the *resized* image; map back to original
        # via scale = (W / pw, H / ph) so the caller can convert.
        ys = (np.arange(Hp) + 0.5) * DINOV2_PATCH_SIZE * (H / ph)
        xs = (np.arange(Wp) + 0.5) * DINOV2_PATCH_SIZE * (W / pw)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        grid = np.stack([gx, gy], axis=-1)
        # L2-normalize for cosine matching
        feats /= (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
        return feats, grid, (W, H)

    def step(self, frame_bgr: np.ndarray
             ) -> tuple[tuple[float, float] | None, float, int]:
        if self._failed or self._ref_feat is None:
            return None, 0.0, 0

        try:
            cur_feat, cur_grid, _ = self._embed_image(frame_bgr)
        except Exception as e:
            log.warning("Signal D: embed failed (%s).", e)
            return None, 0.0, 0

        Rh, Rw, C = self._ref_feat.shape
        Ch, Cw, _ = cur_feat.shape
        ref_flat = self._ref_feat.reshape(-1, C)         # (Rh*Rw, C)
        cur_flat = cur_feat.reshape(-1, C)               # (Ch*Cw, C)

        # Cosine similarity (already L2-normalized)
        sim = ref_flat @ cur_flat.T                       # (Rh*Rw, Ch*Cw)

        # Best-buddy pairs: ref->cur argmax must equal cur->ref argmax
        ref_to_cur = np.argmax(sim, axis=1)               # (Rh*Rw,)
        cur_to_ref = np.argmax(sim, axis=0)               # (Ch*Cw,)

        ref_idx = np.arange(ref_flat.shape[0])
        bb_mask = (cur_to_ref[ref_to_cur] == ref_idx)
        bb_sim = sim[ref_idx, ref_to_cur]
        keep = bb_mask & (bb_sim > DINOV2_SIM_THRESH)
        n_match = int(keep.sum())
        if n_match < DINOV2_MIN_MATCHES:
            return None, 0.0, n_match

        # Map matched cur indices to pixel centers (in CURRENT-frame coords)
        cur_idx = ref_to_cur[keep]
        cur_xy = cur_grid.reshape(-1, 2)[cur_idx]         # (n_match, 2)
        weights = bb_sim[keep]
        wx = float(np.average(cur_xy[:, 0], weights=weights))
        wy = float(np.average(cur_xy[:, 1], weights=weights))
        return (wx, wy), float(weights.mean()), n_match


# ---------------------------------------------------------------------------
# Fusion arbitration
# ---------------------------------------------------------------------------

@dataclass
class SignalReading:
    name: str
    centroid: tuple[float, float] | None
    weight: float
    extra: dict = field(default_factory=dict)


def fuse_centroids(readings: list[SignalReading],
                   c_A: tuple[float, float] | None,
                   override_px: float = FUSION_OVERRIDE_PX
                   ) -> tuple[tuple[float, float] | None, dict]:
    """
    Weighted geometric median fusion + Signal-A sanity override.

    Returns (fused_xy, info). info has 'override_applied' bool and
    'used_weights' dict for logging.
    """
    pts = []
    weights = []
    used = {}
    for r in readings:
        if r.centroid is None or r.weight <= 0:
            used[r.name] = 0.0
            continue
        pts.append(r.centroid)
        weights.append(r.weight)
        used[r.name] = r.weight

    fused = weighted_geometric_median(pts, weights)
    info = dict(override_applied=False, used_weights=used)

    if fused is None:
        return None, info

    # Signal A sanity gate: trust geometry over perception in NEAR
    if c_A is not None:
        d = math.hypot(fused[0] - c_A[0], fused[1] - c_A[1])
        if d > override_px:
            info["override_applied"] = True
            info["override_distance_px"] = d
            return c_A, info

    return fused, info


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State:
    FAR  = "FAR"
    LOCK = "LOCK"      # transient single-frame state; immediately -> NEAR
    NEAR = "NEAR"
    TERM = "TERMINAL"


class StateMachine:
    """
    FAR  -> LOCK  -> NEAR  -> TERMINAL
    Hysteresis: never returns to a previous state during a single approach.
    """

    def __init__(self):
        self.state = State.FAR
        self._consec_lock_candidate = 0
        self._watchdog_disagree = 0
        self.lock_state: LockState | None = None
        self.frame_idx = 0

    def reset(self):
        self.state = State.FAR
        self._consec_lock_candidate = 0
        self._watchdog_disagree = 0
        self.lock_state = None
        self.frame_idx = 0

    def evaluate_far(self, sam3_result: dict, depth_at_centroid_mm: float | None,
                     image_shape: tuple) -> bool:
        """
        Returns True if this frame is a clean lock candidate (sustained
        for LOCK_CONSEC_FRAMES will trigger transition).
        """
        h, w = image_shape[:2]
        box = sam3_result.get("gdino_box")
        mask = sam3_result.get("mask_np")
        if box is None or mask is None:
            self._consec_lock_candidate = 0
            return False

        if bbox_touches_border(np.asarray(box), w, h):
            self._consec_lock_candidate = 0
            return False

        af = bbox_area_fraction(np.asarray(box), w, h)
        if not (LOCK_AREA_MIN_FRAC <= af <= LOCK_AREA_MAX_FRAC):
            self._consec_lock_candidate = 0
            return False

        if depth_at_centroid_mm is None:
            self._consec_lock_candidate = 0
            return False
        if not (LOCK_DEPTH_MIN_MM <= depth_at_centroid_mm <= LOCK_DEPTH_MAX_MM):
            self._consec_lock_candidate = 0
            return False

        self._consec_lock_candidate += 1
        return self._consec_lock_candidate >= LOCK_CONSEC_FRAMES

    def evaluate_near_to_term(self, c_fused: tuple[float, float] | None,
                              z_mm: float | None,
                              image_center: tuple[float, float]) -> bool:
        if z_mm is not None and z_mm < TERM_DEPTH_MM:
            return True
        if c_fused is not None:
            err = math.hypot(c_fused[0] - image_center[0],
                             c_fused[1] - image_center[1])
            if err < TERM_FUSED_ERR_PX:
                return True
        return False

    def near_trigger_fired(self, sam3_result: dict, z_mm: float | None,
                           image_shape: tuple) -> bool:
        """Any of: border, area>30%, depth<80mm."""
        h, w = image_shape[:2]
        box = sam3_result.get("gdino_box")
        if box is not None:
            box_np = np.asarray(box)
            if bbox_touches_border(box_np, w, h):
                return True
            if bbox_area_fraction(box_np, w, h) > NEAR_AREA_MAX_FRAC:
                return True
        if z_mm is not None and z_mm < NEAR_DEPTH_MM:
            return True
        return False

    def update_watchdog(self, fused_xy: tuple[float, float] | None,
                        c_C: tuple[float, float] | None) -> bool:
        if fused_xy is None or c_C is None:
            self._watchdog_disagree = 0
            return False
        d = math.hypot(fused_xy[0] - c_C[0], fused_xy[1] - c_C[1])
        if d > WATCHDOG_DISAGREE_PX:
            self._watchdog_disagree += 1
        else:
            self._watchdog_disagree = 0
        return self._watchdog_disagree >= WATCHDOG_DISAGREE_FRAMES


# ---------------------------------------------------------------------------
# TERMINAL phase: plane fit + normal alignment
# ---------------------------------------------------------------------------

def terminal_plane_fit(depth_mm: np.ndarray, K: np.ndarray,
                       center_xy: tuple[float, float],
                       patch_half: int = 50,
                       valid_min_mm: float = 5.0,
                       valid_max_mm: float = 1000.0,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fit a plane to a depth patch around center_xy. Returns (normal, centroid)
    in camera frame (mm) or None if insufficient valid points.
    """
    h, w = depth_mm.shape
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x1 = max(0, cx - patch_half)
    x2 = min(w, cx + patch_half)
    y1 = max(0, cy - patch_half)
    y2 = min(h, cy + patch_half)
    patch = depth_mm[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    yy, xx = np.mgrid[y1:y2, x1:x2]
    valid = (patch > valid_min_mm) & (patch < valid_max_mm) & np.isfinite(patch)
    if int(valid.sum()) < 50:
        return None

    z = patch[valid]
    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cxK, cyK = K[0, 2], K[1, 2]
    X = (u - cxK) * z / fx
    Y = (v - cyK) * z / fy
    pts = np.stack([X, Y, z], axis=1)
    return fit_plane_ransac(pts, thresh_mm=4.0, max_iter=80, rng=rng)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class LastMilePipeline:
    """
    Runs the full FAR -> LOCK -> NEAR -> TERMINAL pipeline frame-by-frame.

    Kept dependency-light: SAM3 + ZED + xArm hookups are passed in via
    callables so this orchestrator can be exercised in pure-Python tests
    with synthetic stand-ins.
    """

    def __init__(self,
                 sam3_runner: Callable[[np.ndarray], dict],
                 ee_pose_provider: Callable[[], np.ndarray] | None = None,
                 depth_provider: Callable[[np.ndarray], np.ndarray | None] | None = None,
                 hand_eye_path: str | None = None,
                 intrinsics: dict | None = None):
        self.sam3_runner = sam3_runner
        self.ee_pose_provider = ee_pose_provider or (lambda: np.eye(4))
        self.depth_provider = depth_provider or (lambda f: None)
        self.T_ee_cam = load_hand_eye(hand_eye_path)
        self.intrinsics = intrinsics or DEFAULT_INTRINSICS
        self.K = make_intrinsics_matrix(self.intrinsics)

        self.fsm = StateMachine()
        self.signal_A = SignalA(self.T_ee_cam, self.K)
        self.signal_B = SignalB()
        self.signal_C = SignalC()
        self.signal_D = SignalD()

    def reset(self):
        self.fsm.reset()
        self.signal_B = SignalB()
        self.signal_C = SignalC()
        self.signal_D = SignalD()

    def _depth_at_px(self, depth_map: np.ndarray | None,
                     pt: tuple[float, float]) -> float | None:
        if depth_map is None or pt is None:
            return None
        h, w = depth_map.shape
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < w and 0 <= y < h):
            return None
        # 5x5 robust median
        x1, x2 = max(0, x - 2), min(w, x + 3)
        y1, y2 = max(0, y - 2), min(h, y + 3)
        patch = depth_map[y1:y2, x1:x2]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _enter_lock(self, frame_bgr: np.ndarray, sam3_result: dict,
                    depth_map: np.ndarray | None) -> LockState | None:
        h, w = frame_bgr.shape[:2]
        mask = sam3_result["mask_np"]
        box = np.asarray(sam3_result["gdino_box"], dtype=np.float32)

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())

        z_mm = self._depth_at_px(depth_map, (cx, cy))
        if z_mm is None:
            log.warning("LOCK refused: depth unavailable at centroid.")
            return None

        # 3D centroid in camera frame, then base frame
        K = self.K
        X = (cx - K[0, 2]) * z_mm / K[0, 0]
        Y = (cy - K[1, 2]) * z_mm / K[1, 1]
        pt_cam = np.array([X, Y, z_mm])
        T_base_ee = self.ee_pose_provider()
        T_base_cam = T_base_ee @ self.T_ee_cam
        pt_base = transform_point(T_base_cam, pt_cam)

        # Padded crop for refs (10% pad)
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        px = int(0.10 * bw)
        py = int(0.10 * bh)
        cx1 = max(0, int(x1) - px); cy1 = max(0, int(y1) - py)
        cx2 = min(w, int(x2) + px); cy2 = min(h, int(y2) + py)
        crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
        crop_bbox = np.array([cx1, cy1, cx2, cy2], dtype=np.float32)

        rng = np.random.default_rng(42)
        init_pts = uniform_mask_samples(mask, COTRACKER_NUM_POINTS, rng)

        lock = LockState(
            mask_uint8=mask.copy() if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8),
            centroid_px=(cx, cy),
            bbox_xyxy=box.copy(),
            crop_bgr=crop,
            crop_bbox=crop_bbox,
            centroid_base=pt_base,
            z_at_lock_mm=z_mm,
            T_base_ee=T_base_ee.copy(),
            init_points=init_pts,
            frame_idx=self.fsm.frame_idx,
            timestamp=time.time(),
        )

        # Initialize NEAR-state signals
        self.signal_B.reset(lock, frame_bgr)
        self.signal_C.reset(lock, frame_bgr)
        self.signal_D.reset(lock)

        log.info("LOCK captured @ frame %d: c_px=(%.1f, %.1f), z=%.1fmm, "
                 "3D_base=(%.1f, %.1f, %.1f) mm, %d init points",
                 self.fsm.frame_idx, cx, cy, z_mm,
                 pt_base[0], pt_base[1], pt_base[2], init_pts.shape[0])
        return lock

    def step(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a single frame. Returns the result dict (see _empty_result).
        """
        res = _empty_result()
        h, w = frame_bgr.shape[:2]
        self.fsm.frame_idx += 1
        depth_map = self.depth_provider(frame_bgr)

        # ============ FAR ============
        if self.fsm.state == State.FAR:
            sam3_res = self.sam3_runner(frame_bgr)
            res.update(sam3_res)
            res["state"] = State.FAR

            cx_cy = None
            mask = sam3_res.get("mask_np")
            if mask is not None and mask.any():
                ys, xs = np.where(mask > 0)
                cx_cy = (float(xs.mean()), float(ys.mean()))
            res["best_centroid"] = cx_cy

            z_mm = self._depth_at_px(depth_map, cx_cy) if cx_cy else None
            res["z_mm"] = z_mm

            if self.fsm.evaluate_far(sam3_res, z_mm, frame_bgr.shape):
                lock = self._enter_lock(frame_bgr, sam3_res, depth_map)
                if lock is not None:
                    self.fsm.lock_state = lock
                    self.fsm.state = State.NEAR
                    res["state"] = State.NEAR
                    res["best_centroid"] = lock.centroid_px
            return res

        # ============ NEAR ============
        if self.fsm.state == State.NEAR:
            res["state"] = State.NEAR
            lock = self.fsm.lock_state

            # Signal A: locked-3D projection
            T_base_ee_now = self.ee_pose_provider()
            c_A = self.signal_A.predict(lock, T_base_ee_now)
            res["c_A"] = c_A

            # Signal B: CoTracker3 / LK
            c_B, w_B, pts_now, vis = self.signal_B.step(frame_bgr)
            res["c_B"] = c_B

            # Signal C: SAM2 mask propagation (watchdog)
            c_C, sam2_score, mask_C = self.signal_C.step(frame_bgr)
            res["c_C"] = c_C
            if mask_C is not None:
                res["mask_np"] = mask_C

            # Signal D: DINOv2 best-buddy
            c_D, sim_D, n_match = self.signal_D.step(frame_bgr)
            res["c_D"] = c_D

            # Compute weights
            ifv = in_frame_fraction(pts_now, w, h) if pts_now is not None else 0.0
            res["in_frame_fraction"] = ifv

            # boundary fraction for SAM2 down-weighting
            boundary_frac = 0.0
            if mask_C is not None:
                edge_pixels = (mask_C[0, :].sum() + mask_C[-1, :].sum()
                               + mask_C[:, 0].sum() + mask_C[:, -1].sum())
                total = max(1, int(mask_C.sum() / 255))
                boundary_frac = float(edge_pixels) / float(255 * (h + h + w + w))
                boundary_frac = min(1.0, boundary_frac * 8.0)  # amplify

            readings = [
                SignalReading("A", c_A, 1.0 if (c_A is not None
                              and not np.allclose(self.T_ee_cam, np.eye(4)))
                              else 0.0),
                SignalReading("B", c_B, w_B),
                SignalReading("C", c_C,
                              max(0.0, min(0.4, sam2_score * (1.0 - boundary_frac)))),
                SignalReading("D", c_D,
                              max(0.0, sim_D * min(1.0, n_match / 30.0))),
            ]
            fused, info = fuse_centroids(readings, c_A,
                                         override_px=FUSION_OVERRIDE_PX)
            res["c_fused"] = fused
            res["weights"] = info["used_weights"]
            res["best_centroid"] = fused

            # Watchdog
            alarm = self.fsm.update_watchdog(fused, c_C)
            res["watchdog_alarm"] = alarm
            if alarm:
                log.error("WATCHDOG: SAM2 disagreement sustained "
                          "for %d frames; flagging hold.",
                          self.fsm._watchdog_disagree)

            # Depth at fused centroid for terminal trigger
            z_mm = self._depth_at_px(depth_map, fused)
            res["z_mm"] = z_mm
            ic = (w / 2.0, h / 2.0)
            if self.fsm.evaluate_near_to_term(fused, z_mm, ic):
                self.fsm.state = State.TERM
                res["state"] = State.TERM
                log.info("State -> TERMINAL @ frame %d (z=%s, fused err ok)",
                         self.fsm.frame_idx,
                         f"{z_mm:.1f}mm" if z_mm else "n/a")
            return res

        # ============ TERMINAL ============
        if self.fsm.state == State.TERM:
            res["state"] = State.TERM
            ic = (w / 2.0, h / 2.0)

            # Plane fit on depth patch
            if depth_map is not None:
                fit = terminal_plane_fit(depth_map, self.K, ic)
                if fit is not None:
                    normal, centroid = fit
                    res["plane_normal"] = normal.tolist()
                    res["best_centroid"] = ic  # controller drives axially
                    log.info("TERM: normal=%s, centroid_cam=%.1f,%.1f,%.1fmm",
                             np.round(normal, 3).tolist(),
                             centroid[0], centroid[1], centroid[2])
            # Sanity watchdog still runs in TERM
            T_base_ee_now = self.ee_pose_provider()
            c_A = self.signal_A.predict(self.fsm.lock_state, T_base_ee_now)
            res["c_A"] = c_A
            return res

        return res  # unreachable


# ---------------------------------------------------------------------------
# Default SAM3 runner: reuse existing servo_pipeline_sam3.run_pipeline
# ---------------------------------------------------------------------------

def make_default_sam3_runner(ref_image_path: str | None,
                             prompt: str = "box") -> Callable[[np.ndarray], dict]:
    """
    Build a SAM3 runner that wraps the existing run_pipeline from
    servo_pipeline_sam3.py. If that module fails to import (no model
    weights, etc.), returns a stub that yields empty detections so the
    state machine path is still exercisable.
    """
    if HEAVY_MODELS_DISABLED:
        log.info("SAM3 import skipped (LASTMILE_DISABLE_HEAVY_MODELS=1); "
                 "using empty-detection stub.")

        def stub(_frame: np.ndarray) -> dict:
            return _empty_result()
        return stub

    try:
        from foundation_model.servo_pipeline_sam3 import (  # type: ignore
            run_pipeline, MaskTracker, process_ref_image)
    except Exception:
        try:
            from servo_pipeline_sam3 import (
                run_pipeline, MaskTracker, process_ref_image)
        except Exception as e:
            log.warning("SAM3 pipeline unavailable (%s); using stub runner.", e)

            def stub(_frame: np.ndarray) -> dict:
                return _empty_result()
            return stub

    tracker = MaskTracker()
    ref_crop = ref_features = None
    if ref_image_path and os.path.exists(ref_image_path):
        try:
            from foundation_model.servo_pipeline_sam3 import _load_ref_image  # type: ignore
        except Exception:
            from servo_pipeline_sam3 import _load_ref_image
        ref_bgr = _load_ref_image(ref_image_path)
        if ref_bgr is not None:
            ref_crop, ref_features = process_ref_image(ref_bgr, prompt)

    def runner(frame: np.ndarray) -> dict:
        return run_pipeline(frame, prompt, tracker,
                            ref_crop=ref_crop, ref_features=ref_features)

    return runner


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SemVS last-mile pipeline (FAR -> LOCK -> NEAR -> "
                    "TERMINAL fused-estimator visual servoing).")
    p.add_argument("cam_index", nargs="?", type=int, default=0,
                   help="OpenCV camera index (default: 0)")
    p.add_argument("--ref-image", "--ref_image", metavar="PATH",
                   help="Path to a reference image of the target object.")
    p.add_argument("--no-pyzed", action="store_true",
                   help="Force OpenCV camera capture even if PyZED is available")
    p.add_argument("--no-robot", action="store_true",
                   help="Run vision/camera only; do not connect to xArm")
    p.add_argument("--dry-run", action="store_true",
                   help="Run perception and logging without robot commands")
    p.add_argument("--input-image", metavar="PATH",
                   help="Run offline on a single image")
    p.add_argument("--input-dir", metavar="PATH",
                   help="Run offline on all images in a directory")
    p.add_argument("--input-video", metavar="PATH",
                   help="Run offline on a video file")
    p.add_argument("--output-dir", metavar="PATH",
                   help="Directory for offline overlays and JSON metrics")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximum offline frames to process")
    p.add_argument("--no-overlays", action="store_true",
                   help="Skip overlay PNGs during offline runs")
    p.add_argument("--prompt", default="box",
                   help="Concept prompt for SAM3 (default: 'box')")
    p.add_argument("--hand-eye", metavar="PATH",
                   help="Path to 4x4 T_ee_cam (.npy or .yaml). "
                        "Required for Signal A (locked-3D projection).")
    p.add_argument("--log-dir", default="logs",
                   help="Directory for log files (default: logs)")
    p.add_argument("--debug", action="store_true",
                   help="Per-frame debug dump under artifacts/. Saves input "
                        "frame, all SAM3 candidates, signal A/B/C/D centroids, "
                        "fused centroid, mask, overlay, and the full log file. "
                        "Offline: every frame. Live: every 30th frame. Push "
                        "artifacts/debug_lastmile_<ts>/ to share results.")
    p.add_argument("--no-window", action="store_true",
                   help="Live mode only: skip the cv2 preview window "
                        "(use over SSH without X11 forwarding).")
    return p


def _init_debug_dir(enabled: bool, args: argparse.Namespace,
                    ref_image_path: str | None) -> str | None:
    """
    Mirrors servo_pipeline_sam3._init_debug_dir but namespaced so log files
    and per-frame dumps are easy to identify.
    """
    if not enabled:
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join("artifacts", f"debug_lastmile_{ts}")
    os.makedirs(base, exist_ok=True)

    # Snapshot the invocation context so the artifact dir is self-explanatory.
    meta = dict(
        timestamp=ts,
        argv=sys.argv,
        args={k: (str(v) if isinstance(v, (np.ndarray,)) else v)
              for k, v in vars(args).items()},
        ref_image=ref_image_path,
        prompt=getattr(args, "prompt", "box"),
        hand_eye=getattr(args, "hand_eye", None),
    )
    with open(os.path.join(base, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    if ref_image_path and os.path.exists(ref_image_path):
        ref_bgr = cv2.imread(ref_image_path)
        if ref_bgr is not None:
            cv2.imwrite(os.path.join(base, "ref_image.png"), ref_bgr)

    log.info("Debug artifacts dir: %s", os.path.abspath(base))
    return base


def _finalize_debug_dir(debug_dir: str | None,
                        log_path: str | None) -> None:
    """Copy the active log file into the debug dir for self-contained sharing."""
    if debug_dir is None or log_path is None:
        return
    try:
        import shutil
        if os.path.exists(log_path):
            shutil.copy2(log_path, os.path.join(debug_dir, "run.log"))
            log.info("Copied log to %s", os.path.join(debug_dir, "run.log"))
    except Exception as e:
        log.warning("Failed to copy log to debug dir: %s", e)


def _dump_debug_frame(debug_dir: str | None, frame_idx: int,
                      frame_bgr: np.ndarray, res: dict,
                      overlay_bgr: np.ndarray | None,
                      source: str) -> None:
    """Per-frame debug dump: input frame, mask, overlay, full result JSON."""
    if debug_dir is None:
        return
    try:
        d = os.path.join(debug_dir, f"frame_{frame_idx:06d}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "input.png"), frame_bgr)
        if overlay_bgr is not None:
            cv2.imwrite(os.path.join(d, "overlay.png"), overlay_bgr)
        mask = res.get("mask_np")
        if mask is not None:
            cv2.imwrite(os.path.join(d, "mask.png"),
                        mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8))
        with open(os.path.join(d, "result.json"), "w") as fh:
            json.dump(_serialize_result({**res, "source": source}), fh, indent=2,
                      default=str)
    except Exception as e:
        log.warning("debug dump frame %d failed: %s", frame_idx, e)


def _make_overlay(frame: np.ndarray, res: dict) -> np.ndarray:
    """Render the lastmile overlay onto a BGR copy of frame and return it."""
    img = frame.copy()
    h, w = img.shape[:2]
    state = res.get("state", "?")
    cv2.putText(img, f"STATE: {state}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # SAM3 mask overlay (semi-transparent green) so the user can see what
    # the detector is locked onto.
    mask = res.get("mask_np")
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        green = np.zeros_like(img)
        green[:, :, 1] = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
        img = cv2.addWeighted(img, 0.78, green, 0.22, 0)
        cnts, _ = cv2.findContours(
            mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    # SAM3 bbox if present
    bx = res.get("gdino_box")
    if bx is not None:
        x1, y1, x2, y2 = [int(v) for v in bx]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)

    # Image center cross
    ic = (w // 2, h // 2)
    cv2.drawMarker(img, ic, (255, 80, 0), cv2.MARKER_CROSS, 22, 2)

    # Per-signal centroids (small colored circles)
    color_map = {"A": (0, 255, 0), "B": (255, 200, 0),
                 "C": (255, 0, 255), "D": (0, 200, 255)}
    for k in ("c_A", "c_B", "c_C", "c_D"):
        c = res.get(k)
        if c is None:
            continue
        cv2.circle(img, (int(c[0]), int(c[1])), 6, color_map[k[-1]], 2)
        cv2.putText(img, k[-1], (int(c[0]) + 8, int(c[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[k[-1]], 1)

    # Fused / best centroid (the value the controller consumes)
    cf = res.get("best_centroid") or res.get("c_fused")
    if cf is not None:
        gp = (int(cf[0]), int(cf[1]))
        cv2.circle(img, gp, 18, (0, 0, 255), 2)
        cv2.drawMarker(img, gp, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.putText(img, "GRASP", (gp[0] + 22, gp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        cv2.arrowedLine(img, ic, gp, (0, 220, 255), 2, tipLength=0.12)

    # Z depth readout
    z = res.get("z_mm")
    if z is not None:
        cv2.putText(img, f"z={z:.0f} mm", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    if res.get("watchdog_alarm"):
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(img, "WATCHDOG", (w // 2 - 80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return img


def _save_overlay(frame: np.ndarray, res: dict, out_path: str) -> None:
    """Backward-compatible wrapper: build overlay and write to disk."""
    cv2.imwrite(out_path, _make_overlay(frame, res))


def _serialize_result(res: dict) -> dict:
    """Convert numpy / tuples to plain python for JSONL dump."""
    out = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            out[k] = None if k in ("mask_np", "depth_np") else v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, tuple) and len(v) == 2:
            out[k] = [float(v[0]), float(v[1])]
        else:
            out[k] = v
    return out


def run_offline(args: argparse.Namespace, debug_dir: str | None = None) -> None:
    out_dir = args.output_dir or os.path.join(
        "runs", time.strftime("lastmile_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "frames.jsonl")

    sam3_runner = make_default_sam3_runner(args.ref_image, args.prompt)
    pipeline = LastMilePipeline(
        sam3_runner=sam3_runner,
        ee_pose_provider=lambda: np.eye(4),   # offline: stationary
        depth_provider=lambda f: None,        # offline: no depth
        hand_eye_path=args.hand_eye,
    )

    frames_iter = []
    if args.input_video:
        frames_iter.append(("video", args.input_video))
    elif args.input_image:
        frames_iter.append(("image", args.input_image))
    elif args.input_dir:
        for n in sorted(os.listdir(args.input_dir)):
            p = os.path.join(args.input_dir, n)
            if os.path.splitext(n.lower())[1] in {".png", ".jpg", ".jpeg"}:
                frames_iter.append(("image", p))
    else:
        log.error("No input given. Use --input-image / --input-video / --input-dir.")
        sys.exit(2)

    n = 0
    with open(jsonl_path, "w") as fh:
        for kind, path in frames_iter:
            if kind == "video":
                cap = cv2.VideoCapture(path)
                while cap.isOpened():
                    if args.max_frames is not None and n >= args.max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    n = _process_offline_frame(pipeline, frame, n, path,
                                                out_dir, fh, args, debug_dir)
                cap.release()
            else:
                frame = cv2.imread(path)
                if frame is None:
                    continue
                n = _process_offline_frame(pipeline, frame, n, path,
                                            out_dir, fh, args, debug_dir)

    log.info("Wrote %d frame summaries to %s", n, jsonl_path)


def _process_offline_frame(pipeline, frame, n, source, out_dir, fh,
                           args, debug_dir):
    res = pipeline.step(frame)
    fh.write(json.dumps(_serialize_result(res)) + "\n")
    fh.flush()
    overlay = None
    if not args.no_overlays:
        overlay_path = os.path.join(out_dir, f"overlay_{n:06d}.png")
        _save_overlay(frame, res, overlay_path)
        overlay = cv2.imread(overlay_path)
    if debug_dir is not None:
        _dump_debug_frame(debug_dir, n, frame, res, overlay, source)
    return n + 1


def _xarm_pose_to_T(pos: list[float] | tuple[float, ...]) -> np.ndarray:
    """xArm xyzrpy (mm, deg, ZYX intrinsic) -> 4x4 SE(3) in mm."""
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = pos[0], pos[1], pos[2]
    rx, ry, rz = (math.radians(pos[3]), math.radians(pos[4]),
                  math.radians(pos[5]))
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0.0],
                   [math.sin(rz),  math.cos(rz), 0.0],
                   [         0.0,           0.0, 1.0]])
    Ry = np.array([[ math.cos(ry), 0.0, math.sin(ry)],
                   [          0.0, 1.0,          0.0],
                   [-math.sin(ry), 0.0, math.cos(ry)]])
    Rx = np.array([[1.0,           0.0,            0.0],
                   [0.0,  math.cos(rx), -math.sin(rx)],
                   [0.0,  math.sin(rx),  math.cos(rx)]])
    T[:3, :3] = Rz @ Ry @ Rx
    return T


def run_live(args: argparse.Namespace, debug_dir: str | None = None) -> None:
    """
    Minimal live driver. Opens ZED with stereo depth ENABLED (lastmile
    requires real depth to LOCK), runs perception via LastMilePipeline,
    drives the xArm via the existing RobotController, and dumps lastmile-
    shaped per-frame artifacts.

    Replaces an earlier monkey-patch-on-CameraStreamer approach which had
    three hard-to-diagnose failure modes: silent logger isolation, ZED
    depth disabled in the upstream streamer (so LOCK never fired), and
    SAM3-shaped artifact dumps that didn't surface lastmile state.
    """
    try:
        try:
            from foundation_model.servo_pipeline_sam3 import (  # type: ignore
                RobotController, ZedUndistorter, ROBOT_IP, ZED_RESOLUTION)
        except Exception:
            from servo_pipeline_sam3 import (
                RobotController, ZedUndistorter, ROBOT_IP, ZED_RESOLUTION)
    except Exception as e:
        log.error("Live mode needs servo_pipeline_sam3 importable: %s", e)
        sys.exit(2)

    if args.ref_image and not os.path.exists(args.ref_image):
        log.error("Reference image not found: %s", args.ref_image)
        sys.exit(1)

    # Robot
    robot = RobotController(ROBOT_IP)
    if args.no_robot or args.dry_run:
        log.info("Robot disabled (%s)",
                 "--dry-run" if args.dry_run else "--no-robot")
    else:
        robot.connect()

    # Pipeline plumbing: the orchestrator pulls EE pose and depth via
    # callbacks; we keep the latest values in shared mutable state.
    # latest_frame is also kept here so the J_yz calibration thread (which
    # needs raw camera frames before/after each probe move) can read frames
    # without coupling to the capture loop's internal vars.
    shared = dict(latest_pose=np.eye(4), latest_depth=None,
                  latest_frame=None,
                  intrinsics=DEFAULT_INTRINSICS)

    def ee_pose_provider() -> np.ndarray:
        with shared_lock:
            return shared["latest_pose"].copy()

    def depth_provider(_frame: np.ndarray) -> np.ndarray | None:
        with shared_lock:
            d = shared["latest_depth"]
            return None if d is None else d.copy()

    def get_frame_for_calibration() -> np.ndarray | None:
        with shared_lock:
            f = shared["latest_frame"]
            return None if f is None else f.copy()

    sam3_runner = make_default_sam3_runner(args.ref_image, args.prompt)
    pipeline = LastMilePipeline(
        sam3_runner=sam3_runner,
        ee_pose_provider=ee_pose_provider,
        depth_provider=depth_provider,
        hand_eye_path=args.hand_eye,
        intrinsics=shared["intrinsics"],
    )

    shared_lock = threading.Lock()
    stop_ev = threading.Event()

    # ZED capture path (fall back to OpenCV stereo if PyZED missing)
    use_pyzed = not args.no_pyzed
    try:
        import pyzed.sl as sl  # type: ignore
        pyzed_ok = True
    except Exception:
        pyzed_ok = False
        if use_pyzed:
            log.warning("PyZED not importable; falling back to OpenCV camera "
                        "(no metric depth -> LOCK will never fire).")
        use_pyzed = False

    frame_idx = [0]
    debug_every = max(1, int(getattr(args, "debug_every", 30) or 30))

    # Live OpenCV preview window. Mirrors what servo_pipeline_sam3's
    # CameraStreamer does so the user sees the camera + annotations
    # (fused centroid, state label, mask overlay, watchdog alarm).
    show_window = not getattr(args, "no_window", False)
    win_name = "lastmile  |  [v] servo  [r] reset  [q] quit"
    if show_window:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1280, 720)
        except Exception as e:
            log.warning("Could not create preview window (%s); --no-window?", e)
            show_window = False

    def _handle_key(key: int) -> bool:
        """Returns True if the loop should keep running."""
        if key in (ord("q"), 27):  # q or ESC
            log.info("Quit requested via window key.")
            stop_ev.set()
            return False
        if key == ord("v") and robot is not None:
            robot.enabled = not robot.enabled
            log.info("Servo: %s", "ON" if robot.enabled else "OFF")
        if key == ord("r"):
            log.info("Manual tracker reset.")
            try:
                pipeline.fsm.reset()
                pipeline.signal_B = SignalB()
                pipeline.signal_C = SignalC()
                pipeline.signal_D = SignalD()
            except Exception as e:
                log.error("Reset failed: %s", e)
        return True

    def perception_loop_for_frame(frame_bgr: np.ndarray) -> dict:
        # Update EE pose every frame from xArm
        if not (args.dry_run or args.no_robot) and hasattr(robot, "_get_pos"):
            try:
                pos = robot._get_pos()
                if pos is not None:
                    with shared_lock:
                        shared["latest_pose"] = _xarm_pose_to_T(pos)
            except Exception as e:
                log.debug("Pose read failed: %s", e)

        res = pipeline.step(frame_bgr)

        # Drive robot when controller has a centroid and is enabled
        if (not args.dry_run and not args.no_robot
                and robot is not None
                and res.get("best_centroid") is not None):
            try:
                robot.servo_step(res["best_centroid"], frame_bgr.shape)
            except Exception as e:
                log.error("servo_step failed: %s", e)

        # Build overlay every frame (used for both live window and debug dump)
        overlay = None
        try:
            overlay = _make_overlay(frame_bgr, res)
        except Exception as e:
            log.debug("overlay build failed: %s", e)

        # Live preview window
        if show_window and overlay is not None:
            try:
                cv2.imshow(win_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    _handle_key(key)
            except Exception as e:
                log.debug("imshow failed: %s", e)

        # Periodic per-frame artifact dump
        i = frame_idx[0]
        frame_idx[0] += 1
        if debug_dir is not None and i % debug_every == 0:
            try:
                if overlay is not None:
                    cv2.imwrite(os.path.join(
                        debug_dir, "overlay_preview.png"), overlay)
            except Exception:
                pass
            _dump_debug_frame(debug_dir, i, frame_bgr, res, overlay, "live")

        # Always log a one-line summary so the user sees motion
        st = res.get("state", "?")
        c = res.get("best_centroid")
        z = res.get("z_mm")
        log.info("frame=%d state=%-9s c=%s z=%s",
                 i, st,
                 f"({c[0]:.0f},{c[1]:.0f})" if c else "-",
                 f"{z:.1f}mm" if z is not None else "-")
        return res

    # Spawn the J_yz Jacobian calibration thread BEFORE entering the
    # capture loop. The robot starts disabled (cal_status="waiting for
    # calibration..."); calibrate() flips robot.enabled=True after the
    # Y/Z probe sweep, at which point servo_step actually moves the arm.
    # Without this, the lastmile pipeline runs perception correctly but
    # the robot never moves and every servo_step logs the dreaded
    # "Servo: waiting [waiting for calibration...]" line forever.
    if not (args.dry_run or args.no_robot):
        def _calibration_runner():
            # Wait briefly for the first frame so calibrate() doesn't hit
            # its 30-second timeout on a slow camera open.
            for _ in range(30):
                if stop_ev.is_set():
                    return
                if get_frame_for_calibration() is not None:
                    break
                time.sleep(0.5)
            else:
                log.warning("Calibration: no frames within 15 s; skipping.")
                return
            log.info("Calibration thread: starting Y/Z Jacobian calibration.")
            try:
                robot.calibrate(get_frame_for_calibration)
            except Exception as e:
                log.error("Calibration failed: %s", e, exc_info=True)

        threading.Thread(target=_calibration_runner,
                          daemon=True,
                          name="lastmile-cal").start()

    if use_pyzed and pyzed_ok:
        _run_zed_loop(sl, args, stop_ev, shared, shared_lock,
                       perception_loop_for_frame)
    else:
        _run_opencv_loop(args, stop_ev, shared, shared_lock,
                          perception_loop_for_frame)

    if not (args.dry_run or args.no_robot):
        try:
            robot.stop()
        except Exception:
            pass

    if show_window:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _run_zed_loop(sl, args, stop_ev: threading.Event,
                   shared: dict, shared_lock: threading.Lock,
                   per_frame: Callable[[np.ndarray], dict]) -> None:
    cam = sl.Camera()
    init = sl.InitParameters()
    res_map = {"HD2K": sl.RESOLUTION.HD2K, "HD1080": sl.RESOLUTION.HD1080,
               "HD720": sl.RESOLUTION.HD720, "VGA": sl.RESOLUTION.VGA}
    init.camera_resolution = res_map.get(
        os.environ.get("ZED_RESOLUTION", "HD720"), sl.RESOLUTION.HD720)
    init.camera_fps = 30
    # ENABLE stereo depth — lastmile cannot LOCK without it.
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        log.error("ZED open failed (%s); falling back to OpenCV.", err)
        return _run_opencv_loop(args, stop_ev, shared, shared_lock, per_frame)

    # Pull intrinsics from the actual camera so projection math is right.
    cam_info = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    with shared_lock:
        shared["intrinsics"] = dict(
            fx=float(cam_info.fx), fy=float(cam_info.fy),
            cx=float(cam_info.cx), cy=float(cam_info.cy),
            width=int(cam_info.image_size.width),
            height=int(cam_info.image_size.height))
    log.info("ZED intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  size=%dx%d",
             cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy,
             cam_info.image_size.width, cam_info.image_size.height)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()

    log.info("ZED capture loop running (depth=PERFORMANCE, mm). "
             "Ctrl-C to stop.")
    try:
        while not stop_ev.is_set():
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.005)
                continue
            cam.retrieve_image(image_mat, sl.VIEW.LEFT)
            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            frame_bgra = image_mat.get_data()
            frame = frame_bgra[:, :, :3].copy()
            depth = depth_mat.get_data().astype(np.float32, copy=False)
            with shared_lock:
                shared["latest_depth"] = depth.copy()
                # Publish for the calibration thread (which reads frames
                # via get_frame_for_calibration() to compute optical flow
                # before/after each Y/Z probe move).
                shared["latest_frame"] = frame.copy()
            try:
                per_frame(frame)
            except Exception as e:
                log.error("Pipeline step error: %s", e, exc_info=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cam.close()


def _run_opencv_loop(args, stop_ev: threading.Event,
                      shared: dict, shared_lock: threading.Lock,
                      per_frame: Callable[[np.ndarray], dict]) -> None:
    cap = cv2.VideoCapture(args.cam_index, cv2.CAP_ANY)
    if not cap.isOpened():
        log.error("OpenCV: failed to open camera %d", args.cam_index)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    log.info("OpenCV capture loop running (no metric depth). Ctrl-C to stop.")
    try:
        while not stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            left = frame[:, : w // 2].copy()
            with shared_lock:
                shared["latest_frame"] = left.copy()
            try:
                per_frame(left)
            except Exception as e:
                log.error("Pipeline step error: %s", e, exc_info=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    args.debug_dir = "artifacts" if args.debug else None
    args.debug_every = (1 if (args.input_image or args.input_dir
                              or args.input_video) else 30)

    logger = _setup_logger(log_dir=args.log_dir)
    log_path = getattr(logger.handlers[1], "baseFilename", None) \
        if len(logger.handlers) > 1 else None

    debug_dir = _init_debug_dir(args.debug, args, args.ref_image)

    log.info("=" * 70)
    log.info("servo_lastmile starting")
    log.info("  prompt=%s  ref=%s  hand-eye=%s",
             args.prompt, args.ref_image, args.hand_eye)
    log.info("  offline=%s  debug=%s  dry-run=%s  no-robot=%s",
             bool(args.input_image or args.input_dir or args.input_video),
             args.debug, args.dry_run, args.no_robot)
    log.info("=" * 70)

    offline_inputs = [bool(args.input_image), bool(args.input_dir),
                      bool(args.input_video)]
    if sum(offline_inputs) > 1:
        log.error("Use only one of --input-image / --input-dir / --input-video")
        return 1

    try:
        if any(offline_inputs):
            if args.input_image and not os.path.exists(args.input_image):
                log.error("Input image not found: %s", args.input_image)
                return 1
            if args.input_dir and not os.path.isdir(args.input_dir):
                log.error("Input directory not found: %s", args.input_dir)
                return 1
            if args.input_video and not os.path.exists(args.input_video):
                log.error("Input video not found: %s", args.input_video)
                return 1
            run_offline(args, debug_dir=debug_dir)
        else:
            run_live(args, debug_dir=debug_dir)
        return 0
    finally:
        _finalize_debug_dir(debug_dir, log_path)
        log.info("Done.")


if __name__ == "__main__":
    sys.exit(main())
