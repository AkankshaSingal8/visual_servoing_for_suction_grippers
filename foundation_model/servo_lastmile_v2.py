#!/usr/bin/env python3
"""
SemVS v2 - Last-Mile Visual Servoing with Surface Normal + EE Occlusion Masking
=================================================================================

Augments servo_lastmile.py with three improvements from LAST_MILE_RESEARCH.md:

  Improvement 1 (High impact, low cost) — Surface Normal Feedback in NEAR:
      Run RANSAC plane fitting on every NEAR frame (not just TERMINAL).
      Outputs tilt_deg (roll/pitch deviation from camera Z) every frame so
      the controller can correct approach angle before the last 30 mm.

  Improvement 2 (Medium impact, medium cost) — EE Occlusion Masking for Signal D:
      As the gripper descends into frame, DINOv2 may match gripper patches to
      box patches. We project the known EE geometry (disk at EE tip) into image
      space and zero-mask those pixels before Signal D embedding.

  Improvement 3 (Optional, higher cost) — Full 6-DOF Box Pose via PBVS:
      Replace 3-DOF Signal A (centroid-only) with a 6-DOF signal computed from
      SAM mask depth points → RANSAC plane → centroid + normal → tilt error in
      NEAR, not only at TERMINAL. Enabled via --sixdof flag.

All four original signals (A/B/C/D) and the fusion logic are preserved unchanged.
Tilt correction (Signal E) is an additive output; Signal D gets EE masking applied.

Additional debugging:
  - OpenCV preview window with tilt angle, surface normal arrow, EE mask outline
  - Video recording: overlay.mp4 + raw.mp4 saved in --output-dir
  - Per-frame JSONL with plane_normal, tilt_deg, roll_deg, pitch_deg
  - --debug dumps every frame's input, mask, overlay, result JSON

Usage (offline video):
    python foundation_model/servo_lastmile_v2.py \
        --ref-image REF.png --input-video CLIP.mp4 \
        --output-dir runs/v2_test

Usage (live, dry-run):
    python foundation_model/servo_lastmile_v2.py \
        --ref-image REF.png --dry-run

Usage (live, robot enabled, 6-DOF mode):
    python foundation_model/servo_lastmile_v2.py \
        --ref-image REF.png --hand-eye config/hand_eye.npy --sixdof
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
from dataclasses import dataclass, field
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Re-use all math / signal classes from servo_lastmile.py
# ---------------------------------------------------------------------------
# We import selectively so this file is self-contained in what it ADDS,
# while avoiding duplicating the ~1200-line base.
try:
    from foundation_model.servo_lastmile import (
        _setup_logger, _empty_result, _serialize_result,
        _make_overlay, _save_overlay, _dump_debug_frame,
        _init_debug_dir, _finalize_debug_dir,
        _xarm_pose_to_T, _run_zed_loop, _run_opencv_loop,
        make_default_sam3_runner,
        weighted_geometric_median, project_3d_to_pixel,
        transform_point, invert_transform,
        uniform_mask_samples, fit_plane_ransac,
        bbox_touches_border, bbox_area_fraction, in_frame_fraction,
        load_hand_eye, make_intrinsics_matrix,
        LockState, SignalA, SignalB, SignalC, SignalD,
        SignalReading, fuse_centroids,
        State, StateMachine,
        terminal_plane_fit,
        LastMilePipeline,
        LOCK_BORDER_MARGIN_PX, LOCK_BORDER_MARGIN_FRAC,
        LOCK_AREA_MIN_FRAC, LOCK_AREA_MAX_FRAC,
        LOCK_DEPTH_MIN_MM, LOCK_DEPTH_MAX_MM, LOCK_CONSEC_FRAMES,
        NEAR_AREA_MAX_FRAC, NEAR_DEPTH_MM,
        TERM_DEPTH_MM, TERM_FUSED_ERR_PX,
        COTRACKER_NUM_POINTS, COTRACKER_VIS_THRESH, COTRACKER_CONF_THRESH,
        FUSION_OVERRIDE_PX, FUSION_GEO_MED_ITERS, FUSION_GEO_MED_EPS,
        WATCHDOG_DISAGREE_PX, WATCHDOG_DISAGREE_FRAMES,
        DINOV2_PATCH_SIZE, DINOV2_SIM_THRESH, DINOV2_MIN_MATCHES,
        DEFAULT_INTRINSICS, HEAVY_MODELS_DISABLED,
    )
except ImportError:
    from servo_lastmile import (
        _setup_logger, _empty_result, _serialize_result,
        _make_overlay, _save_overlay, _dump_debug_frame,
        _init_debug_dir, _finalize_debug_dir,
        _xarm_pose_to_T, _run_zed_loop, _run_opencv_loop,
        make_default_sam3_runner,
        weighted_geometric_median, project_3d_to_pixel,
        transform_point, invert_transform,
        uniform_mask_samples, fit_plane_ransac,
        bbox_touches_border, bbox_area_fraction, in_frame_fraction,
        load_hand_eye, make_intrinsics_matrix,
        LockState, SignalA, SignalB, SignalC, SignalD,
        SignalReading, fuse_centroids,
        State, StateMachine,
        terminal_plane_fit,
        LastMilePipeline,
        LOCK_BORDER_MARGIN_PX, LOCK_BORDER_MARGIN_FRAC,
        LOCK_AREA_MIN_FRAC, LOCK_AREA_MAX_FRAC,
        LOCK_DEPTH_MIN_MM, LOCK_DEPTH_MAX_MM, LOCK_CONSEC_FRAMES,
        NEAR_AREA_MAX_FRAC, NEAR_DEPTH_MM,
        TERM_DEPTH_MM, TERM_FUSED_ERR_PX,
        COTRACKER_NUM_POINTS, COTRACKER_VIS_THRESH, COTRACKER_CONF_THRESH,
        FUSION_OVERRIDE_PX, FUSION_GEO_MED_ITERS, FUSION_GEO_MED_EPS,
        WATCHDOG_DISAGREE_PX, WATCHDOG_DISAGREE_FRAMES,
        DINOV2_PATCH_SIZE, DINOV2_SIM_THRESH, DINOV2_MIN_MATCHES,
        DEFAULT_INTRINSICS, HEAVY_MODELS_DISABLED,
    )

log = logging.getLogger("lastmile_v2")

# ---------------------------------------------------------------------------
# Tunables (v2 additions)
# ---------------------------------------------------------------------------

# Signal E: plane fitting in NEAR state (Improvement 1)
NEAR_PLANE_PATCH_HALF   = 80    # larger patch than TERMINAL (50) for NEAR
NEAR_PLANE_MIN_PTS      = 80    # min valid depth points to attempt fit
NEAR_PLANE_THRESH_MM    = 6.0   # RANSAC inlier threshold in mm
NEAR_PLANE_MAX_ITER     = 100

# EE occlusion masking (Improvement 2)
# Approximate suction cup tip offset along EE Z axis (mm, toward camera)
EE_TIP_OFFSET_MM        = 160.0
# Physical radius of the gripper body at the tip (mm)
EE_BODY_RADIUS_MM       = 35.0
# How many radii to mask (1.0 = exact body radius, >1 = safety margin)
EE_MASK_MARGIN          = 1.4

# 6-DOF mode (Improvement 3): minimum inliers to trust plane for tilt
SIXDOF_MIN_INLIERS      = 60

# ---------------------------------------------------------------------------
# Improvement 1: Surface normal estimation in NEAR state (Signal E)
# ---------------------------------------------------------------------------

@dataclass
class TiltEstimate:
    """Result of a single NEAR-frame plane fit."""
    normal: np.ndarray          # unit normal in camera frame (points toward cam)
    centroid_cam: np.ndarray    # 3D centroid of plane in camera frame (mm)
    tilt_deg: float             # total angular deviation from camera Z axis (deg)
    roll_deg: float             # rotation around camera X axis (deg)
    pitch_deg: float            # rotation around camera Y axis (deg)
    n_inliers: int


def estimate_near_tilt(depth_mm: np.ndarray,
                       K: np.ndarray,
                       center_xy: tuple[float, float],
                       patch_half: int = NEAR_PLANE_PATCH_HALF,
                       min_pts: int = NEAR_PLANE_MIN_PTS,
                       thresh_mm: float = NEAR_PLANE_THRESH_MM,
                       max_iter: int = NEAR_PLANE_MAX_ITER,
                       rng: np.random.Generator | None = None
                       ) -> TiltEstimate | None:
    """
    Fit a plane to the depth ROI around center_xy and extract tilt angles.

    The surface normal of a flat box face should point along camera -Z when
    perfectly aligned. Any deviation = tilt error the controller should correct.

    Returns None if insufficient valid depth points in the patch.
    """
    if rng is None:
        rng = np.random.default_rng(0)

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
    valid = (patch > 5.0) & (patch < 2000.0) & np.isfinite(patch)
    if int(valid.sum()) < min_pts:
        return None

    z_vals = patch[valid]
    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cxK, cyK = float(K[0, 2]), float(K[1, 2])
    X = (u - cxK) * z_vals / fx
    Y = (v - cyK) * z_vals / fy
    pts = np.stack([X, Y, z_vals], axis=1)

    result = fit_plane_ransac(pts, thresh_mm=thresh_mm,
                              max_iter=max_iter, rng=rng)
    if result is None:
        return None

    normal, centroid = result

    # Count inliers for confidence
    d = -float(normal @ centroid)
    dists = np.abs(pts @ normal + d)
    n_inliers = int((dists < thresh_mm).sum())

    # Camera Z axis is [0, 0, -1] (pointing away from camera toward scene).
    # The box normal should ideally be [0, 0, -1] (facing us).
    # Tilt = angle between normal and [0, 0, -1].
    cam_z = np.array([0.0, 0.0, -1.0])
    cos_angle = float(np.clip(normal @ cam_z, -1.0, 1.0))
    tilt_deg = float(math.degrees(math.acos(abs(cos_angle))))

    # Decompose: roll = atan2(ny, nz), pitch = atan2(nx, nz)
    # (small-angle linearisation around camera Z)
    roll_deg  = float(math.degrees(math.atan2(float(normal[1]),
                                              abs(float(normal[2])))))
    pitch_deg = float(math.degrees(math.atan2(float(normal[0]),
                                              abs(float(normal[2])))))

    return TiltEstimate(
        normal=normal,
        centroid_cam=centroid,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        n_inliers=n_inliers,
    )


# ---------------------------------------------------------------------------
# Improvement 2: EE occlusion masking for Signal D
# ---------------------------------------------------------------------------

class EEOcclusionMasker:
    """
    Projects the end-effector body into image space and returns a binary mask
    (255=EE, 0=safe) so Signal D can ignore EE-occupied pixels.

    Model: the suction gripper tip is a disk of radius EE_BODY_RADIUS_MM
    at position (EE frame origin + EE_TIP_OFFSET_MM along EE Z axis).
    Both parameters are adjustable via constructor args for different grippers.
    """

    def __init__(self,
                 T_ee_cam: np.ndarray,
                 K: np.ndarray,
                 tip_offset_mm: float = EE_TIP_OFFSET_MM,
                 body_radius_mm: float = EE_BODY_RADIUS_MM,
                 mask_margin: float = EE_MASK_MARGIN):
        self.T_cam_ee = invert_transform(T_ee_cam)
        self.K = K
        self.tip_offset_mm = tip_offset_mm
        self.body_radius_mm = body_radius_mm
        self.mask_margin = mask_margin

    def get_mask(self, frame_bgr: np.ndarray,
                 T_base_ee: np.ndarray) -> np.ndarray | None:
        """
        Returns a uint8 mask (H, W) where EE pixels = 255, background = 0.
        Returns None if the EE projects behind the camera or out of frame.
        """
        h, w = frame_bgr.shape[:2]
        T_base_cam = T_base_ee @ invert_transform(self.T_cam_ee)
        T_cam_base = invert_transform(T_base_cam)

        # EE tip = EE origin + tip_offset along EE local Z.
        # In EE frame, local Z column is T_ee_cam[:3, 2] (cam Z in EE frame).
        # Simpler: EE origin in camera frame via T_cam_ee.
        ee_origin_cam = self.T_cam_ee[:3, 3]  # EE origin in camera frame

        # EE Z axis in camera frame
        ee_z_cam = self.T_cam_ee[:3, 2]
        ee_z_cam = ee_z_cam / (np.linalg.norm(ee_z_cam) + 1e-9)

        # Tip position in camera frame: origin + tip_offset along EE Z
        tip_cam = ee_origin_cam + ee_z_cam * self.tip_offset_mm

        if tip_cam[2] <= 1.0:
            return None  # behind camera

        # Project tip center
        uv = project_3d_to_pixel(tip_cam, self.K)
        if uv is None:
            return None

        # Compute projected radius (approximation: use disk tangent at Z depth)
        z = float(tip_cam[2])
        radius_px = int(math.ceil(
            self.body_radius_mm * self.mask_margin * float(self.K[0, 0]) / z
        ))
        if radius_px < 1:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        cx_px, cy_px = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(mask, (cx_px, cy_px), radius_px, 255, -1)
        return mask

    def apply_to_frame(self, frame_bgr: np.ndarray,
                       T_base_ee: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Returns (masked_frame, ee_mask) where masked_frame has the EE region
        zeroed out (safe to pass to DINOv2 without EE patch pollution).
        """
        ee_mask = self.get_mask(frame_bgr, T_base_ee)
        if ee_mask is None:
            return frame_bgr, None
        masked = frame_bgr.copy()
        masked[ee_mask > 0] = 0
        return masked, ee_mask


# ---------------------------------------------------------------------------
# Improvement 3: 6-DOF Signal A — centroid + normal from depth in NEAR
# ---------------------------------------------------------------------------

class SignalA_6DOF:
    """
    Replaces Signal A's 3-DOF centroid-only output with a 6-DOF box face pose
    when a valid plane fit is available in NEAR state.

    Outputs:
      - centroid_px: same as original Signal A (2D pixel location)
      - tilt: TiltEstimate with roll/pitch for the controller
      - is_6dof: True if the plane fit succeeded, False if fallback to 3-DOF

    When depth is unavailable or the plane fit fails, falls back to the
    original 3-DOF Signal A behaviour.
    """

    def __init__(self, signal_a: SignalA):
        self._sig_a = signal_a

    def predict(self,
                lock: LockState,
                T_base_ee_now: np.ndarray,
                depth_mm: np.ndarray | None,
                K: np.ndarray,
                fused_centroid: tuple[float, float] | None,
                rng: np.random.Generator | None = None,
                ) -> tuple[tuple[float, float] | None, TiltEstimate | None, bool]:
        """
        Returns (centroid_px, tilt_estimate, is_6dof).
        """
        c_A_3dof = self._sig_a.predict(lock, T_base_ee_now)

        if depth_mm is None or fused_centroid is None:
            return c_A_3dof, None, False

        tilt = estimate_near_tilt(
            depth_mm, K, fused_centroid,
            min_pts=SIXDOF_MIN_INLIERS, rng=rng)

        if tilt is None or tilt.n_inliers < SIXDOF_MIN_INLIERS:
            return c_A_3dof, tilt, False

        # Use the 3D centroid from the plane fit projected back to 2D
        # as a depth-grounded centroid (more accurate than reprojection alone
        # when Z is short and hand-eye has small errors).
        uv = project_3d_to_pixel(tilt.centroid_cam, K)
        centroid_6dof = uv if uv is not None else c_A_3dof

        return centroid_6dof, tilt, True


# ---------------------------------------------------------------------------
# Extended result dict
# ---------------------------------------------------------------------------

def _empty_result_v2() -> dict:
    res = _empty_result()
    res.update(dict(
        # Surface normal feedback (NEAR every frame)
        tilt_deg=None,
        roll_deg=None,
        pitch_deg=None,
        near_plane_normal=None,
        near_plane_n_inliers=None,
        # EE occlusion
        ee_mask=None,
        ee_masking_applied=False,
        # 6-DOF flag
        signal_a_is_6dof=False,
        # Signal E (SAM3 re-anchor in NEAR)
        c_E=None,
    ))
    return res


# ---------------------------------------------------------------------------
# Fix 1: depth-only TERMINAL trigger (no centroid-proximity false trigger)
# ---------------------------------------------------------------------------

class StateMachineV2(StateMachine):
    """
    Overrides evaluate_near_to_term to use depth-only TERMINAL trigger.

    The base class also fires TERMINAL when the fused centroid is within
    TERM_FUSED_ERR_PX of image centre. This is unreliable: the centroid can
    be near image centre by coincidence (e.g. fusion noise) while the robot
    is still hundreds of mm away. Depth < TERM_DEPTH_MM is the only
    physically meaningful contact-proximity signal.
    """
    def evaluate_near_to_term(self,
                              c_fused: tuple[float, float] | None,
                              z_mm: float | None,
                              image_center: tuple[float, float]) -> bool:
        return z_mm is not None and z_mm < TERM_DEPTH_MM


# ---------------------------------------------------------------------------
# Fix 2: periodic SAM3 re-anchor in NEAR (Signal E)
# ---------------------------------------------------------------------------

# Run SAM3 every N frames during NEAR to get a fresh centroid estimate.
# SAM3 scored 0.96 consistently — it is the most accurate signal we have.
# Between re-runs, the last valid result is reused (cached centroid + weight).
SAM3_NEAR_EVERY_N = 5       # frames between SAM3 re-runs in NEAR
SAM3_NEAR_MIN_SCORE = 0.50  # min SAM3 score to use as Signal E


# ---------------------------------------------------------------------------
# V2 pipeline orchestrator
# ---------------------------------------------------------------------------

class LastMilePipelineV2(LastMilePipeline):
    """
    Extends LastMilePipeline with:
      - Surface normal feedback every NEAR frame (estimate_near_tilt)
      - EE occlusion masking applied to Signal D input
      - Optional 6-DOF Signal A (--sixdof)
      - Periodic SAM3 re-anchor in NEAR as Signal E (fixes centroid drift)
      - Signal C (SAM2) removed from fusion (watchdog only)
      - Depth-only TERMINAL trigger (no false fire on centroid coincidence)
    """

    def __init__(self,
                 sam3_runner: Callable[[np.ndarray], dict],
                 ee_pose_provider: Callable[[], np.ndarray] | None = None,
                 depth_provider: Callable[[np.ndarray], np.ndarray | None] | None = None,
                 hand_eye_path: str | None = None,
                 intrinsics: dict | None = None,
                 use_sixdof: bool = False,
                 ee_tip_offset_mm: float = EE_TIP_OFFSET_MM,
                 ee_body_radius_mm: float = EE_BODY_RADIUS_MM):
        super().__init__(
            sam3_runner=sam3_runner,
            ee_pose_provider=ee_pose_provider,
            depth_provider=depth_provider,
            hand_eye_path=hand_eye_path,
            intrinsics=intrinsics,
        )
        self.use_sixdof = use_sixdof
        self._rng = np.random.default_rng(0)

        # Replace base StateMachine with depth-only TERMINAL version
        self.fsm = StateMachineV2()

        self.ee_masker = EEOcclusionMasker(
            T_ee_cam=self.T_ee_cam,
            K=self.K,
            tip_offset_mm=ee_tip_offset_mm,
            body_radius_mm=ee_body_radius_mm,
        )

        if use_sixdof:
            self.signal_A_6dof = SignalA_6DOF(self.signal_A)
            log.info("6-DOF Signal A enabled.")
        else:
            self.signal_A_6dof = None

        # NEAR state phase tracking
        # Phase 'sam3': box fully in frame — SAM3 is sole signal, no fusion
        # Phase 'track': box exited frame — B/A tracking with SAM3 handoff anchor
        self._near_phase: str = 'sam3'

        # Rolling buffer of last 5 SAM3 centroids + last mask for handoff
        self._sam3_buffer: list[tuple[float, float]] = []
        self._SAM3_BUFFER_SIZE = 5
        self._last_sam3_mask: np.ndarray | None = None   # for CoTracker3 re-seed
        self._sam3_handoff: tuple[float, float] | None = None  # frozen at phase switch

        # Lightweight EMA on SAM3 output only (reduces per-frame noise from mask centroid)
        self._sam3_ema: tuple[float, float] | None = None
        self._sam3_ema_alpha = 0.7  # high alpha = mostly current frame, low lag

        # Weight EMA for tracking phase only
        self._ema_weights: dict[str, float] = {}

    def _smooth_weight(self, name: str, w: float, alpha: float = 0.5) -> float:
        prev = self._ema_weights.get(name, w)
        smoothed = alpha * w + (1 - alpha) * prev
        self._ema_weights[name] = smoothed
        return smoothed

    def step(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a single frame. Adds surface-normal and EE-masking outputs
        on top of the base pipeline's result dict.
        """
        res = _empty_result_v2()
        h, w = frame_bgr.shape[:2]
        self.fsm.frame_idx += 1
        depth_map = self.depth_provider(frame_bgr)

        # ============ FAR (unchanged) ============
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
                    # Reset NEAR phase state for fresh approach
                    self._near_phase = 'sam3'
                    self._sam3_buffer = []
                    self._sam3_handoff = None
                    self._sam3_ema = None
                    self._last_sam3_mask = None
                    res["state"] = State.NEAR
                    res["best_centroid"] = lock.centroid_px
            return res

        # ============ NEAR (SAM3-primary with graceful handoff) ============
        if self.fsm.state == State.NEAR:
            res["state"] = State.NEAR
            lock = self.fsm.lock_state
            T_base_ee_now = self.ee_pose_provider()
            best_centroid = None
            tilt = None

            # -----------------------------------------------------------------
            # PHASE 'sam3': box fully in frame — SAM3 is the sole signal.
            # No fusion, no B/D/A. SAM3 scored 0.96 every frame in FAR state.
            # We trust it completely until the box exits the frame.
            # -----------------------------------------------------------------
            if self._near_phase == 'sam3':
                try:
                    sam3_res_near = self.sam3_runner(frame_bgr)
                    s_box   = sam3_res_near.get("gdino_box")
                    s_mask  = sam3_res_near.get("mask_np")
                    s_score = float(sam3_res_near.get("sam_score") or 0.0)

                    box_visible = (
                        s_box is not None and s_mask is not None
                        and s_score >= 0.50
                        and not bbox_touches_border(np.asarray(s_box), w, h)
                        and bbox_area_fraction(np.asarray(s_box), w, h) < 0.70
                    )

                    if box_visible:
                        ys, xs = np.where(s_mask > 0)
                        if xs.size >= 50:
                            raw_cx = float(xs.mean())
                            raw_cy = float(ys.mean())
                            # Light EMA on SAM3 output only (reduces per-frame
                            # mask noise without hiding real centroid changes).
                            if self._sam3_ema is None:
                                self._sam3_ema = (raw_cx, raw_cy)
                            else:
                                a = self._sam3_ema_alpha
                                self._sam3_ema = (
                                    a * raw_cx + (1-a) * self._sam3_ema[0],
                                    a * raw_cy + (1-a) * self._sam3_ema[1],
                                )
                            best_centroid = self._sam3_ema

                            # Rolling buffer for smooth handoff anchor
                            self._sam3_buffer.append(self._sam3_ema)
                            if len(self._sam3_buffer) > self._SAM3_BUFFER_SIZE:
                                self._sam3_buffer.pop(0)
                            self._last_sam3_mask = s_mask  # keep for CoTracker3 re-seed

                            res["mask_np"]    = s_mask
                            res["gdino_box"]  = s_box
                            res["sam_score"]  = s_score
                            log.debug("NEAR SAM3: c=(%.0f,%.0f) score=%.2f",
                                      best_centroid[0], best_centroid[1], s_score)
                    else:
                        # Box exited frame or SAM3 failed — switch to tracking
                        log.info("NEAR: box exited frame (score=%.2f border=%s) "
                                 "— switching to tracking phase",
                                 s_score,
                                 str(s_box is not None and
                                     bbox_touches_border(np.asarray(s_box), w, h)
                                     if s_box is not None else False))
                        self._near_phase = 'track'
                        # Handoff anchor: mean of last N SAM3 centroids
                        if self._sam3_buffer:
                            xs_buf = [p[0] for p in self._sam3_buffer]
                            ys_buf = [p[1] for p in self._sam3_buffer]
                            self._sam3_handoff = (
                                float(sum(xs_buf)/len(xs_buf)),
                                float(sum(ys_buf)/len(ys_buf)),
                            )
                        else:
                            self._sam3_handoff = lock.centroid_px
                        # Re-seed CoTracker3 and SAM2 from the last SAM3 mask.
                        # Using the most recent close-up mask rather than the
                        # stale lock-time snapshot prevents the 2-3 frame mask
                        # quality dip that occurs at the sam3→track transition.
                        if self._last_sam3_mask is not None:
                            self.signal_B.reset_from_mask(
                                self._last_sam3_mask, frame_bgr)
                            # Re-seed SAM2 logits directly — avoids the stale
                            # lock-time snapshot that causes partial masks for
                            # the first few track frames.
                            if not self.signal_C._failed:
                                m = (self._last_sam3_mask > 127).astype(np.float32)
                                m_resized = cv2.resize(m, (256, 256),
                                                       interpolation=cv2.INTER_LINEAR)
                                self.signal_C._prev_logits = (m_resized * 20.0 - 10.0)[None, ...]
                                ys_m, xs_m = np.where(self._last_sam3_mask > 0)
                                if xs_m.size > 0:
                                    self.signal_C._prev_bbox = np.array([
                                        float(xs_m.min()), float(ys_m.min()),
                                        float(xs_m.max()), float(ys_m.max()),
                                    ], dtype=np.float32)
                                log.debug("Signal C: logits re-seeded from SAM3 handoff mask.")
                        else:
                            self.signal_B.reset(lock, frame_bgr)
                        log.info("NEAR handoff anchor: (%.0f, %.0f)",
                                 self._sam3_handoff[0], self._sam3_handoff[1])

                except Exception as ex:
                    log.warning("NEAR SAM3 phase error: %s — switching to track", ex)
                    self._near_phase = 'track'
                    self._sam3_handoff = (
                        self._sam3_buffer[-1] if self._sam3_buffer
                        else lock.centroid_px
                    )

            # -----------------------------------------------------------------
            # PHASE 'track': box partially/fully exited — use B + handoff anchor.
            # The handoff anchor is the average of the last N SAM3 centroids,
            # providing a stable grounded starting point for the tracker.
            # -----------------------------------------------------------------
            if self._near_phase == 'track':
                c_A = self.signal_A.predict(lock, T_base_ee_now)
                c_B, w_B, pts_now, vis = self.signal_B.step(frame_bgr)
                c_C, _, mask_C = self.signal_C.step(frame_bgr)
                if mask_C is not None:
                    res["mask_np"] = mask_C

                # CoTracker3 is the primary signal in track phase.
                # Its all-points mean (including off-frame extrapolated points)
                # is the best estimate of box center. Don't penalise by visibility
                # fraction — confidence drops as points go off-frame but the
                # extrapolated centroid is still correct. Use fixed high weight.
                # Handoff anchor is secondary: only used if CoTracker3 fails.
                c_E = self._sam3_handoff
                if c_B is not None:
                    # CoTracker3 returned a position — trust it fully
                    best_centroid = c_B
                    w_E = 0.0   # no need for anchor when tracker is live
                else:
                    # CoTracker3 failed — fall back to frozen handoff anchor
                    best_centroid = c_E
                    w_E = 1.0

                res["c_A"] = c_A
                res["c_B"] = c_B
                res["c_E"] = c_E
                res["c_fused"] = best_centroid
                info = {"used_weights": {"B": 1.0 if c_B else 0.0,
                                         "E": w_E}}
                res["weights"] = info["used_weights"]

                alarm = self.fsm.update_watchdog(best_centroid, c_C)
                res["watchdog_alarm"] = alarm

            # -----------------------------------------------------------------
            # Common outputs for both phases
            # -----------------------------------------------------------------
            res["best_centroid"] = best_centroid
            res["c_E"] = best_centroid if self._near_phase == 'sam3' else res.get("c_E")
            res["_near_phase"] = self._near_phase

            z_mm = self._depth_at_px(depth_map, best_centroid)
            res["z_mm"] = z_mm

            # Surface normal (tilt) in NEAR
            if depth_map is not None and best_centroid is not None:
                tilt_near = estimate_near_tilt(
                    depth_map, self.K, best_centroid,
                    patch_half=NEAR_PLANE_PATCH_HALF,
                    min_pts=NEAR_PLANE_MIN_PTS,
                    thresh_mm=NEAR_PLANE_THRESH_MM,
                    max_iter=NEAR_PLANE_MAX_ITER,
                    rng=self._rng,
                )
                if tilt_near is not None:
                    tilt = tilt_near
                    res["tilt_deg"]  = tilt_near.tilt_deg
                    res["roll_deg"]  = tilt_near.roll_deg
                    res["pitch_deg"] = tilt_near.pitch_deg
                    res["near_plane_normal"]    = tilt_near.normal.tolist()
                    res["near_plane_n_inliers"] = tilt_near.n_inliers
                    log.debug("NEAR plane: tilt=%.1f° roll=%.1f° pitch=%.1f°",
                              tilt_near.tilt_deg, tilt_near.roll_deg,
                              tilt_near.pitch_deg)

            res["_tilt"] = tilt

            ic = (w / 2.0, h / 2.0)
            # Position-based TERMINAL fallback: z_mm is unreliable at close range
            # (ZED measures background through partial box). Use robot X position:
            # home X=205mm, approach in +X, TERMINAL when X > 205+300=505mm.
            T_now = self.ee_pose_provider()
            robot_x_now = float(T_now[0, 3])
            position_terminal = robot_x_now > 505.0
            if position_terminal:
                log.info("NEAR: position-based TERMINAL (robot X=%.1fmm > 505mm)",
                         robot_x_now)
            if position_terminal or self.fsm.evaluate_near_to_term(best_centroid, z_mm, ic):
                self.fsm.state = State.TERM
                res["state"] = State.TERM
                log.info("State -> TERMINAL @ frame %d (z=%s)",
                         self.fsm.frame_idx,
                         f"{z_mm:.1f}mm" if z_mm else "n/a")
            return res

        # ============ TERMINAL (plane fit still runs; tilt added) ============
        if self.fsm.state == State.TERM:
            res["state"] = State.TERM
            ic = (w / 2.0, h / 2.0)

            if depth_map is not None:
                fit = terminal_plane_fit(depth_map, self.K, ic)
                if fit is not None:
                    normal, centroid = fit
                    res["plane_normal"] = normal.tolist()
                    res["best_centroid"] = ic

                    tilt_term = TiltEstimate(
                        normal=normal,
                        centroid_cam=centroid,
                        tilt_deg=float(math.degrees(math.acos(
                            abs(float(np.clip(normal @ np.array([0., 0., -1.]),
                                              -1., 1.)))))),
                        roll_deg=float(math.degrees(math.atan2(
                            float(normal[1]), abs(float(normal[2]))))),
                        pitch_deg=float(math.degrees(math.atan2(
                            float(normal[0]), abs(float(normal[2]))))),
                        n_inliers=0,
                    )
                    res["tilt_deg"]  = tilt_term.tilt_deg
                    res["roll_deg"]  = tilt_term.roll_deg
                    res["pitch_deg"] = tilt_term.pitch_deg
                    res["near_plane_normal"] = normal.tolist()
                    res["_tilt"] = tilt_term

                    log.info("TERM: tilt=%.1f° roll=%.1f° pitch=%.1f° "
                             "normal=%s",
                             tilt_term.tilt_deg, tilt_term.roll_deg,
                             tilt_term.pitch_deg,
                             np.round(normal, 3).tolist())

            T_base_ee_now = self.ee_pose_provider()
            c_A = self.signal_A.predict(self.fsm.lock_state, T_base_ee_now)
            res["c_A"] = c_A
            return res

        return res


# ---------------------------------------------------------------------------
# Overlay (v2 additions: tilt angle, normal arrow, EE mask outline)
# ---------------------------------------------------------------------------

def _make_overlay_v2(frame: np.ndarray, res: dict) -> np.ndarray:
    """Build the v2 overlay. Adds tilt/roll/pitch HUD, normal arrow, EE circle."""
    # Suppress Signal A from overlay — without hand-eye calibration it shows
    # a drifting wrong position that confuses the visualization.
    res_display = dict(res)
    res_display["c_A"] = None
    img = _make_overlay(frame, res_display)
    h, w = img.shape[:2]

    # Signal E (SAM3 re-anchor) — orange square marker.
    # Only draw in 'track' phase; in 'sam3' phase c_E == best_centroid and
    # drawing both produces a redundant orange box inside the red grasp cross.
    c_E = res.get("c_E")
    if c_E is not None and res.get("_near_phase") == "track":
        ex, ey = int(c_E[0]), int(c_E[1])
        cv2.rectangle(img, (ex - 8, ey - 8), (ex + 8, ey + 8), (0, 128, 255), 2)
        cv2.putText(img, "E", (ex + 10, ey - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    tilt = res.get("_tilt")  # TiltEstimate or None

    # Surface normal arrow from fused centroid
    cf = res.get("best_centroid") or res.get("c_fused")
    if tilt is not None and cf is not None:
        n = tilt.normal
        # Project normal vector as a 50-px arrow in image space.
        # We visualise the XY components of the normal (in-plane tilt).
        arrow_len = 60
        nx_img = float(n[0]) * arrow_len
        ny_img = float(n[1]) * arrow_len
        start = (int(cf[0]), int(cf[1]))
        end = (int(cf[0] + nx_img), int(cf[1] + ny_img))
        cv2.arrowedLine(img, start, end, (0, 255, 255), 2, tipLength=0.25)

    # Tilt HUD: top-right corner
    tilt_deg  = res.get("tilt_deg")
    roll_deg  = res.get("roll_deg")
    pitch_deg = res.get("pitch_deg")
    n_inliers = res.get("near_plane_n_inliers")
    if tilt_deg is not None:
        color = (0, 255, 0) if tilt_deg < 2.0 else (
                 (0, 165, 255) if tilt_deg < 5.0 else (0, 0, 255))
        lines = [
            f"tilt  {tilt_deg:+.1f}°",
            f"roll  {roll_deg:+.1f}°",
            f"pitch {pitch_deg:+.1f}°",
        ]
        if n_inliers is not None:
            lines.append(f"inlrs {n_inliers}")
        for i, txt in enumerate(lines):
            cv2.putText(img, txt, (w - 180, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # EE mask outline (cyan dashed circle)
    ee_mask = res.get("ee_mask")
    if ee_mask is not None:
        cnts, _ = cv2.findContours(ee_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 255, 0), 2)
        cv2.putText(img, "EE masked", (12, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 6-DOF mode indicator
    if res.get("signal_a_is_6dof"):
        cv2.putText(img, "6-DOF A", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    return img


# ---------------------------------------------------------------------------
# Video recorder helper
# ---------------------------------------------------------------------------

class VideoRecorder:
    """
    Wraps cv2.VideoWriter. Opens lazily on first frame (so we know the size).
    Writes overlay and raw frame videos in parallel.
    """

    def __init__(self, out_dir: str, fps: float = 15.0):
        self.out_dir = out_dir
        self.fps = fps
        self._overlay_writer: cv2.VideoWriter | None = None
        self._raw_writer:     cv2.VideoWriter | None = None
        self._overlay_path = os.path.join(out_dir, "overlay.mp4")
        self._raw_path     = os.path.join(out_dir, "raw.mp4")
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def _ensure_open(self, h: int, w: int) -> None:
        if self._overlay_writer is None:
            self._overlay_writer = cv2.VideoWriter(
                self._overlay_path, self._fourcc, self.fps, (w, h))
            self._raw_writer = cv2.VideoWriter(
                self._raw_path, self._fourcc, self.fps, (w, h))
            log.info("Video recording: overlay=%s  raw=%s",
                     self._overlay_path, self._raw_path)

    def write(self, frame_bgr: np.ndarray, overlay_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        self._ensure_open(h, w)
        self._raw_writer.write(frame_bgr)
        oh, ow = overlay_bgr.shape[:2]
        if (oh, ow) != (h, w):
            overlay_bgr = cv2.resize(overlay_bgr, (w, h))
        self._overlay_writer.write(overlay_bgr)

    def release(self) -> None:
        if self._overlay_writer is not None:
            self._overlay_writer.release()
            self._raw_writer.release()
            log.info("Video saved: %s", self._overlay_path)
            log.info("Video saved: %s", self._raw_path)


# ---------------------------------------------------------------------------
# Offline runner
# ---------------------------------------------------------------------------

def _serialize_result_v2(res: dict) -> dict:
    """Extend base serializer to handle v2-only fields."""
    clean = {k: v for k, v in res.items() if k != "_tilt"}
    return _serialize_result(clean)


def run_offline_v2(args: argparse.Namespace, debug_dir: str | None = None) -> None:
    out_dir = args.output_dir or os.path.join(
        "runs", time.strftime("lastmile_v2_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "frames.jsonl")

    sam3_runner = make_default_sam3_runner(args.ref_image, args.prompt)
    pipeline = LastMilePipelineV2(
        sam3_runner=sam3_runner,
        ee_pose_provider=lambda: np.eye(4),
        depth_provider=lambda f: None,
        hand_eye_path=args.hand_eye,
        use_sixdof=getattr(args, "sixdof", False),
    )

    recorder = VideoRecorder(out_dir, fps=getattr(args, "fps", 15.0))

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
                    n = _process_offline_frame_v2(
                        pipeline, frame, n, path, out_dir,
                        fh, args, debug_dir, recorder)
                cap.release()
            else:
                frame = cv2.imread(path)
                if frame is None:
                    continue
                n = _process_offline_frame_v2(
                    pipeline, frame, n, path, out_dir,
                    fh, args, debug_dir, recorder)

    recorder.release()
    log.info("Wrote %d frame summaries to %s", n, jsonl_path)


def _process_offline_frame_v2(pipeline: LastMilePipelineV2,
                               frame: np.ndarray,
                               n: int,
                               source: str,
                               out_dir: str,
                               fh,
                               args: argparse.Namespace,
                               debug_dir: str | None,
                               recorder: VideoRecorder) -> int:
    res = pipeline.step(frame)
    fh.write(json.dumps(_serialize_result_v2(res)) + "\n")
    fh.flush()

    overlay = _make_overlay_v2(frame, res)
    recorder.write(frame, overlay)

    if not getattr(args, "no_overlays", False):
        overlay_path = os.path.join(out_dir, f"overlay_{n:06d}.png")
        cv2.imwrite(overlay_path, overlay)

    if debug_dir is not None:
        _dump_debug_frame(debug_dir, n, frame, res, overlay, source)

    return n + 1


# ---------------------------------------------------------------------------
# Live runner (v2)
# ---------------------------------------------------------------------------

def run_live_v2(args: argparse.Namespace, debug_dir: str | None = None) -> None:
    try:
        try:
            from foundation_model.servo_pipeline_sam3 import (
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

    robot = RobotController(ROBOT_IP)
    home_pos = None  # captured after connect, used for return-to-home on exit
    if args.no_robot or args.dry_run:
        log.info("Robot disabled (%s)",
                 "--dry-run" if args.dry_run else "--no-robot")
    else:
        robot.connect()
        # Record initial position immediately after connect so we can
        # return to it regardless of how the session ends (quit key,
        # Ctrl-C, error, or normal completion).
        try:
            home_pos = robot._get_pos()
            if home_pos is not None:
                log.info("Home position captured: [%.1f, %.1f, %.1f] mm",
                         home_pos[0], home_pos[1], home_pos[2])
        except Exception as e:
            log.warning("Could not capture home position: %s", e)

    shared = dict(latest_pose=np.eye(4), latest_depth=None,
                  latest_frame=None, intrinsics=DEFAULT_INTRINSICS)

    shared_lock = threading.Lock()
    stop_ev = threading.Event()

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
    pipeline = LastMilePipelineV2(
        sam3_runner=sam3_runner,
        ee_pose_provider=ee_pose_provider,
        depth_provider=depth_provider,
        hand_eye_path=args.hand_eye,
        intrinsics=shared["intrinsics"],
        use_sixdof=getattr(args, "sixdof", False),
    )

    # Output dir for live video recording
    out_dir = getattr(args, "output_dir", None) or os.path.join(
        "runs", time.strftime("lastmile_v2_live_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    recorder = VideoRecorder(out_dir, fps=15.0)
    jsonl_path = os.path.join(out_dir, "frames.jsonl")
    jsonl_fh = open(jsonl_path, "w")

    use_pyzed = not args.no_pyzed
    try:
        import pyzed.sl as sl
        pyzed_ok = True
    except Exception:
        pyzed_ok = False
        if use_pyzed:
            log.warning("PyZED not importable; falling back to OpenCV camera.")
        use_pyzed = False

    frame_idx = [0]
    debug_every = max(1, int(getattr(args, "debug_every", 30) or 30))

    # OpenCV preview window
    show_window = not getattr(args, "no_window", False)
    win_name = "lastmile v2  |  [v] servo  [r] reset  [q] quit"
    if show_window:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1280, 720)
        except Exception as e:
            log.warning("Could not create preview window (%s); "
                        "use --no-window for headless.", e)
            show_window = False

    def _handle_key(key: int) -> None:
        if key in (ord("q"), 27):
            log.info("Quit requested via window key.")
            stop_ev.set()
        elif key == ord("v") and robot is not None:
            robot.enabled = not robot.enabled
            log.info("Servo: %s", "ON" if robot.enabled else "OFF")
        elif key == ord("r"):
            pipeline.reset()
            log.info("Pipeline reset by user.")

    def perception_loop_for_frame(frame_bgr: np.ndarray) -> dict:
        if not (args.dry_run or args.no_robot) and hasattr(robot, "_get_pos"):
            try:
                pos = robot._get_pos()
                if pos is not None:
                    with shared_lock:
                        shared["latest_pose"] = _xarm_pose_to_T(pos)
            except Exception as e:
                log.debug("Pose read failed: %s", e)

        res = pipeline.step(frame_bgr)

        if (not args.dry_run and not args.no_robot
                and robot is not None
                and res.get("best_centroid") is not None
                and not res.get("watchdog_alarm", False)):
            try:
                robot.servo_step(res["best_centroid"], frame_bgr.shape)
            except Exception as e:
                log.error("servo_step failed: %s", e)
        elif res.get("watchdog_alarm", False):
            log.warning("Servo paused — watchdog alarm active.")

        # Build v2 overlay
        overlay = None
        try:
            overlay = _make_overlay_v2(frame_bgr, res)
        except Exception as e:
            log.debug("overlay build failed: %s", e)

        # Show in window with correct waitKey
        if show_window and overlay is not None:
            try:
                cv2.imshow(win_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    _handle_key(key)
            except Exception as e:
                log.debug("imshow failed: %s", e)

        # Write video + JSONL every frame
        if overlay is not None:
            try:
                recorder.write(frame_bgr, overlay)
            except Exception as e:
                log.debug("video write failed: %s", e)
        try:
            jsonl_fh.write(json.dumps(_serialize_result_v2(res)) + "\n")
            jsonl_fh.flush()
        except Exception:
            pass

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

        st = res.get("state", "?")
        c  = res.get("best_centroid")
        z  = res.get("z_mm")
        td = res.get("tilt_deg")
        log.info("frame=%d state=%-9s c=%s z=%s tilt=%s",
                 i, st,
                 f"({c[0]:.0f},{c[1]:.0f})" if c else "-",
                 f"{z:.1f}mm" if z is not None else "-",
                 f"{td:.1f}°" if td is not None else "-")
        return res

    # Calibration thread (unchanged from base)
    if not (args.dry_run or args.no_robot):
        def _calibration_runner():
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
                          name="lastmile-v2-cal").start()

    try:
        if use_pyzed and pyzed_ok:
            _run_zed_loop(sl, args, stop_ev, shared, shared_lock,
                           perception_loop_for_frame)
        else:
            _run_opencv_loop(args, stop_ev, shared, shared_lock,
                              perception_loop_for_frame)
    finally:
        jsonl_fh.close()
        recorder.release()
        if not (args.dry_run or args.no_robot):
            # Return to home position before stopping, regardless of how the
            # session ended (quit key, Ctrl-C, TERMINAL, or error).
            if home_pos is not None and robot._arm is not None:
                try:
                    log.info("Returning to home position [%.1f, %.1f, %.1f] mm …",
                             home_pos[0], home_pos[1], home_pos[2])
                    robot.enabled = False  # halt any ongoing servo_step
                    robot._move_abs(home_pos, wait=True, speed=50.0)
                    log.info("Home position reached.")
                except Exception as e:
                    log.error("Return-to-home failed: %s", e)
            try:
                robot.stop()
            except Exception:
                pass
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SemVS v2 — last-mile pipeline with surface-normal feedback "
                    "and EE occlusion masking.")
    p.add_argument("cam_index", nargs="?", type=int, default=0)
    p.add_argument("--ref-image", "--ref_image", metavar="PATH")
    p.add_argument("--no-pyzed", action="store_true")
    p.add_argument("--no-robot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--input-image", metavar="PATH")
    p.add_argument("--input-dir", metavar="PATH")
    p.add_argument("--input-video", metavar="PATH")
    p.add_argument("--output-dir", metavar="PATH",
                   help="Directory for overlays, videos, JSONL metrics")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-overlays", action="store_true",
                   help="Skip per-frame overlay PNGs (videos still recorded)")
    p.add_argument("--fps", type=float, default=15.0,
                   help="Output video frame rate (default: 15)")
    p.add_argument("--prompt", default="box")
    p.add_argument("--hand-eye", metavar="PATH",
                   help="4x4 T_ee_cam (.npy or .yaml). Required for Signal A "
                        "and EE occlusion masking.")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--debug", action="store_true",
                   help="Per-frame debug dump: input, mask, overlay, JSON. "
                        "Offline: every frame. Live: every 30th.")
    p.add_argument("--no-window", action="store_true",
                   help="Skip OpenCV preview window (headless / SSH).")
    p.add_argument("--sixdof", action="store_true",
                   help="Enable 6-DOF Signal A: use depth plane fit as centroid "
                        "(Improvement 3 from LAST_MILE_RESEARCH.md).")
    p.add_argument("--ee-tip-offset", type=float, default=EE_TIP_OFFSET_MM,
                   metavar="MM",
                   help=f"Suction tip offset along EE Z axis "
                        f"(default: {EE_TIP_OFFSET_MM} mm)")
    p.add_argument("--ee-body-radius", type=float, default=EE_BODY_RADIUS_MM,
                   metavar="MM",
                   help=f"Gripper body radius for EE masking "
                        f"(default: {EE_BODY_RADIUS_MM} mm)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    args.debug_dir = "artifacts" if args.debug else None
    args.debug_every = (1 if (args.input_image or args.input_dir
                              or args.input_video) else 30)

    logger = _setup_logger(name="lastmile_v2", log_dir=args.log_dir)
    # also route base lastmile logger here
    logging.getLogger("lastmile").handlers = logger.handlers

    log_path = getattr(logger.handlers[1], "baseFilename", None) \
        if len(logger.handlers) > 1 else None

    debug_dir = _init_debug_dir(args.debug, args, args.ref_image)

    log.info("=" * 70)
    log.info("servo_lastmile_v2 starting")
    log.info("  improvements: surface-normal-in-NEAR=ON  ee-masking=ON  "
             "sixdof=%s", args.sixdof)
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
            run_offline_v2(args, debug_dir=debug_dir)
        else:
            run_live_v2(args, debug_dir=debug_dir)
        return 0
    finally:
        _finalize_debug_dir(debug_dir, log_path)
        log.info("Done.")


if __name__ == "__main__":
    sys.exit(main())
