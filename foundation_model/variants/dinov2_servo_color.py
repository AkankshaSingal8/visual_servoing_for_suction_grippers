#!/usr/bin/env python3
"""
SemVS – DINOv2-guided Visual Servoing

Uses DINOv2 feature matching + color similarity for initial object detection,
SAM2 for mask refinement, and anchor-based mask propagation for real-time
tracking. The centroid of the tracked mask drives the robot servo loop.

Pipeline:
  Frame 0:   DINOv2+color → bbox → SAM2 → mask → lock anchor
  Frame 1-N: SAM2 propagation from anchor/prev logits (fast, no DINOv2)
  Every K:   Re-run DINOv2 to correct drift

Usage:
    python dinov2_servo.py \\
        --scene-source 0 \\
        --reference input_image_transparent.png \\
        [--no-pyzed] [--no-robot]
"""
import logging
import time
import sys
import threading
import os

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

# ── Import DINOv2 matching pipeline ──────────────────────────────────
from dinov2_match_segment import (
    extract_patch_features,
    compute_color_similarity,
    compute_similarity_map,
    combine_similarity_maps,
    similarity_to_bbox,
    refine_with_sam2,
)

# ── Import infrastructure from negative_weighing ────────────────────
from negative_weighing import (
    MaskTracker,
    RobotController,
    ZedUndistorter,
    _robust_centroid,
    _mask_iou,
    _get_sam2,
    ROBOT_IP,
    MIN_RECORDING_DIM, ZED_RESOLUTION,
    NEG_POINT_COUNT,
    IOU_DRIFT_THRESH,
    PYZED_AVAILABLE,
)

try:
    import pyzed.sl as sl
except ImportError:
    sl = None

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    DEVICE = "cpu"

# ── DINOv2 matching parameters ───────────────────────────────────────
DINOV2_THRESHOLD_PCT = 93      # percentile for similarity thresholding
DINOV2_COLOR_WEIGHT  = 0.7     # weight for color vs DINOv2 features
DINOV2_REDETECT_INTERVAL = 30  # re-run DINOv2 every N frames
PATCH_SIZE = 14


# ═════════════════════════════════════════════════════════════════════
#  Reference image pre-processing (runs once at startup)
# ═════════════════════════════════════════════════════════════════════

class ReferenceModel:
    """
    Precomputes DINOv2 features and color stats from the reference image
    so they can be reused every time we need to re-detect.
    """

    def __init__(self, ref_path: str):
        ref_full = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        if ref_full is None:
            raise FileNotFoundError(f"Cannot load reference: {ref_path}")

        if ref_full.shape[2] == 4:
            self.ref_bgr = ref_full[:, :, :3]
            self.ref_alpha = ref_full[:, :, 3]
            fg_count = np.count_nonzero(self.ref_alpha > 128)
            logger.info(f"Reference: {self.ref_bgr.shape[1]}x{self.ref_bgr.shape[0]}, "
                        f"alpha present, fg pixels: {fg_count}")
        else:
            self.ref_bgr = ref_full
            self.ref_alpha = np.ones(ref_full.shape[:2], dtype=np.uint8) * 255
            logger.info("Reference: no alpha channel, treating entire image as object")

        # Extract DINOv2 features (one-time cost)
        logger.info("Extracting DINOv2 reference features...")
        self.ref_features, self.ref_proc_h, self.ref_proc_w = \
            extract_patch_features(self.ref_bgr, PATCH_SIZE)
        logger.info(f"  Reference features: {self.ref_features.shape}")

        # Build patch-level foreground mask
        ref_alpha_resized = cv2.resize(
            self.ref_alpha,
            (self.ref_proc_w, self.ref_proc_h),
            interpolation=cv2.INTER_NEAREST,
        )
        rh_patches = self.ref_proc_h // PATCH_SIZE
        rw_patches = self.ref_proc_w // PATCH_SIZE
        self.ref_mask_patches = np.zeros((rh_patches, rw_patches), dtype=np.uint8)
        for py in range(rh_patches):
            for px in range(rw_patches):
                patch_region = ref_alpha_resized[
                    py * PATCH_SIZE:(py + 1) * PATCH_SIZE,
                    px * PATCH_SIZE:(px + 1) * PATCH_SIZE,
                ]
                if np.mean(patch_region > 128) > 0.5:
                    self.ref_mask_patches[py, px] = 1

        fg_patches = np.count_nonzero(self.ref_mask_patches)
        logger.info(f"  Foreground patches: {fg_patches}/{self.ref_mask_patches.size}")

    def detect_in_scene(self, scene_bgr: np.ndarray):
        """
        Run DINOv2+color matching on a scene image.

        Returns
        -------
        bbox : (x1, y1, x2, y2) or None
        sam_mask : (H, W) uint8 binary mask or None
        sam_score : float
        sim_upscaled : (H, W) similarity heatmap
        """
        sh, sw = scene_bgr.shape[:2]

        # Extract scene features
        scene_features, scene_proc_h, scene_proc_w = \
            extract_patch_features(scene_bgr, PATCH_SIZE)

        # DINOv2 similarity
        dinov2_sim = compute_similarity_map(
            self.ref_features, self.ref_mask_patches, scene_features)

        # Color similarity
        color_sim = compute_color_similarity(
            self.ref_bgr, self.ref_alpha, scene_bgr,
            scene_proc_h, scene_proc_w, PATCH_SIZE)

        # Combine
        sim_map = combine_similarity_maps(
            dinov2_sim, color_sim,
            alpha=1.0 - DINOV2_COLOR_WEIGHT)

        # Extract bbox
        bbox, _, sim_upscaled = similarity_to_bbox(
            sim_map, sh, sw, PATCH_SIZE, DINOV2_THRESHOLD_PCT)

        logger.info(f"DINOv2+Color bbox: {bbox}")

        # SAM2 refinement
        sam_mask, sam_score = refine_with_sam2(scene_bgr, bbox)
        mask_area = np.count_nonzero(sam_mask)
        logger.info(f"SAM2 mask: score={sam_score:.3f}, "
                    f"area={mask_area}px ({100.0 * mask_area / (sh * sw):.1f}%)")

        return bbox, sam_mask, sam_score, sim_upscaled


# ═════════════════════════════════════════════════════════════════════
#  Core pipeline: DINOv2 detection + SAM2 propagation
# ═════════════════════════════════════════════════════════════════════

def run_dinov2_pipeline(
    image_bgr: np.ndarray,
    ref_model: ReferenceModel,
    tracker: MaskTracker,
) -> dict:
    """
    Detection → SAM2 segmentation → centroid pipeline.

    Mask-input priority:
      1. tracker.prev_logits   – temporal propagation (most frames)
      2. tracker.anchor_logits – stable anchor (after drift reset)
      3. DINOv2 fresh detection – initial frame or periodic re-detection
    """
    res = dict(mask_np=None, best_centroid=None, bbox=None)
    h, w = image_bgr.shape[:2]

    # ── Decide whether to run full DINOv2 detection this frame ────────
    need_detection = False
    if tracker.prev_logits is None and not tracker.anchor_locked:
        # First frame or no tracking state at all
        need_detection = True
    elif tracker.frame_count > 0 and tracker.frame_count % DINOV2_REDETECT_INTERVAL == 0:
        need_detection = True

    # ── Path A: Full DINOv2 detection ────────────────────────────────
    if need_detection:
        logger.info("--- Running DINOv2 detection ---")
        bbox, sam_mask, sam_score, sim_upscaled = \
            ref_model.detect_in_scene(image_bgr)
        res["bbox"] = bbox

        if sam_mask is not None and sam_score > 0.5:
            res["mask_np"] = sam_mask

            # Get SAM2 logits for anchor seeding
            pred = _get_sam2()
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pred.set_image(rgb)
            x1, y1, x2, y2 = bbox
            box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
            masks, scores, logits = pred.predict(
                box=box_np, multimask_output=True, return_logits=True)

            if masks is not None and len(masks) > 0:
                best_idx = int(np.argmax(scores))
                tracker_logits = logits[best_idx:best_idx + 1]

                centroid = _robust_centroid(sam_mask)
                res["best_centroid"] = centroid

                tracker.update(sam_mask, tracker_logits, centroid)
                logger.info(f"DINOv2 detection done. Centroid: {centroid}, "
                            f"anchor_locked: {tracker.anchor_locked}")
            else:
                tracker.update(sam_mask, None, _robust_centroid(sam_mask))
        else:
            logger.info("DINOv2 detection: low SAM2 score or no mask")
            tracker.update(None, None, None)

        return res

    # ── Path B: SAM2 propagation (fast path, most frames) ────────────
    try:
        pred = _get_sam2()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)

        # Build SAM2 prompt arrays
        point_coords = []
        point_labels = []

        # Positive points from previous mask
        pos_pts = tracker.sample_positive_points(h, w, n=1)
        if pos_pts is not None:
            for pt in pos_pts:
                point_coords.append(pt)
                point_labels.append(1)

        # Negative points from mask history
        neg_pts = tracker.sample_negative_points(h, w, NEG_POINT_COUNT)
        if neg_pts is not None:
            for pt in neg_pts:
                point_coords.append(pt)
                point_labels.append(0)
            logger.debug(f"SAM2 propagation: "
                         f"+{len(pos_pts) if pos_pts is not None else 0} pos, "
                         f"+{len(neg_pts)} neg points")

        # Use multimask_output=True until anchor is locked (better exploration),
        # then switch to single-mask mode for stable propagation
        sam_kwargs = dict(
            multimask_output=not tracker.anchor_locked,
            return_logits=True,
        )

        if point_coords:
            sam_kwargs["point_coords"] = np.array(point_coords, dtype=np.float32)
            sam_kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)

        # Mask-input priority chain
        if tracker.prev_logits is not None:
            sam_kwargs["mask_input"] = tracker.prev_logits
        elif tracker.anchor_logits is not None:
            sam_kwargs["mask_input"] = tracker.anchor_logits
            logger.info("SAM2: using anchor logits as prior")

        # Run SAM2
        masks, scores_sam, logits = pred.predict(**sam_kwargs)

        if masks is not None and len(masks) > 0:
            best_idx = int(np.argmax(scores_sam))
            mask_out = (masks[best_idx] > 0).astype(np.uint8) * 255
            tracker_logits = logits[best_idx:best_idx + 1]

            # Drift check
            if tracker.prev_mask is not None:
                iou = _mask_iou(mask_out, tracker.prev_mask)
                if iou < IOU_DRIFT_THRESH:
                    logger.info(f"SAM2: IoU={iou:.2f} < {IOU_DRIFT_THRESH} — "
                                f"drift detected, forcing DINOv2 re-detection")
                    tracker.reset(keep_anchor=True)
                    return run_dinov2_pipeline(image_bgr, ref_model, tracker)

            res["mask_np"] = mask_out
            centroid = _robust_centroid(mask_out)
            res["best_centroid"] = centroid

            tracker.update(mask_out, tracker_logits, centroid)
            logger.debug(f"SAM2 propagation: score={scores_sam[best_idx]:.3f}, "
                         f"centroid={centroid}")
        else:
            logger.info("SAM2 propagation: no mask — forcing DINOv2 re-detection")
            tracker.reset(keep_anchor=True)
            return run_dinov2_pipeline(image_bgr, ref_model, tracker)

    except Exception as e:
        logger.exception("SAM2 propagation failed: %s", e)
        tracker.update(None, None, None)

    return res


# ═════════════════════════════════════════════════════════════════════
#  Camera streamer with DINOv2 detection
# ═════════════════════════════════════════════════════════════════════

class DINOv2CameraStreamer(threading.Thread):
    """
    Captures frames, runs DINOv2-based segmentation pipeline, overlays
    results, and optionally drives the robot servo loop.
    """

    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController,
                 ref_model: ReferenceModel,
                 use_pyzed: bool = True):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.stop_event = stop_event
        self.robot = robot
        self.ref_model = ref_model
        self._use_pyzed = use_pyzed and PYZED_AVAILABLE

        self._latest_left = None
        self._frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()
        self._tracker = MaskTracker()
        self._undistorter = None

    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        logger.info("Calibration thread: waiting for models...")
        self._models_ready.wait()
        logger.info("Calibration thread: starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    # ── segmentation loop ─────────────────────────────────────────────
    def _seg_loop(self):
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_left.copy() if self._latest_left is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                res = run_dinov2_pipeline(frame, self.ref_model, self._tracker)
                with self.data_lock:
                    self._result = res
                if not self._models_ready.is_set():
                    logger.info("Models ready — calibration may now proceed.")
                    self._models_ready.set()
                if self.robot is not None and res.get("best_centroid") is not None:
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e:
                logger.error("Pipeline error: %s", e)
                logger.exception("Details:")
                time.sleep(1)

    # ── top-level dispatch ────────────────────────────────────────────
    def run(self):
        if self._use_pyzed:
            self._run_pyzed()
        else:
            self._run_opencv()

    # ── PyZED path ────────────────────────────────────────────────────
    def _run_pyzed(self):
        cam = sl.Camera()

        init_params = sl.InitParameters()
        res_map = {
            "HD2K":   sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720":  sl.RESOLUTION.HD720,
            "VGA":    sl.RESOLUTION.VGA,
        }
        init_params.camera_resolution = res_map.get(ZED_RESOLUTION,
                                                     sl.RESOLUTION.HD720)
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.info(f"PyZED: camera open failed ({err}) — falling back to OpenCV")
            self._run_opencv()
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        anno_path = os.path.abspath(f"vs_dinov2_{ts}.mp4")
        video_writer = None

        mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "DINOv2 Servo  |  [v] servo  [r] reset  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        try:
            while not self.stop_event.is_set():
                if cam.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.01)
                    continue

                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame_bgra = mat.get_data()
                frame = frame_bgra[:, :, :3].copy()

                with self._frame_lock:
                    self._latest_left = frame.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(frame, res)
                rendered_write = _ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw = rendered_write.shape[:2]
                    video_writer = cv2.VideoWriter(
                        anno_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if video_writer.isOpened():
                        logger.info(f"Recording → {anno_path}  ({rw}x{rh})")
                    else:
                        video_writer.release()
                        video_writer = None

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    logger.info(f"Servo: {'ON' if self.robot.enabled else 'OFF'}")
                elif key == ord("r"):
                    self._tracker.reset(keep_anchor=True)
                    logger.info("Tracker soft-reset (anchor kept)")
                elif key == ord("R"):
                    self._tracker.reset(keep_anchor=False)
                    logger.info("Tracker FULL reset (anchor cleared)")
        finally:
            cam.close()
            if video_writer is not None:
                video_writer.release()
                logger.info(f"Recording saved: {anno_path}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── OpenCV fallback path ──────────────────────────────────────────
    def _run_opencv(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            logger.info(f"Failed to open camera {self.cam_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "DINOv2 Servo  |  [v] servo  [r] reset  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        video_writer = None
        ts = time.strftime("%Y%m%d_%H%M%S")

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                fh, fw = frame.shape[:2]
                left = frame[:, :fw // 2].copy()

                if self._undistorter is None:
                    lh, lw = left.shape[:2]
                    self._undistorter = ZedUndistorter.from_frame_size(lw, lh)
                left = self._undistorter.undistort(left)

                with self._frame_lock:
                    self._latest_left = left.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(left, res)
                rendered_write = _ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw = rendered_write.shape[:2]
                    video_path = os.path.abspath(f"vs_dinov2_{ts}.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if video_writer.isOpened():
                        logger.info(f"Recording → {video_path}  ({rw}x{rh})")
                    else:
                        video_writer.release()
                        video_writer = None

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    logger.info(f"Servo: {'ON' if self.robot.enabled else 'OFF'}")
                elif key == ord("r"):
                    self._tracker.reset(keep_anchor=True)
                    logger.info("Tracker soft-reset (anchor kept)")
                elif key == ord("R"):
                    self._tracker.reset(keep_anchor=False)
                    logger.info("Tracker FULL reset (anchor cleared)")
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── overlay renderer ──────────────────────────────────────────────
    def _render(self, left: np.ndarray, res: dict) -> np.ndarray:
        display = left.copy()
        h, w = display.shape[:2]

        mask_np = res.get("mask_np")
        if mask_np is not None and mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # SAM2 mask overlay (green tint + contour)
        if mask_np is not None:
            overlay = np.zeros_like(display)
            overlay[:, :, 1] = mask_np
            display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
            cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cnts, -1, (0, 255, 0), 2)

        # Bbox overlay
        bbox = res.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Centroid crosshair + error line to image centre
        centroid = res.get("best_centroid")
        if centroid is not None:
            cx, cy = centroid
            cv2.drawMarker(display, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            ic_x, ic_y = w // 2, h // 2
            cv2.drawMarker(display, (ic_x, ic_y), (255, 0, 0),
                           cv2.MARKER_CROSS, 15, 1)
            cv2.line(display, (ic_x, ic_y), (cx, cy), (255, 255, 0), 1)
            err = np.hypot(cx - ic_x, cy - ic_y)
            cv2.putText(display, f"err={err:.0f}px", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Status HUD
        status_lines = [
            f"anchor: {'LOCKED' if self._tracker.anchor_locked else 'no'}",
            f"frames: {self._tracker.frame_count}",
        ]
        if self.robot is not None:
            status_lines.append(f"servo: {'ON' if self.robot.enabled else 'OFF'}")
            status_lines.append(f"cal: {self.robot.cal_status}")
        for i, line in enumerate(status_lines):
            cv2.putText(display, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display


def _ensure_min_dim(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    if short < MIN_RECORDING_DIM:
        scale = MIN_RECORDING_DIM / short
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
    return img


# ═════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DINOv2-guided Visual Servoing")
    parser.add_argument("--scene-source", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--reference", "-r", required=True,
                        help="Path to reference image (RGBA with transparent bg)")
    parser.add_argument("--no-pyzed", action="store_true",
                        help="Force OpenCV capture even if PyZED is available")
    parser.add_argument("--no-robot", action="store_true",
                        help="Disable robot connection (vision-only mode)")
    parser.add_argument("--threshold-pct", type=float, default=DINOV2_THRESHOLD_PCT,
                        help=f"Similarity threshold percentile (default: {DINOV2_THRESHOLD_PCT})")
    parser.add_argument("--color-weight", type=float, default=DINOV2_COLOR_WEIGHT,
                        help=f"Color similarity weight (default: {DINOV2_COLOR_WEIGHT})")
    parser.add_argument("--redetect-interval", type=int, default=DINOV2_REDETECT_INTERVAL,
                        help=f"Re-run DINOv2 every N frames (default: {DINOV2_REDETECT_INTERVAL})")
    args = parser.parse_args()

    # Apply CLI overrides
    DINOV2_THRESHOLD_PCT = args.threshold_pct
    DINOV2_COLOR_WEIGHT = args.color_weight
    DINOV2_REDETECT_INTERVAL = args.redetect_interval

    logger.info(f"DINOv2 threshold: {DINOV2_THRESHOLD_PCT}%")
    logger.info(f"Color weight: {DINOV2_COLOR_WEIGHT}")
    logger.info(f"Re-detect interval: {DINOV2_REDETECT_INTERVAL} frames")
    logger.info(f"PyZED: {'available' if PYZED_AVAILABLE else 'not available'}"
                f"{' (disabled)' if args.no_pyzed else ''}")

    # Build reference model (loads DINOv2, extracts features)
    ref_model = ReferenceModel(args.reference)

    # Robot
    robot = RobotController(ROBOT_IP)
    if not args.no_robot:
        robot.connect()
    else:
        logger.info("Robot disabled (--no-robot)")

    # Start camera streamer
    stop_ev = threading.Event()
    cam_thread = DINOv2CameraStreamer(
        cam_index=args.scene_source,
        stop_event=stop_ev,
        robot=robot,
        ref_model=ref_model,
        use_pyzed=not args.no_pyzed,
    )
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        logger.info("Done.")
