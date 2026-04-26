#!/usr/bin/env python3
"""
servo_pipeline_sam3_exp1.py — Experiment 1 wrapper around servo_pipeline_sam3.

Imports RobotController, CameraStreamer, and all pipeline helpers from
servo_pipeline_sam3 without modifying that file. Adds:

  * Convergence detection: pixel error ≤ VS_DEAD_ZONE for N consecutive
    servo steps → write final_position.json → auto-exit.
  * --convergence-output PATH   path for final_position.json
  * --convergence-frames N      consecutive in-zone frames required (default 5)

Usage:
    python foundation_model/servo_pipeline_sam3_exp1.py \\
        --ref-image path/to/ref.png \\
        --convergence-output trial_dir/final_position.json \\
        [--convergence-frames 5] \\
        [--debug] [--log-dir logs/exp1]

All other flags (--no-robot, --dry-run, --no-pyzed, --prompt, etc.) are
passed straight through to the underlying pipeline.
"""

import sys
import os
import json
import time
import threading
import argparse

import numpy as np

# ── Import the existing pipeline without modification ────────────────────────
_FM_DIR = os.path.dirname(os.path.abspath(__file__))
if _FM_DIR not in sys.path:
    sys.path.insert(0, _FM_DIR)

import servo_pipeline_sam3 as _base

# Re-export the constants we need to read at runtime (not copies – live refs)
VS_DEAD_ZONE = _base.VS_DEAD_ZONE
ROBOT_IP     = _base.ROBOT_IP


# ── Convergence monitor ──────────────────────────────────────────────────────

class ConvergenceMonitor:
    """
    Wraps a RobotController's servo_step to detect convergence and write
    final_position.json.

    Monkey-patches robot.servo_step so no changes to the base file are needed.
    The original servo_step is still called on every invocation.
    """

    def __init__(self, robot: _base.RobotController,
                 output_path: str,
                 stop_event: threading.Event,
                 required_frames: int = 5):
        self._robot          = robot
        self._output_path    = output_path
        self._stop_event     = stop_event
        self._required       = required_frames

        self._consec_in_zone = 0
        self._done           = False
        self._start_time     = None
        self._pos_history    = []   # list of [x,y,z] EE positions
        self._last_sim       = None

        # Monkey-patch
        self._original_servo_step = robot.servo_step
        robot.servo_step = self._patched_servo_step

    def set_similarity(self, sim: float):
        self._last_sim = sim

    def _patched_servo_step(self, centroid: tuple, image_shape: tuple):
        # Call original first so the robot actually moves
        self._original_servo_step(centroid, image_shape)

        if self._done or not self._robot.enabled:
            return

        # Read current EE position for tracking
        pos = self._robot._get_pos()
        if pos is None:
            return

        if self._start_time is None:
            self._start_time = time.time()
        self._pos_history.append(list(pos[:3]))

        # Compute pixel error (same formula as servo_step)
        h, w = image_shape[:2]
        ex = centroid[0] - w // 2
        ey = centroid[1] - h // 2
        err_r = float(np.hypot(ex, ey))

        if err_r <= VS_DEAD_ZONE:
            self._consec_in_zone += 1
        else:
            self._consec_in_zone = 0

        if self._consec_in_zone >= self._required:
            self._done = True
            self._write(pos, ex, ey, err_r)
            self._stop_event.set()

    def _write(self, pos, ex: float, ey: float, err_r: float):
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        path_mm = 0.0
        for i in range(1, len(self._pos_history)):
            path_mm += float(np.linalg.norm(
                np.array(self._pos_history[i]) - np.array(self._pos_history[i - 1])))

        data = {
            "trial_id":             None,
            "final_ee_position_mm": list(pos[:3]),
            "final_ee_full_pose":   list(pos),
            "final_pixel_error":    [int(ex), int(ey)],
            "final_pixel_error_r":  round(float(err_r), 2),
            "convergence_time_s":   round(elapsed, 3),
            "convergence_frames":   len(self._pos_history),
            "final_similarity":     (float(self._last_sim)
                                     if self._last_sim is not None else None),
            "path_length_mm":       round(path_mm, 2),
            "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = self._output_path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._output_path)
        print(f"[exp1] Convergence: final_position.json -> {self._output_path}")


# ── Similarity pass-through (subclass CameraStreamer) ────────────────────────

class Exp1CameraStreamer(_base.CameraStreamer):
    """
    Thin subclass that passes the per-frame similarity score to the
    ConvergenceMonitor so it ends up in final_position.json.
    """

    def __init__(self, *args, convergence_monitor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv_monitor = convergence_monitor

    def _seg_loop(self):
        self._process_ref_image()

        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = (self._latest_left.copy()
                         if self._latest_left is not None else None)
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                res = _base.run_pipeline(
                    frame, _base.PROMPT, self._tracker,
                    ref_crop=self.ref_crop,
                    ref_features=self.ref_features,
                )
                self._stabilize_grasp(res)

                with self.data_lock:
                    self._result = res

                if (self._debug_dir
                        and self._debug_frame_idx % self._debug_every == 0):
                    overlay = None
                    try:
                        overlay = self._render(frame, res)
                    except Exception:
                        overlay = None
                    _base._dump_debug_frame(
                        self._debug_dir, self._debug_frame_idx,
                        frame, res, overlay_bgr=overlay, source="live")
                self._debug_frame_idx += 1

                if not self._models_ready.is_set():
                    self._models_ready.set()

                if self.robot is not None and res.get("best_centroid") is not None:
                    # Forward similarity to monitor before servo_step is called
                    if (self._conv_monitor is not None
                            and res.get("similarity") is not None):
                        self._conv_monitor.set_similarity(float(res["similarity"]))
                    self.robot.servo_step(res["best_centroid"], frame.shape)

            except Exception as e:
                import traceback
                print(f"[exp1] Pipeline error: {e}")
                traceback.print_exc()
                time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM3 visual servoing — Experiment 1 wrapper "
                    "(convergence detection + auto-exit)")

    # Pass-through args (subset used by the live path)
    parser.add_argument("cam_index", nargs="?", type=int, default=0)
    parser.add_argument("--ref-image", "--ref_image", metavar="PATH")
    parser.add_argument("--no-pyzed",  action="store_true")
    parser.add_argument("--no-robot",  action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--prompt",    default=None)
    parser.add_argument("--sam3-model", default=None)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--mask-thresh",  type=float, default=None)
    parser.add_argument("--log-dir",   default="logs")
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--output-dir", metavar="PATH", default=None,
                        help="Debug artifacts directory")

    # Experiment 1 specific
    parser.add_argument("--convergence-output", metavar="PATH", required=True,
                        help="Write final_position.json here on convergence, "
                             "then exit automatically.")
    parser.add_argument("--convergence-frames", type=int, default=5,
                        help="Consecutive servo steps within dead zone required "
                             "for convergence (default: 5)")

    args = parser.parse_args()

    # Apply overrides to base module globals
    if args.prompt:
        _base.PROMPT = args.prompt
    if args.sam3_model:
        _base.SAM3_HF_MODEL_ID = args.sam3_model
    if args.score_thresh is not None:
        _base.SAM3_SCORE_THRESH = float(args.score_thresh)
    if args.mask_thresh is not None:
        _base.SAM3_MASK_THRESH = float(args.mask_thresh)

    log = _base._setup_logger(log_dir=args.log_dir)
    log.info("Experiment 1 wrapper: convergence → %s  (need %d frames in dead zone)",
             args.convergence_output, args.convergence_frames)

    # Reconstruct args-like object for _init_debug_dir
    args.debug_dir   = "artifacts" if args.debug else None
    args.debug_every = 30
    args.input_image = None
    args.input_dir   = None
    args.input_video = None

    debug_dir = _base._init_debug_dir(args.debug_dir, args, args.ref_image)

    robot = _base.RobotController(ROBOT_IP)
    if args.no_robot or args.dry_run:
        log.info("Robot disabled (%s)", "--dry-run" if args.dry_run else "--no-robot")
    else:
        robot.connect()

    stop_ev = threading.Event()

    monitor = ConvergenceMonitor(
        robot=robot,
        output_path=args.convergence_output,
        stop_event=stop_ev,
        required_frames=args.convergence_frames,
    )

    cam_thread = Exp1CameraStreamer(
        cam_index          = args.cam_index,
        stop_event         = stop_ev,
        robot              = robot,
        ref_image_path     = args.ref_image,
        use_pyzed          = not args.no_pyzed,
        debug_dir          = debug_dir,
        debug_every        = args.debug_every,
        convergence_monitor= monitor,
    )
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        _base._finalize_debug_dir(debug_dir)
        log.info("Done.")
