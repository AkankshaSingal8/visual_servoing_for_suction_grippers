#!/usr/bin/env python3
"""
servo_lastmile_v2_ext — Extended last-mile pipeline with pluggable detectors
and wrist tilt correction.

Extends LastMilePipelineV2 and the CLI from servo_lastmile_v2 to support:
  - --detector {sam3, gdino, gdino+sam2}  (detector backend selection)
  - Tilt correction wired into the live perception loop via RobotControllerExt
  - detector_used field logged in frames.jsonl on every frame
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import warnings

# xformers is optional acceleration used by SAM2/CoTracker3; suppress the noise
# if it is not installed — it does not affect correctness.
warnings.filterwarnings("ignore", message=".*xformers.*")
warnings.filterwarnings("ignore", message=".*No module named 'xformers'.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*sam2.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*cotracker.*")

import cv2
import numpy as np

try:
    from foundation_model.servo_lastmile_v2 import (
        LastMilePipelineV2,
        run_offline_v2,
        _build_argparser,
        _serialize_result_v2,
        VideoRecorder,
        _setup_logger,
        _init_debug_dir,
        _finalize_debug_dir,
        _dump_debug_frame,
    )
    from foundation_model.servo_lastmile import (
        _xarm_pose_to_T,
        _run_zed_loop,
        _run_opencv_loop,
        DEFAULT_INTRINSICS,
        gaussian_mask_samples,
        _make_overlay,
    )
except ImportError:
    from servo_lastmile_v2 import (
        LastMilePipelineV2,
        run_offline_v2,
        _build_argparser,
        _serialize_result_v2,
        VideoRecorder,
        _setup_logger,
        _init_debug_dir,
        _finalize_debug_dir,
        _dump_debug_frame,
    )
    from servo_lastmile import (
        _xarm_pose_to_T,
        _run_zed_loop,
        _run_opencv_loop,
        DEFAULT_INTRINSICS,
        gaussian_mask_samples,
        _make_overlay,
    )

try:
    from full_system_pipeline.detection_backends import make_runner
except ImportError:
    try:
        from detection_backends import make_runner
    except ImportError:
        def make_runner(detector_name: str, ref_image: str | None, prompt: str):
            try:
                from foundation_model.servo_lastmile import make_default_sam3_runner
            except ImportError:
                from servo_lastmile import make_default_sam3_runner
            return make_default_sam3_runner(ref_image, prompt)

log = logging.getLogger("lastmile_v2_ext")

# More tracking points than the base 80 — Gaussian-biased near grasp centroid.
# Higher density keeps more points in-frame as the object exits the border.
EXT_COTRACKER_INIT_POINTS = 150


# ---------------------------------------------------------------------------
# Ground-truth helpers + error plotting
# ---------------------------------------------------------------------------

def _plot_xyz_error(jsonl_path: str, out_dir: str) -> None:
    """Read frames.jsonl and save absolute per-frame X/Y/Z error vs time as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping error plot.")
        return

    times, ex, ey, ez = [], [], [], []
    t0 = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("error_x_mm") is None:
                    continue
                ts = d.get("frame_timestamp")
                if ts is None:
                    continue
                if t0 is None:
                    t0 = ts
                times.append(ts - t0)
                ex.append(d["error_x_mm"])
                ey.append(d["error_y_mm"])
                ez.append(d["error_z_mm"])
    except FileNotFoundError:
        log.warning("JSONL not found at %s; skipping error plot.", jsonl_path)
        return

    if not times:
        log.info("No per-frame error data in JSONL (robot offline?); skipping plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, vals, label, color in zip(
            axes,
            [ex, ey, ez],
            ["Error X (mm)", "Error Y (mm)", "Error Z (mm)"],
            ["#e74c3c", "#2ecc71", "#3498db"]):
        ax.plot(times, vals, color=color, linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Per-frame absolute XYZ error vs ground truth")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "error_xyz_over_time.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Error plot saved → %s", out_path)


def _load_gt_pose(path: str) -> np.ndarray:
    """Load gt_pose JSON and return xyz as a (3,) float64 array in mm."""
    with open(path) as f:
        data = json.load(f)
    gt = data["gt_pose"]  # [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
    return np.array(gt[:3], dtype=np.float64)


def _compute_xyz_error(robot_pos: list, gt_xyz: np.ndarray) -> dict:
    """Return per-axis and Euclidean XYZ error between robot_pos and gt_xyz."""
    pos = np.array(robot_pos[:3], dtype=np.float64)
    diff = pos - gt_xyz
    return {
        "gt_xyz_mm": gt_xyz.tolist(),
        "final_xyz_mm": pos.tolist(),
        "error_x_mm": float(diff[0]),
        "error_y_mm": float(diff[1]),
        "error_z_mm": float(diff[2]),
        "error_xyz_mm": float(np.linalg.norm(diff)),
    }


# ---------------------------------------------------------------------------
# Overlay helper — replaces _make_overlay_v2 in both offline and live runners
# ---------------------------------------------------------------------------

_TILT_KEYS = ("tilt_deg", "roll_deg", "pitch_deg",
              "near_plane_normal", "near_plane_n_inliers", "_tilt")


def _make_overlay_ext(frame_bgr: np.ndarray, res: dict) -> np.ndarray:
    """
    Ext-pipeline overlay:
      SAM3 phase  -> RED cross + circle at best_centroid, label "SAM3 GRASP"
      track phase -> ORANGE cross + circle at best_centroid (c_B),
                     dim-red reference circle at c_E (frozen SAM3 anchor)
    All other elements (green mask, bbox, image-centre crosshair, depth HUD)
    come from the base _make_overlay() call.
    """
    phase = res.get("_near_phase")
    best  = res.get("best_centroid")
    c_E   = res.get("c_E")

    # Strip tilt keys and suppress best_centroid so base _make_overlay does
    # not draw its own red GRASP cross — we draw it ourselves below.
    res_base = {k: v for k, v in res.items()
                if k not in _TILT_KEYS and k != "best_centroid"}
    res_base["c_A"] = None  # suppress Signal A sub-dot; not meaningful in ext

    img = _make_overlay(frame_bgr, res_base)

    if phase == "track":
        # Frozen SAM3 anchor — dim red reference circle (no cross)
        if c_E is not None:
            ex, ey = int(c_E[0]), int(c_E[1])
            cv2.circle(img, (ex, ey), 10, (0, 0, 180), 2)
            cv2.putText(img, "SAM3 ref", (ex + 13, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 180), 1)

        # CoTracker3 tracked grasp point — ORANGE cross
        if best is not None:
            bx, by = int(best[0]), int(best[1])
            cv2.circle(img, (bx, by), 18, (0, 165, 255), 2)
            cv2.drawMarker(img, (bx, by), (0, 165, 255),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.putText(img, "CoTracker3 GRASP", (bx + 22, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2)
    else:
        # SAM3 phase (or FAR / TERMINAL) — RED cross labeled "SAM3 GRASP"
        if best is not None:
            gx, gy = int(best[0]), int(best[1])
            cv2.circle(img, (gx, gy), 18, (0, 0, 255), 2)
            cv2.drawMarker(img, (gx, gy), (0, 0, 255),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.putText(img, "SAM3 GRASP", (gx + 22, gy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    return img


# ---------------------------------------------------------------------------
# Extended pipeline
# ---------------------------------------------------------------------------

class LastMilePipelineV2Ext(LastMilePipelineV2):
    """
    Extends LastMilePipelineV2 with a detector_name tag on every result dict
    so frames.jsonl always carries the detector backend that was active.
    """

    def __init__(self, *args, detector_name: str = "sam3", **kwargs):
        super().__init__(*args, **kwargs)
        self.detector_name = detector_name
        log.info("LastMilePipelineV2Ext: detector_name=%s", detector_name)

    def _enter_lock(self, frame_bgr: np.ndarray, sam3_result: dict,
                    depth_map: np.ndarray | None):
        lock = super()._enter_lock(frame_bgr, sam3_result, depth_map)
        if lock is None:
            return None
        # Replace the base uniform-80 seed with a Gaussian-150 seed.
        # Gaussian bias keeps density near the grasp centroid; more points
        # means CoTracker3 retains coverage longer as the box exits the frame.
        mask = lock.mask_uint8
        rng = np.random.default_rng(42)
        lock.init_points = gaussian_mask_samples(
            mask, EXT_COTRACKER_INIT_POINTS, sigma_frac=0.30, rng=rng)
        self.signal_B.reset(lock, frame_bgr)
        log.info("LOCK (ext): reseeded CoTracker3 with %d gaussian points",
                 lock.init_points.shape[0])
        return lock

    def step(self, frame_bgr: np.ndarray) -> dict:
        prev_phase = self._near_phase  # read before super() may flip it
        res = super().step(frame_bgr)
        res["detector_used"] = self.detector_name

        # Reseed CoTracker3 with more points immediately after the SAM3→track
        # phase switch. super().step() already called reset_from_mask with 80
        # points; we redo it with EXT_COTRACKER_INIT_POINTS Gaussian points so
        # the tracker has denser coverage as the object exits the frame border.
        if prev_phase == "sam3" and self._near_phase == "track":
            if self._last_sam3_mask is not None:
                self.signal_B.reset_from_mask(
                    self._last_sam3_mask, frame_bgr,
                    n_points=EXT_COTRACKER_INIT_POINTS,
                )
                log.info("Handoff: reseeded CoTracker3 with %d gaussian points",
                         EXT_COTRACKER_INIT_POINTS)

        # In track phase, use only CoTracker3 (Signal B) — no SAM3 anchor fallback.
        # If CoTracker3 returns None, best_centroid is None (servo pauses naturally).
        if res.get("_near_phase") == "track":
            c_B = res.get("c_B")
            res["best_centroid"] = c_B
            res["c_fused"] = c_B
            res["c_E"] = None
            res["weights"] = {"B": 1.0 if c_B is not None else 0.0}

        return res


# ---------------------------------------------------------------------------
# Offline runner
# ---------------------------------------------------------------------------

def run_offline_v2_ext(args: argparse.Namespace,
                        debug_dir: str | None = None) -> None:
    runner = make_runner(
        getattr(args, "detector", "sam3"),
        args.ref_image,
        getattr(args, "prompt", "box"),
    )

    out_dir = args.output_dir or os.path.join(
        "runs", time.strftime("lastmile_v2_ext_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "frames.jsonl")

    pipeline = LastMilePipelineV2Ext(
        sam3_runner=runner,
        ee_pose_provider=lambda: np.eye(4),
        depth_provider=lambda f: None,
        hand_eye_path=args.hand_eye,
        use_sixdof=getattr(args, "sixdof", False),
        detector_name=getattr(args, "detector", "sam3"),
    )

    recorder = VideoRecorder(out_dir, fps=getattr(args, "fps", 15.0))
    pipeline.transition_debug_dir = os.path.join(out_dir, "transition_debug")

    frames_iter = []
    if args.input_video:
        frames_iter.append(("video", args.input_video))
    elif args.input_image:
        frames_iter.append(("image", args.input_image))
    elif args.input_dir:
        for name in sorted(os.listdir(args.input_dir)):
            p = os.path.join(args.input_dir, name)
            if os.path.splitext(name.lower())[1] in {".png", ".jpg", ".jpeg"}:
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
                    res = pipeline.step(frame)
                    fh.write(json.dumps(_serialize_result_v2(res)) + "\n")
                    fh.flush()
                    overlay = _make_overlay_ext(frame, res)
                    recorder.write(frame, overlay)
                    if not getattr(args, "no_overlays", False):
                        cv2.imwrite(os.path.join(out_dir, f"overlay_{n:06d}.png"), overlay)
                    if debug_dir is not None:
                        _dump_debug_frame(debug_dir, n, frame, res, overlay, path)
                    n += 1
                cap.release()
            else:
                frame = cv2.imread(path)
                if frame is None:
                    continue
                res = pipeline.step(frame)
                fh.write(json.dumps(_serialize_result_v2(res)) + "\n")
                fh.flush()
                overlay = _make_overlay_ext(frame, res)
                recorder.write(frame, overlay)
                if not getattr(args, "no_overlays", False):
                    cv2.imwrite(os.path.join(out_dir, f"overlay_{n:06d}.png"), overlay)
                if debug_dir is not None:
                    _dump_debug_frame(debug_dir, n, frame, res, overlay, path)
                n += 1

    recorder.release()
    log.info("Wrote %d frame summaries to %s", n, jsonl_path)


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

def run_live_v2_ext(args: argparse.Namespace,
                     debug_dir: str | None = None) -> None:
    try:
        try:
            from full_system_pipeline.pipeline.robot_controller_ext import RobotControllerExt
        except ImportError:
            from robot_controller_ext import RobotControllerExt
    except ImportError as exc:
        log.error("RobotControllerExt not importable: %s — falling back to RobotController", exc)
        try:
            from foundation_model.servo_pipeline_sam3 import RobotController as RobotControllerExt
        except ImportError:
            from servo_pipeline_sam3 import RobotController as RobotControllerExt

    try:
        try:
            from foundation_model.servo_pipeline_sam3 import ZedUndistorter, ROBOT_IP
        except ImportError:
            from servo_pipeline_sam3 import ZedUndistorter, ROBOT_IP
    except Exception as e:
        log.error("Live mode needs servo_pipeline_sam3 importable: %s", e)
        sys.exit(2)

    if args.ref_image and not os.path.exists(args.ref_image):
        log.error("Reference image not found: %s", args.ref_image)
        sys.exit(1)

    gt_xyz = None
    if getattr(args, "gt_pose", None):
        if not os.path.exists(args.gt_pose):
            log.error("GT pose file not found: %s", args.gt_pose)
            sys.exit(1)
        gt_xyz = _load_gt_pose(args.gt_pose)
        log.info("Ground truth XYZ loaded: [%.3f, %.3f, %.3f] mm",
                 gt_xyz[0], gt_xyz[1], gt_xyz[2])

    robot = RobotControllerExt(ROBOT_IP)
    home_pos = None
    robot_armed = False  # True only when connect succeeded and _arm is not None
    if args.no_robot or args.dry_run:
        log.info("Robot disabled (%s)",
                 "--dry-run" if args.dry_run else "--no-robot")
    else:
        robot.connect()
        if robot._arm is None:
            log.error("Robot connect failed — _arm is None. "
                      "All robot operations will be skipped. "
                      "Check that the robot is powered on and reachable at %s.", ROBOT_IP)
        else:
            robot_armed = True
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

    detector_name = getattr(args, "detector", "sam3")
    runner = make_runner(detector_name, args.ref_image,
                         getattr(args, "prompt", "box"))

    pipeline = LastMilePipelineV2Ext(
        sam3_runner=runner,
        ee_pose_provider=ee_pose_provider,
        depth_provider=depth_provider,
        hand_eye_path=args.hand_eye,
        intrinsics=shared["intrinsics"],
        use_sixdof=getattr(args, "sixdof", False),
        detector_name=detector_name,
    )

    out_dir = getattr(args, "output_dir", None) or os.path.join(
        "runs", time.strftime("lastmile_v2_ext_live_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    recorder = VideoRecorder(out_dir, fps=15.0)
    jsonl_path = os.path.join(out_dir, "frames.jsonl")
    jsonl_fh = open(jsonl_path, "w")

    use_pyzed = not args.no_pyzed
    try:
        import pyzed.sl as sl
        pyzed_ok = True
    except Exception:
        sl = None
        pyzed_ok = False
        if use_pyzed:
            log.warning("PyZED not importable; falling back to OpenCV camera.")
        use_pyzed = False

    frame_idx = [0]
    debug_every = max(1, int(getattr(args, "debug_every", 30) or 30))
    terminal_error: dict | None = None  # filled once on first TERMINAL frame

    show_window = not getattr(args, "no_window", False)
    win_name = "lastmile v2 ext  |  [v] servo  [r] reset  [q] quit"
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
        current_robot_pos = None
        if robot_armed:
            try:
                pos = robot._get_pos()
                if pos is not None:
                    current_robot_pos = pos
                    with shared_lock:
                        shared["latest_pose"] = _xarm_pose_to_T(pos)
            except Exception as e:
                log.warning("Pose read failed: %s", e, exc_info=True)

        res = pipeline.step(frame_bgr)
        res["robot_pos"] = current_robot_pos  # [x,y,z,roll,pitch,yaw] mm/deg, None if offline

        if (robot_armed
                and res.get("state") != "TERMINAL"
                and res.get("best_centroid") is not None
                and not res.get("watchdog_alarm", False)):
            robot.servo_step(res["best_centroid"], frame_bgr.shape)
            # Read position after move so the frame log reflects where robot ended up
            try:
                post_pos = robot._get_pos()
                if post_pos is not None:
                    current_robot_pos = post_pos
                    res["robot_pos"] = post_pos
            except Exception as e:
                log.warning("Post-servo pose read failed: %s", e, exc_info=True)
        elif res.get("watchdog_alarm", False):
            log.warning("Servo paused — watchdog alarm active.")
        elif res.get("state") == "TERMINAL":
            log.info("Servo stopped — TERMINAL state reached.")
            nonlocal terminal_error
            if terminal_error is None and gt_xyz is not None and current_robot_pos is not None:
                terminal_error = _compute_xyz_error(current_robot_pos, gt_xyz)
                res.update(terminal_error)
                log.info(
                    "=== TERMINAL ERROR vs GT ===\n"
                    "  final  xyz: [%.3f, %.3f, %.3f] mm\n"
                    "  gt     xyz: [%.3f, %.3f, %.3f] mm\n"
                    "  error  xyz: [%.3f, %.3f, %.3f] mm\n"
                    "  ||error||:   %.3f mm",
                    terminal_error["final_xyz_mm"][0],
                    terminal_error["final_xyz_mm"][1],
                    terminal_error["final_xyz_mm"][2],
                    terminal_error["gt_xyz_mm"][0],
                    terminal_error["gt_xyz_mm"][1],
                    terminal_error["gt_xyz_mm"][2],
                    terminal_error["error_x_mm"],
                    terminal_error["error_y_mm"],
                    terminal_error["error_z_mm"],
                    terminal_error["error_xyz_mm"],
                )

        # Stamp per-frame absolute XYZ error into res so it lands in JSONL
        res["frame_timestamp"] = time.time()
        if gt_xyz is not None:
            rp = res.get("robot_pos")
            if rp is not None:
                pos = np.array(rp[:3], dtype=np.float64)
                err = np.abs(pos - gt_xyz)
                res["error_x_mm"] = float(err[0])
                res["error_y_mm"] = float(err[1])
                res["error_z_mm"] = float(err[2])
                res["error_xyz_mm"] = float(np.linalg.norm(pos - gt_xyz))
            else:
                res["error_x_mm"] = None
                res["error_y_mm"] = None
                res["error_z_mm"] = None
                res["error_xyz_mm"] = None

        overlay = _make_overlay_ext(frame_bgr, res)

        if show_window:
            try:
                cv2.imshow(win_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    _handle_key(key)
            except Exception as e:
                log.warning("imshow failed: %s", e, exc_info=True)

        recorder.write(frame_bgr, overlay)
        jsonl_fh.write(json.dumps(_serialize_result_v2(res)) + "\n")
        jsonl_fh.flush()

        i = frame_idx[0]
        frame_idx[0] += 1
        if debug_dir is not None and i % debug_every == 0:
            try:
                cv2.imwrite(os.path.join(debug_dir, "overlay_preview.png"), overlay)
            except Exception as e:
                log.warning("Debug overlay imwrite failed: %s", e, exc_info=True)
            _dump_debug_frame(debug_dir, i, frame_bgr, res, overlay, "live")

        st  = res.get("state", "?")
        c   = res.get("best_centroid")
        z   = res.get("z_mm")
        td  = res.get("tilt_deg")
        rp  = current_robot_pos  # [x, y, z, roll, pitch, yaw] or None
        log.info(
            "frame=%d state=%-9s c=%s depth=%s  robot xyz=(%.1f, %.1f, %.1f)mm  tilt=%s",
            i, st,
            f"({c[0]:.0f},{c[1]:.0f})" if c else "-",
            f"{z:.1f}mm" if z is not None else "-",
            rp[0] if rp is not None else float("nan"),
            rp[1] if rp is not None else float("nan"),
            rp[2] if rp is not None else float("nan"),
            f"{td:.1f}°" if td is not None else "-",
        )
        return res

    if robot_armed:
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
                         name="lastmile-v2-ext-cal").start()

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
        summary = {
            "detector": getattr(args, "detector", "sam3"),
            "ref_image": args.ref_image,
            "gt_pose_file": getattr(args, "gt_pose", None),
            "terminal_error": terminal_error,
        }
        summary_path = os.path.join(out_dir, "results_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Results summary written to %s", summary_path)
        _plot_xyz_error(jsonl_path, out_dir)
        if terminal_error is not None:
            log.info("Final XYZ error: %.3f mm", terminal_error["error_xyz_mm"])
        elif gt_xyz is not None:
            log.warning("GT pose provided but TERMINAL was never reached — no error computed.")
        if robot_armed:
            if home_pos is not None and robot._arm is not None:
                try:
                    log.info("Returning to home position [%.1f, %.1f, %.1f] mm ...",
                             home_pos[0], home_pos[1], home_pos[2])
                    robot.enabled = False
                    robot._move_abs(home_pos, wait=True, speed=50.0)
                    log.info("Home position reached.")
                except Exception as e:
                    log.error("Return-to-home failed: %s", e)
            try:
                robot.stop()
            except Exception as e:
                log.error("robot.stop() failed: %s", e, exc_info=True)
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                log.warning("cv2.destroyAllWindows() failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main_ext(argv: list[str] | None = None) -> int:
    p = _build_argparser()
    p.add_argument(
        "--detector",
        choices=["sam3", "gdino", "gdino+sam2"],
        default="sam3",
        help="Detection backend to use (default: sam3)",
    )
    p.add_argument(
        "--gt-pose",
        metavar="PATH",
        default="gt/gt_pose_protein_bar.json",
        help="Path to ground-truth pose JSON (default: gt/gt_pose_protein_bar.json). "
             "XYZ error is computed at TERMINAL and written to results_summary.json.",
    )
    args = p.parse_args(argv)
    args.debug_dir = "artifacts" if args.debug else None
    args.debug_every = (1 if (args.input_image or args.input_dir
                              or args.input_video) else 30)

    logger = _setup_logger(name="lastmile_v2_ext", log_dir=args.log_dir)
    logging.getLogger("lastmile_v2").handlers = logger.handlers
    logging.getLogger("lastmile").handlers = logger.handlers

    log_path = getattr(logger.handlers[1], "baseFilename", None) \
        if len(logger.handlers) > 1 else None

    debug_dir = _init_debug_dir(args.debug, args, args.ref_image)

    log.info("=" * 70)
    log.info("servo_lastmile_v2_ext starting")
    log.info("  detector=%s  prompt=%s  ref=%s  hand-eye=%s",
             args.detector, args.prompt, args.ref_image, args.hand_eye)
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
            run_offline_v2_ext(args, debug_dir=debug_dir)
        else:
            run_live_v2_ext(args, debug_dir=debug_dir)
        return 0
    finally:
        _finalize_debug_dir(debug_dir, log_path)
        log.info("Done.")


if __name__ == "__main__":
    sys.exit(main_ext())
