"""
Live and offline runner for the simplified SAM3-only visual servoing pipeline.

Wires tilt correction (robot.correct_tilt) for the first time — enabled by default.
Use --no-tilt-correct to disable for ablation runs.

Usage (offline / no robot):
    python full_system_pipeline/pipeline/run_simple.py \\
        --no-robot \\
        --input-video runs/v2_proteinbar_servo12/raw.mp4 \\
        --ref-image assets/objects/protein_bar.jpeg \\
        --output-dir /tmp/simple_test

Usage (live, tilt correction on):
    python full_system_pipeline/pipeline/run_simple.py \\
        --ref-image assets/objects/protein_bar.jpeg \\
        --output-dir runs/v3_simple_tiltON
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

from foundation_model.servo_lastmile import (
    _setup_logger,
    make_default_sam3_runner,
)
from foundation_model.servo_lastmile_v2 import (
    VideoRecorder,
    _serialize_result_v2,
)
from foundation_model.servo_lastmile_simple import (
    LastMilePipelineSimple,
    _make_overlay_simple,
)
from full_system_pipeline.pipeline.servo_lastmile_v2_ext import (
    _load_gt_pose,
    _compute_xyz_error,
    _plot_xyz_error,
)

log = _setup_logger("run_simple")


# ---------------------------------------------------------------------------
# Argparser
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simplified SAM3 visual servo runner (SAM3 + go straight + tilt)"
    )
    p.add_argument("--ref-image", required=True,
                   help="Reference image of the target object")
    p.add_argument("--prompt", default="box",
                   help="GDINO text prompt (default: box)")

    # Robot
    p.add_argument("--no-robot", action="store_true",
                   help="Disable all xArm commands")
    p.add_argument("--robot-ip", default="192.168.1.223")

    # Tilt correction
    p.add_argument("--tilt-correct", dest="tilt_correct",
                   action="store_true", default=True,
                   help="Enable wrist tilt correction in NEAR (default: on)")
    p.add_argument("--no-tilt-correct", dest="tilt_correct",
                   action="store_false",
                   help="Disable tilt correction (ablation)")

    # Input source
    p.add_argument("--input-video",
                   help=".mp4 path for offline replay")

    # Output
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--no-window",   action="store_true")
    p.add_argument("--max-frames",  type=int, default=None)

    # Optional ground-truth pose for error logging
    p.add_argument("--gt-pose", default=None,
                   help="Path to GT pose JSON (for error metrics)")

    # Optional hand-eye calibration
    p.add_argument("--hand-eye", default=None,
                   help="Path to hand-eye calibration .npy")

    return p


# ---------------------------------------------------------------------------
# Per-frame processing (shared by offline and live loops)
# ---------------------------------------------------------------------------

def _process_frame(
    pipeline: LastMilePipelineSimple,
    frame_bgr: np.ndarray,
    robot,
    args: argparse.Namespace,
    gt_xyz,
    jsonl_file,
    recorder,
    out_dir,
    frame_idx: int,
) -> dict:
    res = pipeline.step(frame_bgr)
    state = res.get("state")
    best  = res.get("best_centroid")
    robot_armed = robot is not None and not args.no_robot

    # XY servo (FAR + NEAR)
    if robot_armed and state in ("FAR", "NEAR") and best is not None:
        robot.servo_step(best, frame_bgr.shape)

    # Tilt correction (NEAR only; correct_tilt internally gates on dead-zone / min_inliers)
    if (args.tilt_correct
            and robot_armed
            and state == "NEAR"
            and res.get("roll_deg") is not None
            and res.get("near_plane_n_inliers") is not None):
        applied = robot.correct_tilt(
            res["roll_deg"], res["pitch_deg"], res["near_plane_n_inliers"]
        )
        res["tilt_correction_applied"] = applied
    else:
        res["tilt_correction_applied"] = False

    # Optional GT error logging (live robot only)
    if gt_xyz is not None and robot_armed:
        robot_pos = robot.get_position()
        res["robot_pos"] = robot_pos
        res.update(_compute_xyz_error(robot_pos, gt_xyz))

    # Serialize
    jsonl_file.write(json.dumps(_serialize_result_v2(res)) + "\n")
    jsonl_file.flush()

    # Overlay / recording
    if not args.no_overlays:
        overlay = _make_overlay_simple(frame_bgr, res)
        if recorder:
            recorder.write(overlay)
        if out_dir:
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.jpg"), overlay)

    log.info("[%4d] %-8s phase=%-4s z=%s tilt=%s hold=%s tilt_corr=%s",
             frame_idx,
             state or "?",
             res.get("phase") or "-",
             f"{res['z_mm']:.0f}"    if res.get("z_mm")    else "—",
             f"{res['tilt_deg']:.1f}deg" if res.get("tilt_deg") else "—",
             res.get("holding_last_centroid", False),
             res.get("tilt_correction_applied", False))
    return res


# ---------------------------------------------------------------------------
# Offline runner
# ---------------------------------------------------------------------------

def run_offline_simple_ext(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LastMilePipelineSimple(
        sam3_runner=make_default_sam3_runner(args.ref_image, prompt=args.prompt),
    )
    pipeline.reset()

    gt_xyz = _load_gt_pose(args.gt_pose) if args.gt_pose else None
    jsonl_path = (out_dir / "frames.jsonl") if out_dir else pathlib.Path("/dev/null")

    cap = cv2.VideoCapture(args.input_video)
    recorder = None
    if out_dir and not args.no_overlays:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        recorder = VideoRecorder(str(out_dir), fps=fps)

    frame_idx = 0
    with open(jsonl_path, "w") as fout:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break
            _process_frame(
                pipeline=pipeline, frame_bgr=frame, robot=None, args=args,
                gt_xyz=gt_xyz, jsonl_file=fout, recorder=recorder,
                out_dir=out_dir, frame_idx=frame_idx,
            )
            frame_idx += 1

    cap.release()
    if recorder:
        recorder.release()

    if gt_xyz is not None and out_dir:
        _plot_xyz_error(str(jsonl_path), str(out_dir))

    log.info("Offline run complete: %d frames → %s", frame_idx, jsonl_path)


# ---------------------------------------------------------------------------
# Live runners (OpenCV webcam fallback; ZED wiring deferred to robot session)
# ---------------------------------------------------------------------------

def _run_opencv_loop(pipeline, robot, args, gt_xyz, jsonl_path, out_dir):
    cap = cv2.VideoCapture(0)
    recorder = None
    if out_dir and not args.no_overlays:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        recorder = VideoRecorder(str(out_dir), fps=fps)

    frame_idx = 0
    with open(jsonl_path, "w") as fout:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Camera capture failed.")
                break
            res = _process_frame(
                pipeline=pipeline, frame_bgr=frame, robot=robot, args=args,
                gt_xyz=gt_xyz, jsonl_file=fout, recorder=recorder,
                out_dir=out_dir, frame_idx=frame_idx,
            )
            if not args.no_window:
                overlay = _make_overlay_simple(frame, res)
                cv2.imshow("simple_servo", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    pipeline.reset()
                    log.info("Pipeline reset.")
            if res.get("state") == "TERMINAL":
                log.info("TERMINAL reached — stopping.")
                break
            frame_idx += 1

    cap.release()
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()


def _run_zed_loop(pipeline, robot, args, gt_xyz, jsonl_path, out_dir):
    # ZED wiring is identical to run_live_v2_ext; deferred to robot session.
    log.info("ZED wiring deferred — falling back to OpenCV capture.")
    _run_opencv_loop(pipeline, robot, args, gt_xyz, jsonl_path, out_dir)


def run_live_simple(args: argparse.Namespace) -> None:
    from full_system_pipeline.pipeline.robot_controller_ext import RobotControllerExt
    robot = RobotControllerExt(ip=args.robot_ip) if not args.no_robot else None

    pipeline = LastMilePipelineSimple(
        sam3_runner=make_default_sam3_runner(args.ref_image, prompt=args.prompt),
    )
    pipeline.reset()

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    gt_xyz = _load_gt_pose(args.gt_pose) if args.gt_pose else None
    jsonl_path = (out_dir / "frames.jsonl") if out_dir else pathlib.Path("/dev/null")

    try:
        import pyzed.sl as sl
        _run_zed_loop(pipeline, robot, args, gt_xyz, jsonl_path, out_dir)
    except ImportError:
        log.warning("pyzed not available — using OpenCV webcam capture.")
        _run_opencv_loop(pipeline, robot, args, gt_xyz, jsonl_path, out_dir)
    finally:
        if robot:
            robot.disconnect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.input_video:
        run_offline_simple_ext(args)
    else:
        run_live_simple(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
