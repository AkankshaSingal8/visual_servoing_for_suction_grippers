"""
run_experiment.py — Runs a batch of visual servoing trials for a given object/condition/pipeline
combination, collects per-trial metrics, and appends results to a shared experiment log CSV.

CLI:
    python run_experiment.py \\
        --object protein_bar \\
        --condition single \\
        --pipeline sam3 \\
        --trials 5 \\
        --ref-image assets/objects/protein_bar.jpeg \\
        --runs-dir runs/ \\
        --experiment-log experiments/experiment_log.csv \\
        [--dry-run] \\
        [--no-robot] \\
        [--hand-eye config/hand_eye.npy]
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


IMAGE_CENTER = (640.0, 360.0)

LOG_FIELDNAMES = [
    "trial_id", "timestamp", "object_name", "condition", "pipeline",
    "trial_num", "run_dir",
    "gt_centroid_x", "gt_centroid_y",
    "gt_pose_x", "gt_pose_y", "gt_pose_z",
    "final_pose_x", "final_pose_y", "final_pose_z",
    "lateral_err_mm", "depth_err_mm", "euclidean_err_mm", "angular_err_deg",
    "final_centroid_err_px", "success",
    "n_frames", "n_far_frames", "n_near_sam3_frames", "n_near_track_frames",
    "n_terminal_frames", "mean_tilt_terminal", "watchdog_alarms", "notes",
]


def _find_repo_root(start: Path) -> Path:
    candidate = start if start.is_dir() else start.parent
    for parent in [candidate] + list(candidate.parents):
        if (parent / "ral_paper_plan.md").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (ral_paper_plan.md not found).")


def _phase_label(frame: dict) -> str:
    state = frame.get("state", "FAR")
    if state != "NEAR":
        return state
    near_phase = frame.get("_near_phase") or ""
    return "NEAR-track" if near_phase == "track" else "NEAR-sam3"


def _centroid_err(centroid, ref=IMAGE_CENTER) -> float | None:
    if centroid is None:
        return None
    return math.hypot(centroid[0] - ref[0], centroid[1] - ref[1])


def _safe_mean(vals):
    valid = [v for v in vals if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _load_frames(run_dir: Path) -> list[dict]:
    path = run_dir / "frames.jsonl"
    if not path.exists():
        return []
    frames = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return frames


def _compute_frame_metrics(frames: list[dict]) -> dict:
    n_far = sum(1 for f in frames if _phase_label(f) == "FAR")
    n_near_sam3 = sum(1 for f in frames if _phase_label(f) == "NEAR-sam3")
    n_near_track = sum(1 for f in frames if _phase_label(f) == "NEAR-track")
    n_terminal = sum(1 for f in frames if _phase_label(f) == "TERMINAL")
    watchdog_alarms = sum(1 for f in frames if f.get("watchdog_alarm"))

    terminal_frames = [f for f in frames if _phase_label(f) == "TERMINAL"]
    mean_tilt_terminal = _safe_mean([f.get("tilt_deg") for f in terminal_frames])

    final_centroid_err_px = None
    if terminal_frames:
        last = terminal_frames[-1]
        final_centroid_err_px = _centroid_err(last.get("best_centroid"))
    elif frames:
        last = frames[-1]
        final_centroid_err_px = _centroid_err(last.get("best_centroid"))

    return {
        "n_frames": len(frames),
        "n_far_frames": n_far,
        "n_near_sam3_frames": n_near_sam3,
        "n_near_track_frames": n_near_track,
        "n_terminal_frames": n_terminal,
        "mean_tilt_terminal": mean_tilt_terminal,
        "watchdog_alarms": watchdog_alarms,
        "final_centroid_err_px": final_centroid_err_px,
    }


def _rotation_matrix(rx_deg, ry_deg, rz_deg):
    import math as m
    rx, ry, rz = map(m.radians, [rx_deg, ry_deg, rz_deg])
    cx, sx = m.cos(rx), m.sin(rx)
    cy, sy = m.cos(ry), m.sin(ry)
    cz, sz = m.cos(rz), m.sin(rz)
    Rx = [[1,0,0],[0,cx,-sx],[0,sx,cx]]
    Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]]
    Rz = [[cz,-sz,0],[sz,cz,0],[0,0,1]]
    def mm(A, B):
        n = len(A)
        return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return mm(mm(Rz, Ry), Rx)


def _angular_err_deg(pose1, pose2) -> float:
    R1 = _rotation_matrix(*pose1[3:6])
    R2 = _rotation_matrix(*pose2[3:6])
    Rt = [[R1[j][i] for j in range(3)] for i in range(3)]
    def mm(A, B):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    Rrel = mm(Rt, R2)
    trace = sum(Rrel[i][i] for i in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))


def _compute_pose_errors(gt_pose: list, final_pose: list) -> dict:
    lateral_err = math.hypot(final_pose[0] - gt_pose[0], final_pose[1] - gt_pose[1])
    depth_err = abs(final_pose[2] - gt_pose[2])
    euclidean_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(final_pose[:3], gt_pose[:3])))
    angular_err = 0.0
    if len(gt_pose) >= 6 and len(final_pose) >= 6:
        try:
            angular_err = _angular_err_deg(gt_pose, final_pose)
        except Exception:
            angular_err = 0.0
    return {
        "lateral_err_mm": lateral_err,
        "depth_err_mm": depth_err,
        "euclidean_err_mm": euclidean_err,
        "angular_err_deg": angular_err,
    }


def _determine_success(pose_errors: dict | None, frame_metrics: dict, terminal_frames: list[dict]) -> bool:
    if pose_errors is not None:
        tilt_deg_terminal = _safe_mean([f.get("tilt_deg") for f in terminal_frames]) or 0.0
        return (
            pose_errors["lateral_err_mm"] < 5.0
            and pose_errors["depth_err_mm"] < 8.0
            and tilt_deg_terminal < 3.0
        )
    centroid_err = frame_metrics.get("final_centroid_err_px")
    if centroid_err is not None:
        return centroid_err < 30.0
    return False


def _ensure_log_header(log_path: Path):
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDNAMES)
            writer.writeheader()


def _append_log_row(log_path: Path, row: dict):
    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)


def _run_pipeline(
    repo_root: Path,
    run_dir: Path,
    ref_image: Path,
    pipeline: str,
    dry_run: bool,
    no_robot: bool,
    hand_eye: Path | None,
) -> subprocess.CompletedProcess:
    script = repo_root / "full_system_pipeline" / "pipeline" / "servo_lastmile_v2_ext.py"
    cmd = [
        sys.executable, str(script),
        "--ref-image", str(ref_image),
        "--output-dir", str(run_dir),
        "--no-window",
        "--detector", pipeline,
        "--no-overlays",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if no_robot:
        cmd.append("--no-robot")
    if hand_eye is not None:
        cmd.extend(["--hand-eye", str(hand_eye)])

    log_file = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    return result


def _prompt_gt_pose(run_dir: Path) -> list | None:
    print("  [GT] Manually move EE to ground-truth grasp point, then press ENTER.")
    input()
    pose_str = input("  [GT] Enter pose as 'x y z rx ry rz' (mm, deg) or leave blank to skip: ").strip()
    if not pose_str:
        return None
    try:
        pose = [float(v) for v in pose_str.split()]
    except ValueError:
        print("  [GT] Invalid pose, skipping.")
        return None
    gt_pose_path = run_dir / "gt_pose.json"
    with gt_pose_path.open("w") as fh:
        json.dump({"gt_pose": pose}, fh)
    print(f"  [GT] Saved gt_pose.json: {pose}")
    print("  [GT] Return robot to home and press ENTER.")
    input()
    return pose


def _record_final_pose(run_dir: Path, gt_pose: list) -> list | None:
    pose_str = input("  [FIN] Enter final EE pose as 'x y z rx ry rz' or blank to skip: ").strip()
    if not pose_str:
        return None
    try:
        final_pose = [float(v) for v in pose_str.split()]
    except ValueError:
        print("  [FIN] Invalid pose, skipping.")
        return None
    gt_pose_path = run_dir / "gt_pose.json"
    data = {"gt_pose": gt_pose, "final_pose": final_pose}
    with gt_pose_path.open("w") as fh:
        json.dump(data, fh)
    return final_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--pipeline", required=True, choices=["sam3", "gdino", "gdino+sam2"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--ref-image", required=True, type=Path)
    parser.add_argument("--runs-dir", default="runs/", type=Path)
    parser.add_argument("--experiment-log", default="experiments/experiment_log.csv", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--hand-eye", type=Path, default=None)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    print(f"Repo root: {repo_root}")

    ref_image = args.ref_image if args.ref_image.is_absolute() else repo_root / args.ref_image
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else repo_root / args.runs_dir
    log_path = args.experiment_log if args.experiment_log.is_absolute() else repo_root / args.experiment_log
    hand_eye = args.hand_eye
    if hand_eye is not None and not hand_eye.is_absolute():
        hand_eye = repo_root / hand_eye

    _ensure_log_header(log_path)

    all_successes = []
    all_lateral_errs = []
    all_depth_errs = []
    all_centroid_errs = []

    for trial_num in range(1, args.trials + 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_id = f"{args.object}_{args.condition}_{args.pipeline}_trial{trial_num}_{ts}"
        run_dir = runs_dir / trial_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Trial {trial_num}/{args.trials}: {trial_id} ===")

        gt_pose = None
        final_pose = None

        if not args.dry_run and not args.no_robot:
            print("  Position robot at home, then press ENTER.")
            input()
            gt_pose = _prompt_gt_pose(run_dir)

        print(f"  Launching pipeline ({args.pipeline})...")
        result = _run_pipeline(
            repo_root=repo_root,
            run_dir=run_dir,
            ref_image=ref_image,
            pipeline=args.pipeline,
            dry_run=args.dry_run,
            no_robot=args.no_robot,
            hand_eye=hand_eye,
        )
        print(f"  Pipeline exited with code {result.returncode}. Log: {run_dir / 'run.log'}")

        frames = _load_frames(run_dir)
        frame_metrics = _compute_frame_metrics(frames)
        terminal_frames = [f for f in frames if _phase_label(f) == "TERMINAL"]

        gt_pose_path = run_dir / "gt_pose.json"
        pose_errors = None
        gt_pose_data = None
        final_pose_data = None

        if gt_pose_path.exists():
            try:
                with gt_pose_path.open() as fh:
                    pose_data = json.load(fh)
                gt_pose_data = pose_data.get("gt_pose")
                final_pose_data = pose_data.get("final_pose")
                if gt_pose_data and final_pose_data:
                    pose_errors = _compute_pose_errors(gt_pose_data, final_pose_data)
            except Exception as exc:
                print(f"  Warning: could not parse gt_pose.json: {exc}")

        success = _determine_success(pose_errors, frame_metrics, terminal_frames)
        all_successes.append(success)

        if frame_metrics.get("final_centroid_err_px") is not None:
            all_centroid_errs.append(frame_metrics["final_centroid_err_px"])
        if pose_errors:
            all_lateral_errs.append(pose_errors["lateral_err_mm"])
            all_depth_errs.append(pose_errors["depth_err_mm"])

        row = {
            "trial_id": trial_id,
            "timestamp": ts,
            "object_name": args.object,
            "condition": args.condition,
            "pipeline": args.pipeline,
            "trial_num": trial_num,
            "run_dir": str(run_dir),
            "gt_centroid_x": None,
            "gt_centroid_y": None,
            "gt_pose_x": gt_pose_data[0] if gt_pose_data else None,
            "gt_pose_y": gt_pose_data[1] if gt_pose_data else None,
            "gt_pose_z": gt_pose_data[2] if gt_pose_data else None,
            "final_pose_x": final_pose_data[0] if final_pose_data else None,
            "final_pose_y": final_pose_data[1] if final_pose_data else None,
            "final_pose_z": final_pose_data[2] if final_pose_data else None,
            "lateral_err_mm": pose_errors["lateral_err_mm"] if pose_errors else None,
            "depth_err_mm": pose_errors["depth_err_mm"] if pose_errors else None,
            "euclidean_err_mm": pose_errors["euclidean_err_mm"] if pose_errors else None,
            "angular_err_deg": pose_errors["angular_err_deg"] if pose_errors else None,
            "final_centroid_err_px": frame_metrics["final_centroid_err_px"],
            "success": int(success),
            "n_frames": frame_metrics["n_frames"],
            "n_far_frames": frame_metrics["n_far_frames"],
            "n_near_sam3_frames": frame_metrics["n_near_sam3_frames"],
            "n_near_track_frames": frame_metrics["n_near_track_frames"],
            "n_terminal_frames": frame_metrics["n_terminal_frames"],
            "mean_tilt_terminal": frame_metrics["mean_tilt_terminal"],
            "watchdog_alarms": frame_metrics["watchdog_alarms"],
            "notes": "",
        }
        _append_log_row(log_path, row)

        print(f"  success={success}  n_frames={frame_metrics['n_frames']}  "
              f"n_terminal={frame_metrics['n_terminal_frames']}  "
              f"centroid_err={frame_metrics['final_centroid_err_px']:.1f}px"
              if frame_metrics["final_centroid_err_px"] is not None
              else f"  success={success}  n_frames={frame_metrics['n_frames']}  "
                   f"n_terminal={frame_metrics['n_terminal_frames']}  centroid_err=N/A")
        if pose_errors:
            print(f"  lateral={pose_errors['lateral_err_mm']:.2f}mm  "
                  f"depth={pose_errors['depth_err_mm']:.2f}mm  "
                  f"euclidean={pose_errors['euclidean_err_mm']:.2f}mm")

    n_success = sum(all_successes)
    print(f"\n=== Batch Summary ===")
    print(f"  Trials: {args.trials}  Successes: {n_success}  "
          f"Rate: {100*n_success/args.trials:.1f}%")
    if all_centroid_errs:
        mean_ce = sum(all_centroid_errs) / len(all_centroid_errs)
        print(f"  Mean centroid err: {mean_ce:.1f}px")
    if all_lateral_errs:
        print(f"  Mean lateral err: {sum(all_lateral_errs)/len(all_lateral_errs):.2f}mm")
    if all_depth_errs:
        print(f"  Mean depth err: {sum(all_depth_errs)/len(all_depth_errs):.2f}mm")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
