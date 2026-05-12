"""
failure_analysis.py — Loads all failed trials from the experiment log, classifies each
failure into one or more diagnostic categories based on frame-level signals, and writes
a structured failure report and summary.

CLI:
    python failure_analysis.py \\
        --experiment-log experiments/experiment_log.csv \\
        --runs-dir runs/ \\
        --out-dir experiments/failure_analysis/
"""

import argparse
import collections
import csv
import json
import math
import shutil
from pathlib import Path


FAILURE_CATEGORIES = [
    "cotracker_drift",
    "false_detection",
    "no_terminal",
    "tilt_overshoot",
    "depth_noise",
    "watchdog_fired",
    "unknown",
]

REPORT_FIELDNAMES = [
    "trial_id", "object_name", "condition", "pipeline",
    "run_dir", "failure_categories", "notes",
]


def _find_repo_root(start: Path) -> Path:
    candidate = start if start.is_dir() else start.parent
    for parent in [candidate] + list(candidate.parents):
        if (parent / "ral_paper_plan.md").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (ral_paper_plan.md not found).")


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


def _phase_label(frame: dict) -> str:
    state = frame.get("state", "FAR")
    if state != "NEAR":
        return state
    near_phase = frame.get("_near_phase") or ""
    return "NEAR-track" if near_phase == "track" else "NEAR-sam3"


def _pt_dist(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _classify_failures(frames: list[dict]) -> list[str]:
    categories = []

    near_track_frames = [f for f in frames if _phase_label(f) == "NEAR-track"]
    for i in range(1, len(near_track_frames)):
        prev = near_track_frames[i - 1]
        curr = near_track_frames[i]
        if _pt_dist(curr.get("c_B"), prev.get("c_B")) > 50.0:
            categories.append("cotracker_drift")
            break

    far_frames = [f for f in frames if _phase_label(f) == "FAR"]
    if far_frames:
        low_score_count = sum(
            1 for f in far_frames
            if f.get("sam_score") is not None and f["sam_score"] < 0.5
        )
        if low_score_count / len(far_frames) > 0.30:
            categories.append("false_detection")

    terminal_frames = [f for f in frames if _phase_label(f) == "TERMINAL"]
    if not terminal_frames:
        categories.append("no_terminal")

    near_frames = [f for f in frames if _phase_label(f) in ("NEAR-sam3", "NEAR-track")]
    for f in near_frames:
        tilt = f.get("tilt_deg")
        if tilt is not None and abs(tilt) > 20.0:
            categories.append("tilt_overshoot")
            break

    window_size = 10
    if len(near_frames) >= window_size:
        for i in range(len(near_frames) - window_size + 1):
            window = near_frames[i:i + window_size]
            z_vals = [f.get("z_mm") for f in window if f.get("z_mm") is not None]
            if len(z_vals) >= 2:
                mean_z = sum(z_vals) / len(z_vals)
                var_z = sum((v - mean_z) ** 2 for v in z_vals) / len(z_vals)
                if var_z > 5000.0:
                    categories.append("depth_noise")
                    break

    watchdog_frames = [f.get("watchdog_alarm", False) for f in frames]
    max_consecutive = 0
    current_run = 0
    for alarm in watchdog_frames:
        if alarm:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 0
    if max_consecutive > 3:
        categories.append("watchdog_fired")

    if not categories:
        categories.append("unknown")

    return categories


def _extract_frame_image(run_dir: Path, frames: list[dict], out_path: Path):
    try:
        import cv2
    except ImportError:
        print("    cv2 not available, skipping frame extraction.")
        return

    overlay_path = run_dir / "overlay.mp4"
    if not overlay_path.exists():
        overlay_path = run_dir / "raw.mp4"
    if not overlay_path.exists():
        print(f"    No video found in {run_dir}, skipping frame extraction.")
        return

    terminal_indices = [i for i, f in enumerate(frames) if _phase_label(f) == "TERMINAL"]
    target_frame_idx = terminal_indices[0] if terminal_indices else max(0, len(frames) - 1)

    cap = cv2.VideoCapture(str(overlay_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        print(f"    Saved frame: {out_path}")
    else:
        print(f"    Could not read frame {target_frame_idx} from {overlay_path}")


def _write_failure_report(out_dir: Path, rows: list[dict]):
    out_path = out_dir / "failure_report.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


def _write_failure_summary(out_dir: Path, rows: list[dict]):
    category_total: collections.Counter = collections.Counter()
    category_per_pipeline: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    category_per_condition: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for row in rows:
        cats = [c.strip() for c in row["failure_categories"].split(",") if c.strip()]
        pipeline = row["pipeline"]
        condition = row["condition"]
        for cat in cats:
            category_total[cat] += 1
            category_per_pipeline[pipeline][cat] += 1
            category_per_condition[condition][cat] += 1

    lines = []
    lines.append("Failure Analysis Summary")
    lines.append("=" * 60)
    lines.append(f"Total failed trials: {len(rows)}")
    lines.append("")

    lines.append("Failure Category Totals:")
    for cat in FAILURE_CATEGORIES:
        count = category_total.get(cat, 0)
        if count > 0:
            lines.append(f"  {cat:<24} {count}")
    lines.append("")

    lines.append("Failure Categories per Pipeline:")
    all_pipelines = sorted(category_per_pipeline.keys())
    for pipeline in all_pipelines:
        lines.append(f"  {pipeline}:")
        counter = category_per_pipeline[pipeline]
        for cat, count in counter.most_common():
            lines.append(f"    {cat:<22} {count}")
    lines.append("")

    lines.append("Most Common Failure Mode per Condition:")
    all_conditions = sorted(category_per_condition.keys())
    for cond in all_conditions:
        counter = category_per_condition[cond]
        if counter:
            most_common_cat, most_common_cnt = counter.most_common(1)[0]
            lines.append(f"  {cond:<20} {most_common_cat} ({most_common_cnt}x)")
    lines.append("")

    out_path = out_dir / "failure_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"Saved: {out_path}")

    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-log", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    log_path = args.experiment_log if args.experiment_log.is_absolute() else repo_root / args.experiment_log
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else repo_root / args.runs_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_out_dir = out_dir / "frames"

    if not log_path.exists():
        print(f"Experiment log not found: {log_path}")
        return

    failed_rows = []
    with log_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                success_val = row.get("success", "").strip()
                is_success = int(float(success_val)) if success_val else 0
            except (ValueError, TypeError):
                is_success = 0
            if not is_success:
                failed_rows.append(row)

    print(f"Found {len(failed_rows)} failed trial(s) in {log_path}")

    report_rows = []
    for row in failed_rows:
        trial_id = row.get("trial_id", "unknown")
        run_dir_str = row.get("run_dir", "")
        run_dir = Path(run_dir_str) if run_dir_str else None

        print(f"\nAnalyzing: {trial_id}")

        frames = []
        if run_dir and run_dir.exists():
            frames = _load_frames(run_dir)
            print(f"  Loaded {len(frames)} frames from {run_dir}")
        else:
            print(f"  Run dir not found: {run_dir_str}")

        categories = _classify_failures(frames)
        print(f"  Failure categories: {categories}")

        if run_dir and run_dir.exists() and frames:
            frame_out = frames_out_dir / f"{trial_id}.png"
            _extract_frame_image(run_dir, frames, frame_out)

        report_rows.append({
            "trial_id": trial_id,
            "object_name": row.get("object_name", ""),
            "condition": row.get("condition", ""),
            "pipeline": row.get("pipeline", ""),
            "run_dir": run_dir_str,
            "failure_categories": ", ".join(categories),
            "notes": row.get("notes", ""),
        })

    _write_failure_report(out_dir, report_rows)
    _write_failure_summary(out_dir, report_rows)
    print(f"\nFailure analysis complete. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
