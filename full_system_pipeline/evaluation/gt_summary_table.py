#!/usr/bin/env python3
"""
gt_summary_table.py — Collect all v2_* runs with gt_centroid.json and build a
summary table of GT centroid errors.

Usage:
    python full_system_pipeline/evaluation/gt_summary_table.py
    python full_system_pipeline/evaluation/gt_summary_table.py --runs-dir runs/ --out-csv runs/gt_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow importing from evaluation package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from full_system_pipeline.evaluation.analyze_run import compute_summary, load_frames

FIELDNAMES = [
    "run_dir",
    "gt_x",
    "gt_y",
    "n_frames",
    "final_centroid_err_px",
    "mean_centroid_err_at_terminal",
    "reached_terminal",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs",
                        help="Root directory containing run subdirectories (default: runs/)")
    parser.add_argument("--out-csv", default="runs/gt_summary.csv",
                        help="Output CSV path (default: runs/gt_summary.csv)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_csv = Path(args.out_csv)

    # Scan for v2_* run dirs that have both frames.jsonl and gt_centroid.json
    annotated_runs = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.glob("v2_*")):
            if (run_dir / "frames.jsonl").exists() and (run_dir / "gt_centroid.json").exists():
                annotated_runs.append(run_dir)

    # Always write CSV (with headers), even if empty
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()

        if not annotated_runs:
            print("No GT-annotated runs found — run annotate_gt.py first")
            print(f"(Empty CSV with headers written to: {out_csv})")
            return

        rows = []
        for run_dir in annotated_runs:
            # Load GT centroid
            with (run_dir / "gt_centroid.json").open() as fh_gt:
                gt_data = json.load(fh_gt)
            gt_centroid = (float(gt_data["x"]), float(gt_data["y"]))

            # Load frames
            frames = load_frames(str(run_dir))

            # compute_summary expects a list of frame-lists; use gt_centroid as ref
            summary = compute_summary(
                [frames],
                ref_centroid=gt_centroid,
                gt_centroid=gt_centroid,
            )

            reached_terminal = int(any(f.get("state") == "TERMINAL" for f in frames))

            row = {
                "run_dir": run_dir.name,
                "gt_x": gt_centroid[0],
                "gt_y": gt_centroid[1],
                "n_frames": summary["n_frames"],
                "final_centroid_err_px": (
                    round(summary["final_centroid_err_px"], 2)
                    if summary.get("final_centroid_err_px") is not None else ""
                ),
                "mean_centroid_err_at_terminal": (
                    round(summary["mean_centroid_err_at_terminal"], 2)
                    if summary.get("mean_centroid_err_at_terminal") is not None else ""
                ),
                "reached_terminal": reached_terminal,
            }
            rows.append(row)
            writer.writerow(row)

    print(f"GT summary written to: {out_csv}")
    print(f"\n{'Run':<35} {'N frames':>8} {'Final err (px)':>15} {'Mean err @ term':>16} {'Terminal':>9}")
    print("-" * 88)
    for r in rows:
        print(
            f"{r['run_dir']:<35} "
            f"{r['n_frames']:>8} "
            f"{str(r['final_centroid_err_px']):>15} "
            f"{str(r['mean_centroid_err_at_terminal']):>16} "
            f"{str(r['reached_terminal']):>9}"
        )


if __name__ == "__main__":
    main()
