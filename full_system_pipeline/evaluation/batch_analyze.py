#!/usr/bin/env python3
"""
batch_analyze.py — Run analyze_run.py on every v2_* run directory
and write a summary CSV.

Usage:
    python full_system_pipeline/evaluation/batch_analyze.py
    python full_system_pipeline/evaluation/batch_analyze.py --runs-dir runs/ --out-csv runs/batch_summary.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

# Allow both `python batch_analyze.py` and `python -m` style imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from full_system_pipeline.evaluation.analyze_run import (
    load_frames, compute_summary, plot_error_over_distance,
    plot_signal_centroids, plot_tilt_timeline, plot_phase_summary,
    plot_signal_dominance, IMAGE_CENTER,
)

FIELDNAMES = [
    "run_dir", "n_frames", "n_far", "n_near_sam3", "n_near_track", "n_terminal",
    "reached_terminal", "mean_z_at_terminal", "mean_tilt_at_terminal",
    "mean_centroid_err_at_terminal", "final_centroid_err_px", "watchdog_alarms_count",
]


def _find_runs(runs_dir: Path) -> list[Path]:
    return sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("v2_") and (d / "frames.jsonl").exists()
    )


def analyze_one(run_dir: Path, write_plots: bool = True) -> dict:
    frames = load_frames(str(run_dir))
    summary = compute_summary([frames], ref_centroid=IMAGE_CENTER)
    summary["run_dir"] = run_dir.name
    summary["reached_terminal"] = int(summary.get("n_terminal", 0) > 0)

    if write_plots:
        out_dir = run_dir / "analysis"
        out_dir.mkdir(exist_ok=True)
        plot_error_over_distance([frames], str(out_dir), IMAGE_CENTER)
        plot_signal_centroids([frames], str(out_dir))
        plot_tilt_timeline([frames], str(out_dir))
        plot_phase_summary([frames], str(out_dir))
        plot_signal_dominance([frames], str(out_dir))
        with open(out_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory containing v2_* run subdirs (default: runs/)")
    parser.add_argument("--out-csv", default="runs/batch_summary.csv",
                        help="Output CSV path (default: runs/batch_summary.csv)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip per-run plot generation (faster)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = _find_runs(runs_dir)
    if not runs:
        print(f"No runs with frames.jsonl found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} runs in {runs_dir}")
    rows = []
    for run_dir in runs:
        try:
            summary = analyze_one(run_dir, write_plots=not args.no_plots)
            rows.append({k: summary.get(k) for k in FIELDNAMES})
            status = "TERMINAL" if summary.get("reached_terminal") else "no-term"
            print(f"  {run_dir.name}: {summary['n_frames']} frames  [{status}]  "
                  f"watchdog={summary.get('watchdog_alarms_count', 0)}")
        except Exception as e:
            print(f"  {run_dir.name}: FAILED ({e})", file=sys.stderr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBatch summary written to: {out_csv}")
    reached = sum(1 for r in rows if r.get("reached_terminal"))
    print(f"Reached TERMINAL: {reached}/{len(rows)} runs")


if __name__ == "__main__":
    main()
