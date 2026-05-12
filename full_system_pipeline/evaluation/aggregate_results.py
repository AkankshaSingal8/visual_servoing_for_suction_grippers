"""
aggregate_results.py — Aggregates experiment results from the log CSV and generates summary
tables, bar charts, box plots, and distance-vs-error figures for all pipeline/condition
combinations.

CLI:
    python aggregate_results.py \\
        --experiment-log experiments/experiment_log.csv \\
        --out-dir experiments/figures/ \\
        [--runs-dir runs/]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE_COLORS = {
    "sam3": "mediumseagreen",
    "gdino+sam2": "steelblue",
    "gdino": "darkorange",
}

PIPELINE_ORDER = ["sam3", "gdino+sam2", "gdino"]


def _find_repo_root(start: Path) -> Path:
    candidate = start if start.is_dir() else start.parent
    for parent in [candidate] + list(candidate.parents):
        if (parent / "ral_paper_plan.md").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (ral_paper_plan.md not found).")


def _load_log(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    for col in ["lateral_err_mm", "depth_err_mm", "euclidean_err_mm",
                 "angular_err_deg", "final_centroid_err_px", "mean_tilt_terminal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "success" in df.columns:
        df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    return df


def _make_success_rate_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grp = df.groupby(["pipeline", "condition"])["success"].agg(["sum", "count"]).reset_index()
    grp["success_rate_pct"] = 100.0 * grp["sum"] / grp["count"]
    pivot = grp.pivot(index="pipeline", columns="condition", values="success_rate_pct").round(1)
    out_path = out_dir / "table_success_rate.csv"
    pivot.to_csv(out_path)
    print("\n=== Success Rate (%) ===")
    print(pivot.to_string())
    print(f"Saved: {out_path}")
    return grp


def _make_errors_table(df: pd.DataFrame, out_dir: Path):
    metrics = ["lateral_err_mm", "depth_err_mm", "euclidean_err_mm", "final_centroid_err_px"]
    rows = []
    for pipeline in df["pipeline"].unique():
        sub = df[df["pipeline"] == pipeline]
        row = {"pipeline": pipeline}
        for m in metrics:
            if m not in sub.columns:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
                continue
            vals = sub[m].dropna()
            row[f"{m}_mean"] = round(vals.mean(), 2) if len(vals) > 0 else None
            row[f"{m}_std"] = round(vals.std(), 2) if len(vals) > 1 else None
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_path = out_dir / "table_errors.csv"
    out_df.to_csv(out_path, index=False)
    print("\n=== Error Statistics ===")
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


def _fig_success_rate(df: pd.DataFrame, out_dir: Path):
    conditions = sorted(df["condition"].unique())
    pipelines = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]
    n_cond = len(conditions)
    n_pipe = len(pipelines)
    width = 0.8 / n_pipe
    x = np.arange(n_cond)

    fig, ax = plt.subplots(figsize=(max(6, n_cond * 2), 5))
    for i, pipeline in enumerate(pipelines):
        rates = []
        for cond in conditions:
            sub = df[(df["pipeline"] == pipeline) & (df["condition"] == cond)]
            if len(sub) == 0:
                rates.append(0.0)
            else:
                rates.append(100.0 * sub["success"].sum() / len(sub))
        offset = (i - n_pipe / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width=width * 0.9,
                      color=PIPELINE_COLORS.get(pipeline, "gray"),
                      label=pipeline, zorder=3)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Success Rate by Pipeline and Condition")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    out_path = out_dir / "fig_success_rate.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _fig_error_distribution(df: pd.DataFrame, out_dir: Path):
    conditions = sorted(df["condition"].unique())
    pipelines = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]
    n_cond = len(conditions)

    fig, axes = plt.subplots(1, n_cond, figsize=(max(6, n_cond * 4), 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        data_per_pipe = []
        labels = []
        colors = []
        for pipeline in pipelines:
            sub = df[(df["pipeline"] == pipeline) & (df["condition"] == cond)]
            vals = sub["final_centroid_err_px"].dropna().tolist()
            data_per_pipe.append(vals)
            labels.append(pipeline)
            colors.append(PIPELINE_COLORS.get(pipeline, "gray"))
        tick_labels_kwarg = "tick_labels" if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 9) else "labels"
        bps = ax.boxplot(data_per_pipe, patch_artist=True, notch=False,
                         **{tick_labels_kwarg: labels})
        for patch, color in zip(bps["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"condition: {cond}")
        ax.set_ylabel("final_centroid_err_px" if ax is axes[0] else "")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Centroid Error Distribution per Pipeline", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "fig_error_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_run_frames(run_dir: Path) -> list[dict]:
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


def _compute_centroid_err(frame) -> float | None:
    bc = frame.get("best_centroid")
    if bc is None:
        return None
    return math.hypot(bc[0] - 640.0, bc[1] - 360.0)


def _bin_by_z(frames: list[dict], bin_size: float = 50.0) -> dict[float, list]:
    bins: dict[float, list] = {}
    for frame in frames:
        z = frame.get("z_mm")
        if z is None:
            continue
        key = round(math.floor(z / bin_size) * bin_size + bin_size / 2, 1)
        bins.setdefault(key, []).append(frame)
    return bins


def _fig_metric_over_distance(df: pd.DataFrame, runs_dir: Path, out_dir: Path,
                               metric_fn, ylabel: str, filename: str):
    pipelines = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]
    fig, ax = plt.subplots(figsize=(10, 5))

    for pipeline in pipelines:
        sub = df[df["pipeline"] == pipeline]
        all_bin_vals: dict[float, list] = {}
        for _, row in sub.iterrows():
            run_dir = Path(str(row.get("run_dir", "")))
            if not run_dir.exists():
                continue
            frames = _load_run_frames(run_dir)
            binned = _bin_by_z(frames)
            for z_bin, bin_frames in binned.items():
                vals = [metric_fn(f) for f in bin_frames]
                vals = [v for v in vals if v is not None]
                all_bin_vals.setdefault(z_bin, []).extend(vals)

        if not all_bin_vals:
            continue

        z_sorted = sorted(all_bin_vals.keys())
        means = [np.mean(all_bin_vals[z]) for z in z_sorted]
        stds = [np.std(all_bin_vals[z]) for z in z_sorted]
        color = PIPELINE_COLORS.get(pipeline, "gray")
        ax.plot(z_sorted, means, color=color, label=pipeline, linewidth=2)
        ax.fill_between(
            z_sorted,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color, alpha=0.2,
        )

    ax.invert_xaxis()
    ax.set_xlabel("z_mm (approach: right → left)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Distance (binned 50mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _write_ablation_summary(df: pd.DataFrame, out_dir: Path):
    pipelines = [p for p in PIPELINE_ORDER if p in df["pipeline"].unique()]
    conditions = sorted(df["condition"].unique())
    metrics = ["lateral_err_mm", "depth_err_mm", "euclidean_err_mm", "final_centroid_err_px"]
    metric_labels = {
        "lateral_err_mm": "Lat. Err (mm)",
        "depth_err_mm": "Depth Err (mm)",
        "euclidean_err_mm": "Eucl. Err (mm)",
        "final_centroid_err_px": "Centroid Err (px)",
    }

    lines = []
    lines.append("Ablation Study Summary")
    lines.append("=" * 80)
    lines.append("")

    for cond in conditions:
        lines.append(f"Condition: {cond}")
        col_headers = ["Pipeline", "Success%"] + [metric_labels[m] for m in metrics]
        lines.append("  " + " | ".join(f"{h:>18}" for h in col_headers))
        lines.append("  " + "-" * (22 * len(col_headers)))
        for pipeline in pipelines:
            sub = df[(df["pipeline"] == pipeline) & (df["condition"] == cond)]
            if len(sub) == 0:
                continue
            succ_rate = 100.0 * sub["success"].sum() / len(sub)
            row_vals = [f"{pipeline:>18}", f"{succ_rate:>17.1f}%"]
            for m in metrics:
                vals = sub[m].dropna()
                if len(vals) == 0:
                    row_vals.append(f"{'N/A':>18}")
                else:
                    row_vals.append(f"{vals.mean():>14.2f}±{vals.std():>4.2f}" if len(vals) > 1
                                    else f"{vals.mean():>14.2f}      ")
            lines.append("  " + " | ".join(row_vals))
        lines.append("")

    lines.append("")
    lines.append("LaTeX-ready table (success rate):")
    lines.append(r"\begin{tabular}{l" + "c" * len(conditions) + "}")
    lines.append(r"  \hline")
    header_cells = ["Pipeline"] + list(conditions)
    lines.append("  " + " & ".join(f"\\textbf{{{h}}}" for h in header_cells) + r" \\")
    lines.append(r"  \hline")
    for pipeline in pipelines:
        row_cells = [pipeline]
        for cond in conditions:
            sub = df[(df["pipeline"] == pipeline) & (df["condition"] == cond)]
            if len(sub) == 0:
                row_cells.append("--")
            else:
                rate = 100.0 * sub["success"].sum() / len(sub)
                row_cells.append(f"{rate:.0f}\\%")
        lines.append("  " + " & ".join(row_cells) + r" \\")
    lines.append(r"  \hline")
    lines.append(r"\end{tabular}")

    out_path = out_dir / "ablation_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-log", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    log_path = args.experiment_log if args.experiment_log.is_absolute() else repo_root / args.experiment_log
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = args.runs_dir
    if runs_dir is not None and not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir

    if not log_path.exists():
        print(f"Experiment log not found: {log_path}")
        return

    print(f"Loading experiment log: {log_path}")
    df = _load_log(log_path)
    print(f"  {len(df)} rows, {df['pipeline'].nunique()} pipelines, "
          f"{df['condition'].nunique()} conditions")

    _make_success_rate_table(df, out_dir)
    _make_errors_table(df, out_dir)
    _fig_success_rate(df, out_dir)
    _fig_error_distribution(df, out_dir)

    if runs_dir is not None:
        print("\nLoading run frames for distance plots (this may take a while)...")
        _fig_metric_over_distance(
            df, runs_dir, out_dir,
            metric_fn=_compute_centroid_err,
            ylabel="Centroid Error (px)",
            filename="fig_error_over_distance.png",
        )
        _fig_metric_over_distance(
            df, runs_dir, out_dir,
            metric_fn=lambda f: f.get("tilt_deg"),
            ylabel="Tilt (deg)",
            filename="fig_tilt_over_distance.png",
        )
    else:
        print("\nSkipping distance plots (--runs-dir not provided).")

    _write_ablation_summary(df, out_dir)
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
