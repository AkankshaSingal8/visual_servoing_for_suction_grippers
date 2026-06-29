"""
Tilt ablation: compare tilt_deg trajectories across two runs — one with
tilt correction ON and one with it OFF.

Usage:
    python full_system_pipeline/evaluation/tilt_ablation.py \\
        --run-dir-with-tilt    runs/v3_simple_tiltON \\
        --run-dir-without-tilt runs/v3_simple_tiltOFF \\
        --output-dir           /tmp/tilt_ablation
"""
import argparse
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_near_tilt(run_dir: pathlib.Path) -> list:
    frames = [json.loads(l) for l in open(run_dir / "frames.jsonl")]
    return [f["tilt_deg"] for f in frames
            if f.get("state") == "NEAR" and f.get("tilt_deg") is not None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir-with-tilt",    required=True)
    p.add_argument("--run-dir-without-tilt", required=True)
    p.add_argument("--output-dir",           required=True)
    args = p.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with_tilt    = _load_near_tilt(pathlib.Path(args.run_dir_with_tilt))
    without_tilt = _load_near_tilt(pathlib.Path(args.run_dir_without_tilt))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(without_tilt, label="No tilt correction", color="tab:red",  alpha=0.7)
    ax.plot(with_tilt,    label="With tilt correction", color="tab:blue", alpha=0.7)
    ax.axhline(3.0, color="gray", linestyle="--", label="3 deg threshold")
    ax.set_xlabel("NEAR frame index")
    ax.set_ylabel("Surface tilt (degrees)")
    ax.set_title("Tilt correction ablation")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out / "tilt_ablation.png"), dpi=150)
    plt.close(fig)

    summary = {
        "with_tilt_mean_deg":    (sum(with_tilt)    / len(with_tilt))    if with_tilt    else None,
        "without_tilt_mean_deg": (sum(without_tilt) / len(without_tilt)) if without_tilt else None,
        "with_tilt_n_frames":    len(with_tilt),
        "without_tilt_n_frames": len(without_tilt),
    }
    (out / "tilt_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print("Tilt ablation summary:", summary)


if __name__ == "__main__":
    main()
