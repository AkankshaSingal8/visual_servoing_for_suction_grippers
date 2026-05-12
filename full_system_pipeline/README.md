# full_system_pipeline

RAL paper implementation. All new code lives here; nothing in `foundation_model/` is modified.
Scripts borrow from `foundation_model/` via imports.

## Structure

```
full_system_pipeline/
├── detection_backends.py        # GDINO-only / GDINO+SAM2 / SAM3 runner factories
├── pipeline/
│   ├── robot_controller_ext.py  # RobotControllerExt — adds correct_tilt()
│   └── servo_lastmile_v2_ext.py # Extended pipeline with --detector flag + tilt correction
└── evaluation/
    ├── analyze_run.py           # Offline metrics + plots from a run's frames.jsonl
    ├── annotate_gt.py           # Click-to-annotate GT grasp point on a video frame
    ├── run_experiment.py        # Automated multi-trial experiment runner
    ├── aggregate_results.py     # Cross-run statistics and paper figures
    └── failure_analysis.py      # Failed trial categorization
```

## Phase 0 — Baseline Audit

Run the existing v2 pipeline on a recorded video (no changes to `foundation_model/`):
```bash
python foundation_model/servo_lastmile_v2.py \
  --input-video runs/v2_proteinbar_servo12/raw.mp4 \
  --ref-image assets/objects/protein_bar.jpeg \
  --output-dir runs/smoke_test \
  --no-overlays
```

## Phase 1 — Metrics & Analysis

Analyze any existing run:
```bash
python full_system_pipeline/evaluation/analyze_run.py runs/v2_proteinbar_servo12/
# outputs: runs/v2_proteinbar_servo12/analysis/{5 plots + summary.json}

# Multiple runs overlaid:
python full_system_pipeline/evaluation/analyze_run.py runs/v2_servo10/ runs/v2_servo11/ --out-dir plots/

# With image-space GT:
python full_system_pipeline/evaluation/analyze_run.py runs/v2_servo10/ --gt-centroid 660 430
```

Annotate a GT grasp point on a run's video:
```bash
python full_system_pipeline/evaluation/annotate_gt.py \
  --run-dir runs/v2_proteinbar_servo12/ \
  --ref-image assets/objects/protein_bar.jpeg
# saves: runs/v2_proteinbar_servo12/gt_centroid.json
```

## Phase 2 — Ablation Pipeline Variants

Run with a different detection backend (offline):
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --detector gdino+sam2 \
  --ref-image assets/objects/protein_bar.jpeg \
  --input-video runs/v2_proteinbar_servo12/raw.mp4 \
  --output-dir runs/ablation_gdino_sam2/

python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --detector gdino \
  --ref-image assets/objects/protein_bar.jpeg \
  --input-video runs/v2_proteinbar_servo12/raw.mp4 \
  --output-dir runs/ablation_gdino_only/
```

Available detectors: `sam3` (default), `gdino+sam2`, `gdino`.

## Phase 4 — Tilt-Correcting Controller (live only)

Dry-run to check what tilt corrections would be sent:
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --detector sam3 \
  --ref-image assets/objects/protein_bar.jpeg \
  --dry-run
```

Live with tilt correction enabled (xArm + ZED required):
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --detector sam3 \
  --ref-image assets/objects/protein_bar.jpeg \
  --hand-eye config/hand_eye.npy \
  --output-dir runs/live_tilt_corrected/
```

## Phase 6 — Systematic Experiments

Run a batch of trials for one object/condition/pipeline:
```bash
python full_system_pipeline/evaluation/run_experiment.py \
  --object protein_bar \
  --condition single \
  --pipeline sam3 \
  --trials 5 \
  --ref-image assets/objects/protein_bar.jpeg \
  --experiment-log experiments/experiment_log.csv \
  --no-robot   # or omit for live robot
```

## Phase 7 — Analysis & Paper Figures

Aggregate results across all trials:
```bash
python full_system_pipeline/evaluation/aggregate_results.py \
  --experiment-log experiments/experiment_log.csv \
  --runs-dir runs/ \
  --out-dir experiments/figures/
```

Failure mode analysis:
```bash
python full_system_pipeline/evaluation/failure_analysis.py \
  --experiment-log experiments/experiment_log.csv \
  --runs-dir runs/ \
  --out-dir experiments/failure_analysis/
```

## Checkpoint Verification

| Checkpoint | Command | Pass criterion |
|-----------|---------|----------------|
| 0 | `python foundation_model/servo_lastmile_v2.py --input-video ... --no-overlays` | No crash; frames.jsonl written |
| 1 | `python full_system_pipeline/evaluation/analyze_run.py runs/*/` | 6 plots + summary.json in analysis/ |
| 2 | Run all 3 `--detector` modes on same video | Different centroids in frames.jsonl |
| 3 | Check `c_A` in NEAR frames with `--hand-eye` | Reprojection error < 5px |
| 4 | `--dry-run` with tilt box | Tilt commands ≤ 3°/step |
| 5 | 10 manual trials with GT pose | Final error computable |
| 6a | 25 SAM3+v2 single-box trials | ≥ 70% success |
| 6b | 300 trials full matrix | experiment_log.csv complete |
| 7 | Aggregate analysis | SAM3+v2 ≥ 90%; ablation monotonic |
