# Project Status

## Current Layout

- `foundation_model/`: canonical foundation-model visual servoing code.
- `foundation_model/variants/`: preserved variants that differ from canonical scripts.
- `model_training/stereo/`: stereo learned offset prediction.
- `model_training/mono/`: monocular learned offset prediction.
- `assets/objects/`: source object images.
- `artifacts/dinov2_example/`: checked-in example outputs.
- `vendor/`: checked-in third-party binary artifacts.

## What Is Implemented

- DINOv2 reference matching plus SAM2 segmentation.
- GroundingDINO/OWLv2 detection prototypes.
- SAM2 tracking and top-box/stack heuristics.
- ZED/xArm live camera and robot hooks.
- IBVS-style centroid-to-Y/Z robot servoing.
- Stereo and monocular model training/evaluation code.
- Checkpoint loading in model evaluation scripts.
- Vision-only and dry-run modes in canonical `foundation_model/servo_pipeline.py`.
- Offline image, image-directory, and video evaluation with JSONL metrics and overlays.
- Example stereo and monocular training configs.
- Reference-mask handling that prefers the full package/box over printed product art.

## What Still Needs Work

- Pick one canonical runnable pipeline and keep variants secondary.
- Add learned-policy inference for trained checkpoints.
- Add tests for datasets, preprocessing, centroid/mask logic, and controller signs.
- Add run manifests for hardware experiments.
