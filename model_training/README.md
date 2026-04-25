# Model Training

This directory contains the learned last-mile servoing models.

## Layout

- `stereo/`: dual-image models that consume left and right camera frames.
- `mono/`: single-image models that consume the left camera frame.
- `configs/`: editable example configs for training runs.

## Dataset Format

Both trainers expect an Isaac Sim-style dataset root with:

```text
dataset_root/
  index/
    index0.json
  render/
    render0/
      gripper_left_rgb/rgb_0000.png
      gripper_right_rgb/rgb_0000.png
```

Each index JSON must contain `key_frames`; every frame must include:

- `index`: frame index used to locate `rgb_####.png`
- `offset`: target servo offset, currently `[x, y, z]`

The monocular trainer uses `gripper_left_rgb`; the stereo trainer uses both left
and right images.

## Training

From the repository root:

```bash
python model_training/stereo/train.py \
  --config-path model_training/configs/stereo_example.yaml

python model_training/mono/train.py \
  --config-path model_training/configs/mono_example.yaml
```

Set `data.dataset_path` before running. Checkpoints are written under
`checkpoints/` by default.

## Evaluation

```bash
python model_training/stereo/eval.py \
  --config-path model_training/configs/stereo_example.yaml \
  --checkpoint-path checkpoints/stereo_example.yaml/0.pth

python model_training/mono/eval.py \
  --config-path model_training/configs/mono_example.yaml \
  --checkpoint-path checkpoints/mono_example.yaml/0.pth
```
