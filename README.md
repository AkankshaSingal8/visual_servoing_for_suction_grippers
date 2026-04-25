# Visual Servoing for Suction Grippers

Research code for target-conditioned visual servoing with suction grippers.

For the research framing, current codebase map, and code-completion roadmap,
see [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Repository Layout

- `foundation_model/`: foundation-model perception and servoing prototypes
  using GroundingDINO/OWLv2, DINOv2, SAM2, Depth Anything V2, ZED, and xArm.
- `foundation_model/variants/`: preserved variants that differ from the canonical scripts.
- `model_training/stereo/`: stereo learned-offset training and evaluation.
- `model_training/mono/`: monocular learned-offset training and evaluation.
- `assets/objects/`: source object/reference images.
- `artifacts/dinov2_example/`: generated DINOv2/SAM2 example outputs.
- `vendor/`: checked-in third-party binary artifacts, if any.

## Foundation Model Setup

```bash
git clone --recurse-submodules https://github.com/uksangyoo/SemVS.git
cd SemVS
cd foundation_model
```

Install each third-party dependency per its own README:
- `third-party/GroundingDINO`
- `third-party/sam2`
- `third-party/Depth-Anything-V2`

Download model weights into:
- `third-party/GroundingDINO/weights/`
- `third-party/sam2/checkpoints/`
- `third-party/Depth-Anything-V2/checkpoints/`

## Usage

From the repository root:

```bash
python foundation_model/dinov2_match_segment.py \
  --scene foundation_model/zed_input.png \
  --reference foundation_model/input_image_transparent.png \
  --output artifacts/dinov2_example \
  --no-display
```

Run the canonical pipeline without robot hardware:

```bash
python foundation_model/servo_pipeline.py \
  --ref-image foundation_model/input_image_transparent.png \
  --ref-mask-mode auto \
  --input-image foundation_model/zed_input.png \
  --output-dir runs/offline_smoke
```

For tightly cropped package references, `auto` treats the whole reference image
as the target box. Use `--ref-mask-mode foreground` only when the reference
image has a real foreground/background separation and you want Otsu extraction.

Run live camera perception without robot motion:

```bash
python foundation_model/servo_pipeline.py \
  --ref-image foundation_model/input_image_transparent.png \
  --dry-run
```

Train learned last-mile models:

```bash
python model_training/stereo/train.py \
  --config-path model_training/configs/stereo_example.yaml

python model_training/mono/train.py \
  --config-path model_training/configs/mono_example.yaml
```
