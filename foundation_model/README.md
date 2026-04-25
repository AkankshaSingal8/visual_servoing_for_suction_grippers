# Foundation Model Visual Servoing

Foundation-model perception and visual servoing prototypes.

```bash
git clone --recurse-submodules https://github.com/AkankshaSingal8/visual_servoing_for_suction_grippers.git
cd visual_servoing_for_suction_grippers/foundation_model
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

```bash
python dinov2_match_segment.py \
  --scene zed_input.png \
  --reference input_image_transparent.png \
  --output ../artifacts/dinov2_example \
  --no-display
```
