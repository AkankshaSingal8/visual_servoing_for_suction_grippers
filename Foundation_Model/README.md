# SemVS Semantic Visual Servoing

Image-based visual servoing 

```bash
git clone --recurse-submodules https://github.com/uksangyoo/SemVS.git
cd SemVS
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
python test_vs.py
```

