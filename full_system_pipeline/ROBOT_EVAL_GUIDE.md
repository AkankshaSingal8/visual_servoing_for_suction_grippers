# Robot Evaluation Guide — xArm + ZED Mini

How to run live evaluation trials with the robot arm and ZED Mini stereo camera.

---

## Hardware Requirements

- **Robot**: xArm 6 (6-DOF), powered on, connected via Ethernet to the host machine.
  Default IP: `192.168.1.241` (set in `servo_pipeline_sam3.py:ROBOT_IP`).
- **Camera**: ZED Mini, mounted on the end effector in place of the gripper. Connected via USB 3.0.
- **Host machine**: Ubuntu with CUDA GPU (for SAM3/DINOv2 inference). Must be on the same subnet as the robot.

---

## Software Prerequisites

### 1. PyZED SDK
The pipeline uses PyZED for stereo depth. If PyZED is not installed it falls back to OpenCV (no depth, no LOCK trigger). Install from Stereolabs:
```bash
# Download the ZED SDK installer from https://www.stereolabs.com/developers/release/
# Then install the Python package:
pip install pyzed
# Or use the path the ZED SDK installer sets up:
python /usr/local/zed/get_python_api.py
```
Verify: `python -c "import pyzed.sl as sl; print(sl.__version__)"`.

### 2. xArm Python SDK
```bash
pip install xarm-python-sdk
```
Verify: `python -c "from xarm.wrapper import XArmAPI"`.

### 3. Model weights
All model weights must be present before running live:
```bash
# GroundingDINO (only needed for gdino / gdino+sam2 detector):
ls third-party/GroundingDINO/weights/

# SAM2 (required for the full pipeline):
ls third-party/sam2/checkpoints/

# Depth Anything V2 (optional, not used in live mode — ZED provides depth):
ls third-party/Depth-Anything-V2/checkpoints/
```

---

## Network Setup

1. Set the host machine's Ethernet interface to a static IP on the same subnet as the robot, e.g. `192.168.1.100/24`.
2. Ping the robot to confirm connectivity: `ping 192.168.1.241`.
3. Open uFactory Studio in a browser (`http://192.168.1.241`) and confirm the arm is in **Manual mode** before running. Switch to **PC mode** when ready to run the script (the script connects via SDK and takes control).

> **Safety**: always have your hand on the physical emergency stop button while the robot is moving autonomously.

---

## Physical Setup

1. Place the target box on the table in front of the robot, within the reachable workspace (roughly X: 200–600 mm forward from the robot base, Y: ±300 mm, Z: ±200 mm).
2. Make sure the ZED Mini cable is long enough to reach the full approach range without pulling.
3. Start the robot from its home position. The pipeline records the home position on connect and returns to it when the session ends.
4. If running with distractors, place them now.

---

## Running the Pipeline

### Dry-run (camera only, no robot motion)
Use this to verify detection and calibration before moving anything:
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --dry-run \
  --output-dir runs/dryrun_$(date +%Y%m%d_%H%M%S)/
```
The preview window shows the overlay. The robot does not move. Tilt correction commands are logged but not sent.

### Live with robot (SAM3 detector, default)
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --output-dir runs/live_$(date +%Y%m%d_%H%M%S)/
```

### Live with a different detector (ablation)
```bash
# GDINO + SAM2:
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --detector gdino+sam2 \
  --output-dir runs/ablation_gdino_sam2_$(date +%Y%m%d_%H%M%S)/

# GDINO only:
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --detector gdino \
  --output-dir runs/ablation_gdino_$(date +%Y%m%d_%H%M%S)/
```

### Live with hand-eye calibration (enables Signal A)
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --hand-eye config/hand_eye.npy \
  --output-dir runs/live_with_A_$(date +%Y%m%d_%H%M%S)/
```
Without `--hand-eye`, Signal A is disabled and fusion relies on B/C/D/E only.

### Headless (SSH, no display)
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --no-window \
  --output-dir runs/headless_$(date +%Y%m%d_%H%M%S)/
```

---

## What Happens on Startup

When the script launches with robot enabled:

1. **Robot connects** — `RobotController` opens a TCP connection to `192.168.1.241`. Home position is recorded immediately (`robot._get_pos()`).
2. **Models load** — SAM3 (or GDINO/SAM2 depending on `--detector`), DINOv2, CoTracker3, SAM2 load onto GPU. This takes 10–30 seconds on first run.
3. **ZED opens** — PyZED opens the camera at HD720 (1280×720), 30 fps, stereo depth mode `PERFORMANCE`. Intrinsics are pulled from the camera's internal calibration (not from hardcoded defaults).
4. **Calibration thread starts** — A background thread runs the Y/Z Jacobian calibration automatically (see below). The robot does not servo until calibration completes.

### Automatic Y/Z Jacobian Calibration

The control law needs to know how many pixels the target moves on screen per mm of robot Y/Z movement. This is measured automatically at startup:

1. Wait for the first live frame from the ZED (up to 30 seconds).
2. Record a frame before the move.
3. Move the robot +25 mm in Y, record a frame after, return to home.
4. Measure the optical flow between the two frames (Lucas-Kanade).
5. Repeat for Z axis.
6. Compute the 2×2 Jacobian `J_yz` (px/mm) and its pseudoinverse.
7. Set `robot.enabled = True` — only now does the servo loop start moving.

Log lines to watch:
```
=== Calibration: waiting for camera frame ===
  [Y] moving +25 mm at 80 mm/s...
  [Y] flow: (+12.3, -0.1) px / 25 mm  -> J[:,0]=[+0.4920, -0.0040]
  [Z] moving +25 mm at 80 mm/s...
  [Z] flow: (-0.2, +11.8) px / 25 mm  -> J[:,1]=[-0.0080, +0.4720]
=== Calibration complete [calibrated] ===
```

If flow is too small (< 1 px), that axis is ignored and the robot approaches in X only (no lateral correction). This usually means the robot is too far from the target — move closer before starting, or point the camera at a textured surface.

---

## State Machine During a Live Run

```
FAR  ──── box detected by SAM3, area 5–25% of frame, depth 150–400 mm ────►  NEAR
          (3 consecutive qualifying frames required to lock)

NEAR (sam3 phase)  ──── box fills frame, SAM3 is sole signal ────────────────────►
NEAR (track phase) ──── box exits frame, CoTracker3 + SAM3 anchor ──────────────►
                    (tilt correction fires here every frame, before XY servo step)

NEAR ────  depth < 30 mm  OR  robot X > 505 mm  ─────────────────────────────►  TERMINAL
```

In **TERMINAL** state: the robot stops lateral correction. The plane-fit surface normal is computed one last time and reported. The script does not command a final plunge — that is left to the operator.

---

## Control Law

Each servo step (runs at most once every 0.3 s while robot is moving):

1. **Tilt correction** (before XY): if `tilt_deg > 1°` and RANSAC inliers ≥ 500:
   - `delta_roll  = clamp(-0.6 × roll_deg,  ±3°)`
   - `delta_pitch = clamp(-0.6 × pitch_deg, ±3°)`
   - Sent as a relative wrist rotation: `arm.set_tool_position(relative=True)`.
2. **XY centering**: pixel error `(ex, ey)` from image center → `J_yz_inv @ [ex, ey]` → `(dy_mm, dz_mm)`, each clamped to ±12 mm/step.
3. **Approach**: constant +5 mm/step in robot X (toward box), every step regardless of centroid error.
4. **Dead-zone**: if centroid is within 8 px of image center, no lateral correction (small Z nudge only).
5. **Jump guard**: if centroid jumps > 80 px between steps, the step is ignored.

Safety limits:
- `MIN_Z_MM = -300 mm` (robot Z is clamped at this floor).
- `VS_SPEED = 150 mm/s` (maximum lateral speed).
- `MAX_YZ_STEP = 12 mm` per step.
- Watchdog alarm: if SAM2 centroid disagrees with fused estimate > 60 px for 5 consecutive frames, servo pauses.

---

## Window Controls

The preview window (disable with `--no-window`) shows the annotated overlay. Keyboard controls:

| Key | Action |
|-----|--------|
| `v` | Toggle servo on/off (robot motion) |
| `r` | Reset pipeline to FAR state |
| `q` or `Esc` | Quit — robot returns to home position |

---

## Stopping

- Press `q` in the window, or `Ctrl-C` in the terminal.
- On exit, the script automatically moves the robot back to the home position it recorded at startup (at 50 mm/s), then disconnects.
- If home return fails (e.g. joint limit), the arm is disabled and the script exits. Move the arm manually via uFactory Studio.

---

## Evaluating a Trial

### Step 1 — Run the trial
```bash
python full_system_pipeline/pipeline/servo_lastmile_v2_ext.py \
  --ref-image assets/objects/protein_bar.jpeg \
  --output-dir runs/trial_01/
```

### Step 2 — Capture ground truth (optional, for real-world error)
After the autonomous run ends and the robot is back at home, manually jog the EE to the correct grasp point (box face center) using uFactory Studio or jog mode. Then record the pose:
```bash
python -c "
from foundation_model.servo_pipeline_sam3 import RobotController
import json, time
r = RobotController('192.168.1.241')
r.connect()
time.sleep(1)
pos = r._get_pos()
print('GT pose:', pos)
json.dump({'gt_pose': pos}, open('runs/trial_01/gt_pose.json', 'w'))
r.stop()
"
```
Also record what the pipeline's final EE pose was (from the last TERMINAL frame's robot state) — the `run_experiment.py` automation does this for you.

### Step 3 — Annotate the grasp point in the image
```bash
python full_system_pipeline/evaluation/annotate_gt.py \
  --run-dir runs/trial_01/ \
  --ref-image assets/objects/protein_bar.jpeg
# Click the box face center on the displayed frame, press ENTER
# Saves: runs/trial_01/gt_centroid.json
```

### Step 4 — Analyse the run
```bash
python full_system_pipeline/evaluation/analyze_run.py \
  runs/trial_01/ \
  --gt-centroid $(python -c "import json; c=json.load(open('runs/trial_01/gt_centroid.json'))['gt_centroid']; print(c[0], c[1])")
```
Outputs go to `runs/trial_01/analysis/`:
- `error_over_distance.png` — centroid error vs depth (does it decrease on approach?)
- `tilt_timeline.png` — roll/pitch/tilt vs depth
- `signal_centroids.png` — per-signal X/Y vs frame
- `summary.json` — final error, mean tilt at terminal, success flag

---

## Running a Full Experiment Batch

Use `run_experiment.py` to automate multi-trial collection:
```bash
python full_system_pipeline/evaluation/run_experiment.py \
  --object protein_bar \
  --condition single \
  --pipeline sam3 \
  --trials 5 \
  --ref-image assets/objects/protein_bar.jpeg \
  --experiment-log experiments/experiment_log.csv
```

The script prompts you between trials to reposition the box and capture GT pose. After all trials complete, run aggregate analysis:
```bash
python full_system_pipeline/evaluation/aggregate_results.py \
  --experiment-log experiments/experiment_log.csv \
  --runs-dir runs/ \
  --out-dir experiments/figures/
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ZED open failed` | USB not connected or ZED SDK not installed | Check USB 3.0 connection; reinstall ZED SDK |
| Falling back to OpenCV camera | PyZED import fails | `pip install pyzed` or run `/usr/local/zed/get_python_api.py` |
| `Calibration skipped: no camera frame` | ZED opens but no frames arrive | Check `cam_index` arg (default 0); try `--cam-index 1` |
| Flow too small — axis ignored | Robot too far from box during calibration, or featureless background | Move robot closer; point at a textured surface |
| `J_yz rank=0 — approach-only mode` | Both Y and Z calibration axes failed | Pipeline still approaches in X; no lateral correction |
| Robot doesn't move after calibration | `robot.enabled` stays False | Check log for calibration errors; try `--no-robot` to verify pipeline runs |
| Centroid jump ignored | Large detection discontinuity | Normal for state transitions; watchdog will fire if persistent |
| Watchdog alarm (red border) | SAM2 centroid disagrees with fused estimate for 5+ frames | Pipeline pauses servo. Press `r` to reset, reposition box |
| `Connection refused` on robot connect | Robot not in PC mode or wrong IP | Open uFactory Studio, switch to PC mode; verify IP |
| `ModuleNotFoundError: xarm` | xArm SDK not installed | `pip install xarm-python-sdk` |
| TERMINAL fires too early (robot X > 505 mm) | Position-based fallback triggered | Box may be too far away; check starting position and `ROBOT_IP` |

---

## Key Constants (for tuning)

All in `foundation_model/servo_pipeline_sam3.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `ROBOT_IP` | `192.168.1.241` | xArm network address |
| `VS_GAIN` | `0.08` | Overall servo proportional gain |
| `VS_APPROACH` | `5.0 mm/step` | Fixed X approach per step |
| `VS_DEAD_ZONE` | `8 px` | No lateral correction inside this radius |
| `VS_SPEED` | `150 mm/s` | Robot movement speed |
| `MAX_YZ_STEP` | `12 mm` | Max lateral correction per step |
| `VS_RATE` | `0.3 s` | Minimum time between servo commands |
| `CAL_DELTA` | `25 mm` | Probe distance for Y/Z Jacobian calibration |
| `MIN_Z_MM` | `-300 mm` | Z floor safety limit |

Tilt correction constants in `full_system_pipeline/pipeline/robot_controller_ext.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `TILT_DEAD_ZONE_DEG` | `1°` | No wrist rotation below this tilt |
| `TILT_MAX_STEP_DEG` | `3°/step` | Max wrist rotation per step |
| `TILT_MIN_INLIERS` | `500` | RANSAC inlier count gate for tilt trust |
| `TILT_GAIN` | `0.6` | Fraction of tilt error to correct each step |
