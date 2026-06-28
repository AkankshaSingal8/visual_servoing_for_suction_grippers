# SemVS — Project Progress & Experiment Plan

> **Last updated:** 2026-06-28
> **Canonical pipeline:** `foundation_model/servo_lastmile_v2.py` + `full_system_pipeline/`
> **Target venue:** IEEE Robotics and Automation Letters (RAL)

---

## Code Complete as of 2026-06-28

All 7 planned implementation tasks were completed on 2026-06-28:

| Task | Description | Key file(s) |
|------|-------------|-------------|
| Task 1 | Batch analysis pipeline — processes all v2_* run dirs, outputs batch_summary.csv | `full_system_pipeline/evaluation/batch_analyze.py` |
| Task 2 | Tilt/surface-normal fix — RANSAC plane fit in NEAR state (was hardcoded None) | `foundation_model/servo_lastmile_v2.py` |
| Task 3 | Unit tests — fusion logic, centroid error, weighted geometric median | `full_system_pipeline/tests/` |
| Task 4 | Offline detector ablation — SAM3 vs GDINO+SAM2 vs GDINO on recorded frames | `full_system_pipeline/evaluation/offline_detector_ablation.py` |
| Task 5 | Signal subset ablation — isolates contribution of each tracking signal A–E | `full_system_pipeline/evaluation/signal_ablation.py` |
| Task 6 | GT centroid annotation tool — click-based annotation, outputs gt_summary.csv | `full_system_pipeline/evaluation/annotate_gt.py` |
| Task 7 | Progress document — this file; single source of truth for system state | `docs/PROGRESS.md` |

---

## What Has Been Built (Code Complete)

### Canonical Pipeline: `foundation_model/servo_lastmile_v2.py`

The single runnable pipeline. All other files in `foundation_model/` are either imports or variants.

**State machine:** `FAR → NEAR (sam3 phase) → NEAR (track phase) → TERMINAL`

| State | Trigger | What runs |
|-------|---------|-----------|
| FAR | always | SAM3 detection + depth |
| NEAR-sam3 | LOCK (area 5–25%, depth 150–400mm, 3 consec frames) | SAM3 is sole signal; EMA smoothing |
| NEAR-track | box area > 45% OR bbox touches border | CoTracker3 (Signal B) + frozen SAM3 handoff anchor |
| TERMINAL | robot X > 505mm OR depth < 30mm | Signal A only; plane fit at TERMINAL |

**Four signals (NEAR-track only):**

| Signal | Method | Current status |
|--------|--------|---------------|
| A | Reproject locked 3D centroid via EE pose | Implemented. Requires `--hand-eye` |
| B | CoTracker3 dense point tracking from SAM3 mask | Implemented |
| C | SAM2 mask propagation | Implemented but intentionally low weight (watchdog only) |
| D | DINOv2 best-buddy patch correspondences | Implemented |
| E | Frozen SAM3 handoff anchor (NEAR-track fallback) | Implemented |

**Key improvements in v2 (over base):**
- Surface normal estimation in NEAR state via RANSAC plane fit — **FIXED 2026-06-28** (was hardcoded None)
- EE occlusion masking for Signal D
- 6-DOF Signal A via depth plane fit (`--sixdof`)
- Depth-only TERMINAL trigger (removed false trigger on centroid proximity)
- Periodic SAM3 re-anchor in NEAR-track phase
- Monotonic depth filter (rejects background bleed)

**Known remaining gaps:**
- Tilt correction command is computed but NOT sent to the robot controller. `RobotControllerExt.correct_tilt()` exists in `full_system_pipeline/pipeline/robot_controller_ext.py` but is not yet wired into the live servo loop.
- EE masking requires hand-eye calibration (`--hand-eye`). Without it, masking is disabled.
- Signal D (`c_D`) — DINOv2 — has a ~2cm positioning floor from ViT patch resolution (known ViT-VS limitation; we fuse it, not rely on it solely).
- 80px fusion override threshold is empirical, not derived from calibration data.

---

## Existing Run Data Inventory

<!-- AUTO-FILLED from runs/batch_summary.csv generated 2026-06-28 -->

19 runs total. 4 reached TERMINAL. Numbers below from `batch_analyze.py` output.

| Run | Frames | FAR | NEAR-sam3 | NEAR-track | TERMINAL | Reached? | Mean Z@term (mm) | Mean tilt@term (°) | Mean centroid err@term (px) | Watchdog |
|-----|--------|-----|-----------|------------|----------|----------|------------------|--------------------|-----------------------------|----------|
| v2_proteinbar_servo2 | 171 | 2 | 16 | 0 | 153 | YES | 437.5 | 4.9 | 0.12 | 0 |
| v2_proteinbar_servo12 | 103 | 2 | 46 | 29 | 26 | YES | 890.9 | 34.4 | 3.96 | 0 |
| v2_proteinbar_servo11 | 89 | 2 | 74 | 0 | 13 | YES | 1146.7 | 50.7 | 0.16 | 0 |
| v2_live_nocup_20260510_194142 | 95 | 2 | 80 | 0 | 13 | YES | — | 55.5 | 0.52 | 0 |
| v2_live_nocup_20260510_195718 | 111 | 2 | 40 | 69 | 0 | no | — | — | — | 42 |
| v2_live_nocup_20260510_204507 | 168 | 2 | 37 | 129 | 0 | no | — | — | — | 103 |
| v2_live_nocup_20260510_213018 | 749 | 2 | 747 | 0 | 0 | no | — | — | — | 0 |
| v2_servo11 | 161 | 2 | 159 | 0 | 0 | no | — | — | — | 108 |
| v2_servo9 | 65 | 2 | 63 | 0 | 0 | no | — | — | — | 20 |
| v2_servo6 | 184 | 2 | 182 | 0 | 0 | no | — | — | — | 0 |
| v2_servo7 | 98 | 2 | 96 | 0 | 0 | no | — | — | — | 0 |
| v2_servo5 | 92 | 2 | 90 | 0 | 0 | no | — | — | — | 0 |
| v2_servo8 | 42 | 2 | 40 | 0 | 0 | no | — | — | — | 3 |
| v2_servo3 | 24 | 2 | 22 | 0 | 0 | no | — | — | — | 3 |
| v2_servo4 | 29 | 2 | 27 | 0 | 0 | no | — | — | — | 0 |
| v2_servo10 | 28 | 2 | 26 | 0 | 0 | no | — | — | — | 0 |
| v2_live_nocup_20260510_204157 | 48 | 2 | 46 | 0 | 0 | no | — | — | — | 0 |
| v2_proteinbar_servo | 24 | 24 | 0 | 0 | 0 | no | — | — | — | 0 |
| v2_live_nocup_20260427_233614 | 517 | 517 | 0 | 0 | 0 | no | — | — | — | 0 |

**Key observations:**
- 4/19 runs (21%) reached TERMINAL. All 4 are protein_bar or no-cup runs with proper NEAR-sam3 phase.
- High tilt at terminal (34–50°) in 3 of 4 successful runs confirms tilt correction is not yet active.
- Watchdog alarms are concentrated in runs with a NEAR-track phase (signals diverging); NEAR-sam3-only runs are alarm-free.
- `v2_live_nocup_20260427_233614`: 517 FAR frames, never locked — camera not pointed at object.

**Annotated with GT centroid (gt_summary.csv):**
- Annotation tool (`annotate_gt.py`) is built. Run on protein_bar runs to populate gt_summary.csv before Phase 1 experiments.

---

## What Is NOT Built Yet

| Gap | Why it matters | When to fix |
|-----|---------------|-------------|
| Tilt correction not sent to robot | Roll/pitch errors accumulate throughout NEAR | First robot session |
| No suction gripper trials | Can't claim suction success rate | Requires physical gripper |
| EE masking not tested on real gripper | Body radius and tip offset are guesses | First robot session with gripper |
| Single start pose only | Generalization unproven | Phase 2 experiments |
| No multi-pose or multi-lighting tests | Robustness claims untested | Phase 2 experiments |

---

## Ablation Results

Both `signal_ablation.py` and `offline_detector_ablation.py` are built and runnable as of 2026-06-28.

### Detector Backend Comparison
Tool built: `full_system_pipeline/evaluation/offline_detector_ablation.py`

Run command:
```bash
python full_system_pipeline/evaluation/offline_detector_ablation.py \
  --run-dir runs/v2_proteinbar_servo12 \
  --output-dir /tmp/detector_ablation
```
Results pending robot-session data with sufficient FAR-state frames.

### Signal Subset Ablation

Tool built: `full_system_pipeline/evaluation/signal_ablation.py`

Results from two protein_bar runs (NEAR frames only, all conditions evaluated offline):

**v2_proteinbar_servo11** (74 NEAR frames, 28 with Signal A/B populated):

| Condition | N frames | Mean err (px) | Std (px) |
|-----------|----------|---------------|----------|
| E_only | 73 | 22.6 | 34.8 |
| all_signals | 73 | 23.0 | 34.6 |
| canonical | 74 | 23.5 | 34.8 |
| no_A | 73 | 31.0 | 32.1 |
| B_only | 28 | 46.4 | 21.5 |
| B_plus_D | 28 | 46.4 | 21.5 |
| no_B | 73 | 95.4 | 85.7 |
| A_plus_B | 28 | 176.0 | 32.4 |
| A_only | 28 | 385.1 | 79.2 |
| D_only | 0 | N/A | — |

**v2_proteinbar_servo12** (previously run): Signal A mean error ~370px, E_only mean error ~45px. Consistent with servo11 results.

**Key finding:** Signal A (reprojected 3D centroid via EE pose) is actively harmful without hand-eye calibration — mean error ~370–385px vs ~23px canonical. Signal E (frozen SAM3 handoff anchor) is the strongest single signal. Signal B (CoTracker3) adds value but dominates when A is excluded. Signal D (DINOv2) had zero populated frames in these runs (no NEAR-track phase reached for long enough).

Run on more runs once hand-eye calibration is available to get valid Signal A numbers.

---

## Experiment Plan (For When Robot Is Available)

### Phase 1: Baseline Validation (single object, single pose)
**Goal:** Establish a quantitative baseline for the canonical pipeline.
**Objects:** protein_bar (smallest, hardest), cardboard_box (easiest), cheez_it_box (medium)
**Start pose:** P1 (fixed, tape-marked on table, ~350mm from robot home in X)
**Trials per condition:** 10

| Condition | Pipeline | N trials | Success metric |
|-----------|----------|----------|----------------|
| Single box, P1 | SAM3 + v2 | 10 | Final 3D pos err ≤ 5mm AND tilt ≤ 3° |
| Single box, P1 | gdino+sam2 + v2 | 10 | Same |
| Single box, P1 | gdino + v2 | 10 | Same |

**Command (when robot available):**
```bash
python full_system_pipeline/evaluation/run_experiment.py \
  --object protein_bar \
  --condition single \
  --pipeline sam3 \
  --trials 10 \
  --ref-image assets/objects/protein_bar.jpeg \
  --experiment-log experiments/experiment_log.csv \
  --hand-eye config/hand_eye.npy
```

**What to log per trial:**
- Run tag (auto)
- Final robot position [x,y,z,roll,pitch,yaw] in mm/deg
- GT position measured manually with caliper (mm, 3-axis)
- Whether approach looked clean (video frame)
- Failure mode if it failed (see categories below)

### Phase 2: Robustness Sweep (once baseline passes Phase 1)
**Goal:** Demonstrate generalization across the stated robustness axes.

| Axis | Condition name | Change from baseline |
|------|---------------|---------------------|
| Object appearance | 5 different objects | Replace protein_bar with each object |
| Background | cluttered_shelf | Add other boxes on table |
| Distractors | distractor_hard | Same-size box of different object next to target |
| 3D rotation | tilted_15 | Tilt target box 15° around its Z axis |
| Lighting | dim_lighting | Reduce ambient light, no direct overhead |
| Multi-pose | P2, P3, P4 | 3 additional start poses at different XY offsets |

**Trials per condition:** 5 (minimum for a single data point in the paper)
**Total experiment time at 5 trials × 2min/trial × 8 conditions:** ~80 min

### Phase 3: Tilt Correction Validation
**Goal:** Prove the surface normal loop actually reduces tilt at contact.
**Requires:** `RobotControllerExt.correct_tilt()` wired into the live servo loop.

| Condition | Tilt correction | Expected result |
|-----------|----------------|----------------|
| control | OFF | tilt_at_terminal > 2° for tilted boxes |
| treatment | ON | tilt_at_terminal < 1.5° |

---

## Failure Mode Taxonomy

Use these labels when noting trial failures. `failure_analysis.py` uses these categories.

| Code | Name | Description |
|------|------|-------------|
| F1 | Detection failure | SAM3 or GDINO fails to find the object in FAR state |
| F2 | Lock failure | FAR→NEAR transition never fires (area/depth thresholds not met) |
| F3 | Track loss | CoTracker3 loses all points in NEAR-track phase |
| F4 | Watchdog alarm | Fusion signals disagree beyond threshold, servo paused |
| F5 | Wrong object | Pipeline locks onto distractor or background |
| F6 | Depth failure | ZED depth missing or noisy; z_mm unreliable |
| F7 | Geometric error | Centroid correct but final 3D position error > 5mm |
| F8 | Tilt error | Position OK but surface alignment > 3° |
| F9 | Robot error | xArm stops, E-stop, limits hit |

---

## Assumptions and Limitations (Baked Into the System)

These are **not** bugs — they are design choices that must be stated clearly in any paper or presentation.

### Geometry & Calibration
- **Hand-eye calibration required for Signal A and EE masking.** Without `--hand-eye`, both are disabled. The calibration file at `config/hand_eye.npy` must be re-run if the camera is remounted.
- **EE tip offset (160mm) and body radius (35mm) are estimates.** These must be measured from the physical gripper when it replaces the camera.
- **ZED depth accuracy degrades below ~100mm.** The TERMINAL trigger at 30mm depth is therefore unreliable. The position-based fallback (robot X > 505mm) compensates but depends on the robot starting at X = 205mm.
- **80px fusion override threshold** was chosen empirically. It corresponds to ~12.5% of image width at 640px. Not derived from calibration.

### Perception
- **"One-shot" not "zero-shot."** A reference image is required. The system does not work from a text prompt alone (unlike GDINO in text-only mode).
- **DINOv2 Signal D has a ~2cm positioning floor** from ViT patch resolution (1/14 of image width = ~4.6mm per patch at 640px). Fusion reduces this but does not eliminate it.
- **SAM3 prompt is always "box."** For non-box objects, this must be changed. For ambiguous scenes (multiple boxes), GDINO with a more specific prompt is recommended.
- **CoTracker3 extrapolates off-screen.** In NEAR-track phase, if all tracked points exit the frame, the extrapolated mean can diverge. The handoff anchor (Signal E) is the fallback.

### Evaluation
- **All current data is from approach only.** No suction contact. Success is defined as "final EE position within 5mm of GT" — not "suction cup forms a seal."
- **GT is collected via click-annotation** on a video frame, not an external measurement system. Annotation error is estimated at ±5px (~2mm at 350mm depth with ZED Mini).
- **Single start pose in current data.** Generalization to other poses is unvalidated.

### Codebase
- **Canonical script is `foundation_model/servo_lastmile_v2.py`.** Everything in `foundation_model/variants/` is frozen. Do not modify variants.
- **`full_system_pipeline/` is the RAL paper framework.** New analysis and experiment code goes here, not in `foundation_model/`.
- **The 3 detector backends (sam3, gdino+sam2, gdino) are defined in `full_system_pipeline/pipeline/detection_backends.py`** and imported by `servo_lastmile_v2_ext.py`. The base `servo_lastmile_v2.py` always uses SAM3.
