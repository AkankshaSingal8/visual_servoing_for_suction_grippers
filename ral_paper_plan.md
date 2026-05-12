# Implementation Plan: Visual Servoing for Suction Grippers (RAL)

## Context

The goal is a provably robust visual servoing system for suction-cup grippers that works off-the-shelf from a single reference image, targeting > 90% success across all robustness conditions (appearance, background, distractor, 3D rotation). The paper is an ablation study comparing three detection/segmentation backbones within the `servo_lastmile_v2.py` framework. The robot currently has a camera on the EE instead of a gripper; success = EE within threshold of the correct grasp point.

---

## Assumptions

1. **ZED Mini depth** is usable in the 30–400 mm approach range (minimum stereo baseline ~15 cm may fail below 30 mm; that's already the TERMINAL threshold).
2. **SAM3** (`facebook/sam3` on HuggingFace) is stable and available — if the model ID changes or weights move, the detection backbone needs updating.
3. **Hand-eye calibration** (`T_ee_cam`) can be measured and stays stable within a session. Signal A is non-functional without it.
4. **GT via manual placement**: moving the EE to the face center by eye is repeatable to ~3–5 mm — this is the floor of real-world GT precision.
5. **Image-space GT** is the SAM3 centroid on a clean, fully-visible, distant frame — valid as a reference but not the same as the true 3D face center.
6. **GDINO ablation**: GroundingDINO does not produce masks natively; the "GDINO only" variant uses the bounding box centroid, and "GDINO + SAM2" routes the GDINO box into SAM2 for a mask. Both run in the v2 state machine but without SAM3's one-step detect+segment.
7. **xArm wrist**: The xArm has 6-DOF; wrist rotation for tilt correction is achievable via `set_servo_angle` or `set_tool_position` with orientation commands. Speed and safety limits must be tuned.
8. **14 reference objects** in `assets/objects/` span enough diversity for the paper's robustness claim.
9. **> 90% success** is achievable with the full v2 pipeline before submission — this is the core paper claim and is unverified until Phase 6.

---

## Limitations

1. **No actual suction cup** — success is proximity-based (EE within X mm of face center), not vacuum grasp. The paper must frame this clearly.
2. **Single robot platform** (xArm 6) — no cross-platform generalization demonstrated.
3. **Manual GT precision floor** is ~3–5 mm, which constrains the resolution of real-world error claims.
4. **GDINO NEAR phase**: When the GDINO+SAM2 ablation enters NEAR state and the box exits frame, there is no SAM3-quality re-detection signal. Signal B (CoTracker3) and D (DINOv2) still run, but the SAM3 phase does not exist for those variants — the NEAR phase starts in `track` mode immediately.
5. **Error-over-distance hypothesis test**: Approach trajectories are robot-generated, not ground-truth. Measuring error during approach requires either a secondary calibrated camera or image-space annotation per frame — both are expensive. This means the hypothesis is tested indirectly (aggregate terminal error by distance bucket).
6. **Tilt correction safety**: Commanding wrist rotation during approach while also commanding XY centering creates coupled motions — rate limiting and sequential (not simultaneous) correction is required initially.
7. **CoTracker3 drift**: Drift accumulates during NEAR/track phase; long approaches (~600 mm) may exceed the 5 mm threshold from tracking error alone.
8. **Depth Anything V2** is available as fallback depth but untested in live mode — offline runs currently use ZED stereo depth only.
9. **No distractor experiments** until a physical setup with distractor objects is available — planned but requires physical lab setup.
10. **Tilt correction adds risk**: Wrist rotation commands near the box could cause collision if surface normal estimation is wrong. Needs a tilt-confidence gate.

---

## Step-by-Step Plan

### Phase 0 — Baseline Audit (Week 1)
*Goal: Confirm the current pipeline is reproducible and logging is correct before adding anything.*

**Step 0.1: Offline smoke test on all 14 reference objects**
- Run `servo_lastmile_v2.py --input-video` on an existing recording using each of the 14 `assets/objects/*.jpeg` reference images.
- Verify: no crashes, `frames.jsonl` written, `overlay.mp4` shows correct detections, `transition_debug/` populated on phase switch.
- Files: `foundation_model/servo_lastmile_v2.py`, `assets/objects/`

**Step 0.2: Verify frames.jsonl completeness**
- Confirm every field needed for metrics is present: `state`, `best_centroid`, `z_mm`, `tilt_deg`, `roll_deg`, `pitch_deg`, `near_plane_n_inliers`, `c_B`, `c_E`, `watchdog_alarm`, `_near_phase`.
- Confirm `_near_phase` is `sam3` vs `track` as expected.

**✅ Checkpoint 0**: Pipeline runs cleanly on 5+ objects offline; `frames.jsonl` has all required fields.

---

### Phase 1 — Metrics & Analysis Infrastructure (Weeks 2–3)
*Goal: Build the offline analysis tooling before collecting any new data.*

**Step 1.1: Write `evaluation/analyze_run.py`**
- Input: one or more `runs/*/frames.jsonl` paths.
- Outputs:
  - **Error-over-distance curve**: `|best_centroid - image_center|` vs `z_mm` per frame, grouped by state.
  - **Per-signal centroid plot**: c_A, c_B, c_C, c_D, c_E vs frame index.
  - **Tilt timeline**: `tilt_deg`, `roll_deg`, `pitch_deg` vs `z_mm`.
  - **Phase summary**: frames in FAR / NEAR-sam3 / NEAR-track / TERMINAL.
  - **Signal dominance**: which signals are non-null per frame.
- File: `evaluation/analyze_run.py`

**Step 1.2: Validate on existing runs**
- Run against `runs/v2_proteinbar_servo12/`, `runs/v2_servo10/`, etc.
- Verify plots are interpretable; identify any obvious logging gaps.

**Step 1.3: Image-space GT annotation tool**
- Simple script: given a reference image and a clean FAR-state frame from a run, allow clicking to mark the GT grasp point centroid.
- Saves per-run GT centroid as `runs/*/gt_centroid.json`.
- File: `evaluation/annotate_gt.py`

**✅ Checkpoint 1**: Can generate error-over-distance plots for 5 existing runs. Can annotate GT centroid and compute image-space centroid error (px) at terminal frame.

---

### Phase 2 — Ablation Pipeline Variants (Weeks 3–5)
*Goal: Implement GDINO-only and GDINO+SAM2 detection backends that slot into the v2 state machine.*

**Step 2.1: Define a common `DetectionBackend` interface**
- The `sam3_runner` callable in `LastMilePipelineV2.__init__` already provides the interface: `(frame_bgr) → dict` with keys `mask_np`, `gdino_box`, `sam_score`, `best_centroid`.
- Document this interface explicitly so any backend can plug in.
- File: `foundation_model/servo_lastmile_v2.py`

**Step 2.2: Implement `make_gdino_only_runner(ref_image, prompt)`**
- Uses GroundingDINO (already in `servo_pipeline.py`) for detection.
- Returns bounding box centroid as `best_centroid`; `mask_np = None`.
- In the v2 FAR state, this means no mask → NEAR phase starts in `track` immediately (no SAM3 phase).
- File: `foundation_model/detection_backends.py`

**Step 2.3: Implement `make_gdino_sam2_runner(ref_image, prompt)`**
- GroundingDINO for detection, SAM2 for mask (pipe GDINO box → SAM2 predictor).
- Returns full `mask_np` — NEAR phase can enter SAM3-phase-equivalent using SAM2 mask.
- Reuse `SAM2ImagePredictor` already used in `SignalC`.
- File: `foundation_model/detection_backends.py`

**Step 2.4: Add `--detector` CLI flag to `servo_lastmile_v2.py`**
- `--detector {sam3, gdino, gdino+sam2}` (default: `sam3`)
- Selects which runner factory to use.
- Logs detector used in `frames.jsonl` as `detector_used`.

**✅ Checkpoint 2**: All 3 variants run offline on the same input video. Side-by-side `frames.jsonl` comparison shows different `best_centroid` trajectories and `sam_score` distributions.

---

### Phase 3 — Hand-Eye Calibration (Week 4–5)
*Goal: Validate Signal A, which requires an accurate `T_ee_cam`.*

**Step 3.1: Perform hand-eye calibration**
- Use a calibration board (checkerboard or ChArUco) attached to the work surface.
- Move the EE through 15–20 poses; record EE pose + camera detection of board.
- Use OpenCV `calibrateHandEye` (TSAI or PARK method).
- Output: `config/hand_eye.npy` (4×4 T_ee_cam).

**Step 3.2: Validate reprojection error**
- Hold EE at known pose; project `lock.centroid_base` through T_ee_cam → image → compare to detected centroid.
- Target: < 5 px reprojection error across 5 validation poses.

**Step 3.3: Measure Signal A contribution**
- Run offline with `--hand-eye config/hand_eye.npy`; check that `c_A` is within ~10px of `best_centroid` during NEAR state.

**✅ Checkpoint 3**: Signal A reprojection error < 5px. `c_A` non-null in NEAR frames. Fusion weights show Signal A contributing meaningfully.

---

### Phase 4 — Tilt-Correcting Controller (Weeks 5–7)
*Goal: Close the loop on tilt — use measured roll/pitch to rotate the robot wrist during approach.*

**Step 4.1: Add wrist rotation commands to `RobotController`**
- `robot.correct_tilt(roll_deg, pitch_deg, confidence)`: sends relative wrist rotation.
- Use `arm.set_tool_position(roll=..., pitch=..., relative=True)` or `set_servo_angle`.
- Safety gate: only apply if `|roll_deg| > 1°` or `|pitch_deg| > 1°` (dead-zone), and `n_inliers > 500` (confidence gate).
- Rate limit: max 3°/step.
- File: `foundation_model/servo_pipeline_sam3.py` (`RobotController`)

**Step 4.2: Wire tilt correction into the live loop**
- In `perception_loop_for_frame` in `run_live_v2()`: after getting `res`, if `state == NEAR` and `tilt_deg > 1°` and inliers sufficient, call `robot.correct_tilt(roll_deg, pitch_deg, n_inliers)`.
- Apply tilt correction BEFORE the centroid-based XY servo step (sequential, not simultaneous).
- File: `foundation_model/servo_lastmile_v2.py`

**Step 4.3: Dry-run validation**
- Run with `--dry-run`: log what tilt correction commands would have been sent per frame.
- Verify: commands are bounded, not oscillating.

**Step 4.4: Live test on static tilted box**
- Place box at ~15° tilt. Robot approaches; verify tilt_deg decreases over approach.
- Record at least 5 trials; plot tilt_deg vs. z_mm.

**✅ Checkpoint 4**: On a statically tilted box (15°), tilt error decreases to < 3° before TERMINAL trigger. No collisions in 10 trials.

---

### Phase 5 — Real-World Ground Truth Protocol (Week 7–8)
*Goal: Establish the real-world error measurement pipeline.*

**Step 5.1: Define GT capture protocol**
- Manually jog EE to box face center (using live camera + overlay for alignment).
- Record EE pose via `robot._get_pos()` → save as `runs/*/gt_pose.json`.
- Repeat for each trial before running the autonomous approach.

**Step 5.2: Implement final-error computation**
- Load `gt_pose.json` + final EE pose from `frames.jsonl` TERMINAL frame.
- Compute lateral error (XY), depth error (Z), 3D Euclidean error (mm), angular error (°).
- File: `evaluation/analyze_run.py` (extend from Phase 1)

**Step 5.3: Define success thresholds**
- Lateral error < 5 mm AND depth error < 8 mm AND tilt error < 3° = success.
- Log binary success per trial.

**✅ Checkpoint 5**: GT captured for 10 manual trials. Final error computable. Mean error and success rate reported for current v2 SAM3 pipeline.

---

### Phase 6 — Systematic Experiment Collection (Weeks 8–14)
*Goal: Collect data across the robustness axes for all 3 pipeline variants.*

**Experiment Matrix (minimum viable: 300 trials):**

| Axis | Levels | Count |
|------|--------|-------|
| Objects | protein_bar, cheez_it, amazon_tissue, cardboard_box, mac_and_cheese | 5 |
| Condition | single / 1 distractor same texture / 1 distractor different / 3D tilt (30°) | 4 |
| Pipeline | SAM3+v2 / GDINO+SAM2 / GDINO-only | 3 |
| Trials | 5 per cell | 5 |
| **Total** | | **300 trials** |

**Step 6.1: Build run automation script**
- `evaluation/run_experiment.py --object cheez_it --condition distractor_same --pipeline sam3 --trials 5`
- Automatically names output dirs, records GT, runs pipeline, saves all artifacts.
- Logs to `experiments/experiment_log.csv` (one row per trial).

**Step 6.2: Collect data in batches**
- **Batch A** (→ Checkpoint 6a): 5 objects × single-box × SAM3+v2 × 5 trials = 25 trials.
- **Batch B**: Add GDINO+SAM2 and GDINO-only on same single-box conditions.
- **Batch C**: Add distractor conditions.
- **Batch D**: Add 3D tilt conditions.

**✅ Checkpoint 6a**: 25 trials complete (SAM3+v2, single box, 5 objects). Preliminary success rate ≥ 70% confirms hardware setup is valid.

**✅ Checkpoint 6b**: Full 300-trial matrix complete. `experiments/experiment_log.csv` populated.

---

### Phase 7 — Statistical Analysis & Paper Figures (Weeks 14–16)
*Goal: Produce publication-ready results.*

**Step 7.1: Aggregate analysis**
- Tables: success rate per pipeline × condition; mean ± std errors; ablation contributions.
- Figures: error-over-distance curves; tilt error vs z_mm; success rate bar chart; qualitative overlays.
- File: `evaluation/aggregate_results.py`

**Step 7.2: Hypothesis test — does error decrease with distance?**
- Fit monotonic regression to centroid error vs. z_mm across all trials.
- Report whether error decreases monotonically, and where it stalls (FAR / NEAR-sam3 / NEAR-track).

**Step 7.3: Failure mode analysis**
- Categorize all failed trials: CoTracker drift / false detection / tilt overshoot / depth noise.
- File: `evaluation/failure_analysis.py`

**✅ Checkpoint 7**: All tables and figures drafted. SAM3+v2 ≥ 90% success on single-box. Ablation monotonically improving: GDINO < GDINO+SAM2 < SAM3+v2.

---

### Phase 8 — Paper Writing (Weeks 16–20)

- System description: state machine diagram, signal source table, controller block diagram.
- Experiments: robustness matrix, error-over-distance figures, ablation table, failure discussion.
- Reproducibility appendix: one-command offline demo, system requirements.

---

## New Files & Folder Structure

```
visual_servoing_for_suction_grippers/
├── foundation_model/
│   ├── servo_lastmile_v2.py         # MODIFIED: --detector flag, tilt correction wiring
│   ├── servo_pipeline_sam3.py       # MODIFIED: RobotController.correct_tilt()
│   └── detection_backends.py        # NEW: make_gdino_only_runner, make_gdino_sam2_runner
├── evaluation/                       # NEW FOLDER
│   ├── analyze_run.py               # NEW: offline metrics + plots from frames.jsonl
│   ├── annotate_gt.py               # NEW: click-to-annotate GT grasp point
│   ├── run_experiment.py            # NEW: automated trial runner
│   ├── aggregate_results.py         # NEW: cross-run statistics + paper figures
│   └── failure_analysis.py          # NEW: failed trial categorization
├── experiments/                      # NEW FOLDER (auto-created by run_experiment.py)
│   └── experiment_log.csv           # one row per trial
├── config/
│   └── hand_eye.npy                 # NEW: hand-eye calibration result
└── ral_paper_plan.md                # THIS FILE
```

---

## Critical Existing Files (Do Not Break)

| File | Role |
|------|------|
| `foundation_model/servo_lastmile.py` | Base signals A/B/C/D + math utils — reused, not modified |
| `foundation_model/servo_pipeline.py` | GDINO+SAM2 reference — port detection logic into backends |
| `foundation_model/test_servo_lastmile.py` | Existing unit tests — extend, never break |
| `assets/objects/*.jpeg` | 14 reference images for all experiments |

---

## Reusable Existing Functions

| Function | File | Use |
|----------|------|-----|
| `fit_plane_ransac` | `servo_lastmile.py:373` | Tilt estimation (already used) |
| `project_3d_to_pixel` | `servo_lastmile.py` | Signal A validation |
| `_make_overlay_v2` | `servo_lastmile_v2.py:833` | All variant overlays |
| `VideoRecorder`, `_serialize_result_v2` | `servo_lastmile_v2.py` | Experiment runner |
| `RobotController.calibrate` | `servo_pipeline_sam3.py` | Extend for tilt correction |
| `make_default_sam3_runner` | `servo_pipeline_sam3.py` | Template for detection_backends |

---

## Verification Checkpoints Summary

| # | Checkpoint | Pass criterion |
|---|-----------|----------------|
| 0 | Offline smoke test | No crash; frames.jsonl complete; transition_debug/ populated |
| 1 | Metrics tooling | Error-over-distance plots for 5 existing runs |
| 2 | Ablation variants | All 3 `--detector` modes run; centroids differ as expected |
| 3 | Hand-eye calibration | Signal A reprojection error < 5px |
| 4 | Tilt correction | Tilt error < 3° on tilted box; no collisions |
| 5 | Real-world GT | 10 trials; final error computable |
| 6a | Pilot experiments | 25 trials; ≥ 70% success on single-box |
| 6b | Full experiment matrix | 300 trials; experiment_log.csv complete |
| 7 | Paper results | SAM3+v2 ≥ 90% success; ablation monotonic |
