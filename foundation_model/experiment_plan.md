# Experiment 1: Servoing Precision with ArUco Ground Truth

## Overview

This document is the complete protocol for producing the **Servoing Precision** table in your RAL paper. The core idea is a two-phase trial:

1. **Ground truth phase**: An ArUco marker is placed at the exact center of the target box's top face. You record its 3D position in the robot's base frame using the ZED camera intrinsics and the xArm's known end-effector pose. Then you remove the marker.
2. **Servoing phase**: The robot is moved back to the starting pose. The SAM3 pipeline servos toward the same box (now without the marker). When servoing converges, you record the final end-effector position and attempt a suction grasp.

The difference between the ground truth position and the final end-effector position is your **alignment error**. The suction grasp outcome is your **grasp success** metric. Together these fill the main results table.

---

## Hardware Checklist

Before you begin, confirm you have all of the following ready:

- xArm mounted to tabletop, powered on, connected at `192.168.1.241`
- ZED Mini stereo camera mounted eye-in-hand on the xArm end-effector
- Suction cup gripper attached and connected to the vacuum pump
- A set of ArUco markers printed on plain white paper (details below)
- Double-sided tape or removable adhesive dots for attaching markers to boxes
- A ruler or calipers for measuring marker placement accuracy
- The boxes you will test (see Box Set section below)
- A notebook or spreadsheet open for manual annotations (grasp success, notes)
- The SAM3 pipeline (`servo_pipeline_sam3.py`) tested and working on at least one box

---

## ArUco Marker Preparation

### Marker Specifications

- **Dictionary**: `cv2.aruco.DICT_4X4_50` (small dictionary, fast detection, sufficient for this experiment)
- **Marker ID**: Use IDs 0 through 4 (one per box type, so you can identify which box is which in post-processing)
- **Printed size**: 30mm x 30mm. This is small enough to not interfere with the box surface appearance but large enough for reliable detection from 100-400mm distance with the ZED Mini
- **Print on white paper**, cut to size, attach with double-sided tape

### Why 30mm

At your closest servoing distance (~100mm from the box), the ZED Mini at HD720 resolution gives roughly 0.5mm/pixel. A 30mm marker occupies about 60x60 pixels, which is well above the minimum for ArUco detection (around 20x20 pixels). At your farthest starting distance (~400mm), the marker is still about 15x15 pixels, which is marginal but workable. If you find detection is unreliable at 400mm, increase the marker size to 40mm.

### Marker Placement

The marker must be placed at the **exact geometric center** of the target box's top face. Measure the box top face dimensions with calipers, divide by two, mark the center with a pencil, and align the marker center (not corner) with that pencil mark. This is critical: any offset in marker placement directly adds to your ground truth error floor.

For **stacked box** configurations, the marker goes on the **topmost box only** (since that is the suction target).

### Printing the Markers

Use OpenCV to generate them. Run this once on your laptop:

```python
import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for marker_id in range(5):
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    cv2.imwrite(f"aruco_marker_id{marker_id}.png", img)
```

Print each PNG at exactly 30mm x 30mm. Verify the printed size with a ruler.

---

## Box Set

You need a minimum of **5 visually distinct box types** to show generalization. Suggested set:

| Box ID | Description | Why it matters |
|--------|-------------|----------------|
| B1 | Plain brown cardboard (no printing) | Baseline, minimal visual texture |
| B2 | Printed graphics / branding on all faces | Tests whether SAM3 is confused by printed content (the key failure mode you are addressing) |
| B3 | Glossy or laminated surface | Specular reflections challenge depth estimation |
| B4 | Dark colored box (black or dark blue) | Low contrast against shadows, harder segmentation |
| B5 | Small form factor (e.g., phone-sized box) | Tests precision on small targets where mask quality matters more |

You can add more, but 5 is the minimum for a convincing table.

---

## Configuration Set

Each box type is tested in multiple physical configurations:

| Config ID | Description | Stack height | Notes |
|-----------|-------------|-------------|-------|
| C1 | Single box on table | 1 | Box sitting directly on the tabletop |
| C2 | Stack of 2 identical boxes | 2 | Two copies of the same box stacked vertically. Marker on top box. |
| C3 | Stack of 3 identical boxes | 3 | Three copies. Marker on top box. |
| C4 | Stack of 2 different boxes | 2 | Target box on top of a different box. Tests disambiguation when the stack is heterogeneous. |

If you do not have 3 copies of every box type, drop C3 for those types and note it in the paper.

---

## Starting Pose Set

The starting pose is the xArm end-effector position at the beginning of each servoing trial. You need variation here to show the system works from different viewpoints, not just one lucky configuration.

Define **3 starting poses** by manually jogging the xArm to positions where the target box is visible in the ZED frame but offset from the image center:

| Pose ID | Description | Approximate offset from box center |
|---------|-------------|-----------------------------------|
| P1 | Centered above, far | Directly above the box, ~400mm distance, small lateral offset |
| P2 | Left-offset, medium | ~200mm distance, box appears in the right half of the frame |
| P3 | Right-offset, close | ~150mm distance, box appears in the left half of the frame |

**Record the exact xArm joint angles or Cartesian coordinates for each starting pose.** You will need to return to the same starting pose for every trial in that condition. Save them in a JSON file:

```json
{
  "P1": {"x": 350.0, "y": 0.0, "z": 300.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
  "P2": {"x": 350.0, "y": -80.0, "z": 250.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
  "P3": {"x": 350.0, "y": 60.0, "z": 200.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}
}
```

Adjust these numbers to your actual tabletop geometry. The key requirement is that all three poses give a clear view of the target box in the ZED frame.

---

## Reference Images

For each box type, capture **one reference image** before starting trials:

1. Place the box (without the ArUco marker) in roughly the same position it will occupy during trials
2. Position the xArm so the box fills approximately 30-50% of the frame
3. Capture a single frame and save it as `ref_B1.png`, `ref_B2.png`, etc.

These reference images are used by the SAM3 pipeline's ResNet disambiguator. Use the **same reference image** for all trials of a given box type. Do not recapture between trials.

---

## Trial Protocol (Step by Step)

For each combination of (Box ID, Config ID, Pose ID), you will run **10 identical trials**. The full procedure for one trial is:

### Phase 1: Ground Truth Capture

1. **Set up the box configuration.** Place the box(es) on the table in the target configuration. Ensure the stack is stable and will not shift during the trial.

2. **Attach the ArUco marker.** Place the 30mm marker at the measured center of the top box's top face. Press firmly so it lies flat.

3. **Move the xArm to the starting pose.** Use the xArm SDK or teach pendant to move to the saved starting pose coordinates for this trial's Pose ID.

4. **Capture a ground truth frame.** Grab a single ZED frame from the starting pose. Run ArUco detection on this frame to get the marker's corner coordinates in pixels. Use `cv2.aruco.estimatePoseSingleMarkers()` with the ZED's camera intrinsics (from your `_load_zed_calibration()`) to get the marker's 3D position relative to the camera. Transform this into the robot base frame using the xArm's current end-effector pose (forward kinematics). Save:
   - The raw frame as `gt_frame.png`
   - The marker's 3D position in robot base frame as `gt_position_mm` (a 3-element vector: x, y, z)
   - The marker's 2D pixel center as `gt_center_px`
   - The xArm's current position as `gt_ee_position_mm`

5. **Remove the ArUco marker.** Peel it off carefully. Do not move the box. Do not bump the table.

### Phase 2: SAM3 Servoing

6. **Verify the box has not moved.** Quick visual check. If the box shifted, reposition it and redo Phase 1.

7. **Move the xArm back to the starting pose.** The exact same coordinates as step 3.

8. **Start the SAM3 pipeline.** Run:
   ```bash
   python foundation_model/servo_pipeline_sam3.py \
     --ref-image ref_B1.png \
     --debug \
     --log-dir logs/exp1
   ```
   The pipeline opens the camera, processes the reference image, calibrates the Jacobian, and begins servoing.

9. **Enable servoing.** Press `v` in the OpenCV window to toggle servo ON.

10. **Wait for convergence.** The pipeline servos until the pixel error enters the dead zone (`VS_DEAD_ZONE = 15 px`). Watch the overlay. When the error readout stabilizes near zero for at least 2 seconds, servoing has converged.

11. **Record the final state.** At convergence:
    - Note the final pixel error from the overlay (ex, ey)
    - Read the final xArm position from the SDK: `final_ee_position_mm`
    - Compute alignment error: `||final_ee_position_mm - gt_position_mm||` (Euclidean distance in 3D, and also decomposed into lateral and depth components)

12. **Attempt suction grasp.** Activate the vacuum pump. Command the xArm to descend the last few mm to make contact. Record:
    - `grasp_success`: 1 if suction engages and holds the box during a 5-second lift to 50mm above the table, 0 otherwise
    - `grasp_failure_mode`: if failed, note why (missed center, insufficient suction, box too heavy, etc.)

13. **Press `q` to quit the pipeline.** The SVO recording and annotated MP4 are saved automatically.

14. **Save all trial data.** Move the debug dump, logs, and recordings into a trial-specific directory:
    ```
    data/exp1/B1_C1_P1_trial01/
      gt_frame.png
      gt_position.json        # gt_position_mm, gt_center_px, gt_ee_position_mm
      final_position.json     # final_ee_position_mm, final_pixel_error
      grasp_result.json       # grasp_success, grasp_failure_mode
      artifacts/              # debug dump from --debug
      logs/                   # pipeline logs
      vs_recording_*.svo2     # raw SVO recording
      vs_annotated_*.mp4      # annotated overlay video
    ```

15. **Repeat from step 3** for the next trial (same box, config, pose). After 10 trials, change to the next condition.

---

## Trial Matrix

The full experiment is a grid of conditions. Here is the complete matrix:

```
Box types:     B1, B2, B3, B4, B5                     = 5
Configurations: C1, C2, C3, C4                        = 4
Starting poses: P1, P2, P3                            = 3
Trials per condition:                                  = 10
                                                       -----
Total trials:  5 x 4 x 3 x 10                         = 600
```

**If 600 trials is too many**, reduce to the minimum viable set:

```
Box types:     B1, B2, B3                              = 3
Configurations: C1, C2                                 = 2
Starting poses: P1, P2, P3                            = 3
Trials per condition:                                  = 10
                                                       -----
Total trials:  3 x 2 x 3 x 10                         = 180
```

180 trials is the absolute minimum for a credible RAL table. At roughly 3-5 minutes per trial (including setup, marker placement/removal, servoing, grasp attempt, data saving), expect 9-15 hours of lab time for the minimum set.

---

## Data File Formats

### gt_position.json

```json
{
  "trial_id": "B1_C1_P1_trial01",
  "box_id": "B1",
  "config_id": "C1",
  "pose_id": "P1",
  "trial_number": 1,
  "marker_id": 0,
  "marker_size_mm": 30.0,
  "gt_position_mm": [352.1, -3.4, 182.7],
  "gt_center_px": [641, 358],
  "gt_ee_position_mm": [350.0, 0.0, 300.0],
  "camera_intrinsics": {
    "fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0
  },
  "timestamp": "2026-04-27T14:23:01"
}
```

### final_position.json

```json
{
  "trial_id": "B1_C1_P1_trial01",
  "final_ee_position_mm": [352.8, -2.9, 185.1],
  "final_pixel_error": [3, -5],
  "convergence_time_s": 8.4,
  "convergence_frames": 28,
  "n_redetections": 0,
  "final_similarity": 0.87,
  "path_length_mm": 142.3
}
```

### grasp_result.json

```json
{
  "trial_id": "B1_C1_P1_trial01",
  "grasp_success": 1,
  "grasp_failure_mode": null,
  "lift_height_mm": 50.0,
  "hold_duration_s": 5.0,
  "notes": ""
}
```

---

## Computing Alignment Error

After all trials are complete, compute these metrics per trial from the saved JSON files:

```
alignment_error_3d = sqrt(
    (final_ee_x - gt_x)^2 +
    (final_ee_y - gt_y)^2 +
    (final_ee_z - gt_z)^2
)

lateral_error = sqrt(
    (final_ee_y - gt_y)^2 +
    (final_ee_z - gt_z)^2
)

depth_error = abs(final_ee_x - gt_x)
```

Here x is the xArm's forward axis (toward the box), and y/z are the lateral axes. Adjust axis assignments to match your xArm base frame orientation. The lateral error is what matters most for suction cup alignment. The depth error affects whether the cup makes proper contact but is partially handled by the final descent command.

---

## The Table You Will Put in the Paper

The main results table should look like this:

```
Table I: Servoing Precision Across Box Types and Configurations
(mean +/- std over N=10 trials per cell, 3 starting poses pooled)

| Box  | Config | Lateral Err (mm) | Depth Err (mm) | 3D Err (mm) | Grasp (%) | Conv. Time (s) |
|------|--------|-------------------|-----------------|-------------|-----------|-----------------|
| B1   | C1     | 2.3 +/- 0.8      | 1.1 +/- 0.5    | 2.7 +/- 0.9 | 100       | 7.2 +/- 1.4     |
| B1   | C2     | 3.1 +/- 1.2      | 1.4 +/- 0.7    | 3.5 +/- 1.3 | 97        | 8.1 +/- 1.8     |
| B2   | C1     | ...               | ...             | ...         | ...       | ...             |
| ...  | ...    | ...               | ...             | ...         | ...       | ...             |
|------|--------|-------------------|-----------------|-------------|-----------|-----------------|
| ALL  | ALL    | X.X +/- X.X      | X.X +/- X.X    | X.X +/- X.X | XX        | X.X +/- X.X     |
```

The "ALL / ALL" row at the bottom aggregates across all conditions. This is the number reviewers will look at first.

If the table gets too large (5 boxes x 4 configs = 20 rows), condense it by grouping:

- Rows grouped by configuration (pooling across box types): shows how stacking affects precision
- A separate smaller table grouped by box type (pooling across configurations): shows how surface appearance affects precision
- Starting pose effects reported in text, not in the main table, unless there is a significant difference

---

## Secondary Metrics Table

A second table reports the perception quality metrics that explain the servoing results:

```
Table II: Perception Metrics Per Condition
(mean over all trials in that condition)

| Box  | Config | Detection Recall (%) | Disambig. Acc (%) | Mask IoU | Similarity |
|------|--------|----------------------|-------------------|----------|------------|
| B1   | C1     | 100                  | 100               | 0.91     | 0.89       |
| B1   | C2     | 100                  | 93                | 0.85     | 0.82       |
| ...  | ...    | ...                  | ...               | ...      | ...        |
```

Where:
- **Detection recall**: % of frames during servoing where SAM3 produced at least one valid detection
- **Disambiguation accuracy**: % of frames where the correct box in the stack was selected (for C1 this is trivially 100%; for C2-C4 you verify by checking that the chosen box's centroid is on the top box, not a lower one)
- **Mask IoU**: average IoU between the SAM3 mask and a manually annotated mask on a subset of frames (sample 5 frames per trial, annotate offline)
- **Similarity**: average ResNet cosine similarity between the chosen crop and the reference image

---

## Statistical Reporting

For each metric, report:

- **Mean and standard deviation** across the 10 trials (or 30 trials if pooling across 3 starting poses)
- **Median** if the distribution is skewed (check with a histogram)
- **95% confidence interval** if you want to be thorough: `mean +/- 1.96 * std / sqrt(N)`

For comparing conditions (e.g., "is C2 significantly worse than C1?"), use a **Wilcoxon signed-rank test** (non-parametric, paired by starting pose). Report p-values. A common threshold is p < 0.05 for significance.

---

## Directory Structure for the Full Experiment

```
experiment_1_servoing_precision/
  README.md                          # This file
  starting_poses.json                # Saved xArm coordinates for P1, P2, P3
  reference_images/
    ref_B1.png
    ref_B2.png
    ref_B3.png
    ref_B4.png
    ref_B5.png
  aruco_markers/
    aruco_marker_id0.png             # For B1
    aruco_marker_id1.png             # For B2
    aruco_marker_id2.png             # For B3
    aruco_marker_id3.png             # For B4
    aruco_marker_id4.png             # For B5
  data/
    B1_C1_P1/
      trial_01/
        gt_frame.png
        gt_position.json
        final_position.json
        grasp_result.json
        artifacts/
        logs/
        vs_recording_*.svo2
        vs_annotated_*.mp4
      trial_02/
        ...
      ...
      trial_10/
        ...
    B1_C1_P2/
      ...
    B1_C2_P1/
      ...
    ...
  analysis/
    compute_metrics.py               # Script to parse all JSONs and produce tables
    results_table.csv                 # Computed metrics per trial
    results_summary.csv              # Aggregated table (the one you put in the paper)
    figures/
      alignment_error_boxplot.pdf
      convergence_time_boxplot.pdf
      grasp_success_bar.pdf
```

---

## Timing Estimate

| Task | Time |
|------|------|
| Print and cut ArUco markers | 30 min |
| Capture reference images (5 boxes) | 30 min |
| Define and save starting poses | 30 min |
| Per trial (marker on, GT capture, marker off, servo, grasp, save) | 3-5 min |
| 180 trials (minimum set) | 9-15 hours |
| 600 trials (full set) | 30-50 hours |
| Post-processing and analysis | 3-5 hours |
| Manual mask annotation for IoU (subset) | 2-3 hours |

Spread the 180-trial minimum across 3 to 4 lab sessions. Do not rush. Fatigue leads to sloppy marker placement and missed data.

---

## Common Pitfalls to Avoid

1. **Moving the box when removing the marker.** Use removable adhesive dots, not strong tape. Practice peeling technique. If the box shifts even slightly, redo Phase 1.

2. **Forgetting to return to the starting pose.** After each trial, always command the xArm back to the exact saved coordinates before starting the next trial.

3. **Inconsistent lighting between trials.** Keep the lab lighting the same for all trials in a session. If you are running Experiment 4 (lighting variation), do that as a separate batch.

4. **Reference image mismatch.** If you recapture a reference image mid-experiment, all trials after that point are not comparable to trials before it. Use one fixed reference image per box type for the entire experiment.

5. **Not recording grasp success immediately.** Write it down or enter it into the JSON right after the grasp attempt. Do not rely on memory.

6. **ArUco detection failure at the starting pose.** If the marker is too small or too far, detection will fail. Test detection from each starting pose before beginning trials. If detection fails from P1 (far), either increase marker size or move P1 closer.

7. **Servo convergence ambiguity.** Define convergence as: pixel error magnitude below `VS_DEAD_ZONE` (15 px) for at least 5 consecutive frames. If the system oscillates and never stabilizes, record the trial as a convergence failure and note it.

8. **Depth coordinate confusion.** The xArm base frame and the camera frame have different axis conventions. The ZED camera frame has z-forward, x-right, y-down. The xArm base frame has x-forward, y-left, z-up (verify this with your specific mounting). The ArUco pose estimation gives you position in the camera frame. You must transform it to the robot base frame using the end-effector's known pose (forward kinematics). Get this transform right once, verify it by moving the marker to a known position and checking the computed coordinates, then use it for all trials.

---

## What to Do If Results Are Bad

If your overall grasp success rate is below 80% or your mean alignment error exceeds 10mm, the paper will not be competitive. Before concluding the system does not work:

- Check the Jacobian calibration quality. Run calibration 3 times and compare the J_yz matrices. If they vary wildly, the calibration procedure is unreliable and you may need to increase `CAL_DELTA` or `CAL_WAIT`.
- Check whether failures cluster on specific box types or configurations. If B3 (glossy) always fails, that is a result worth reporting as a limitation, not a reason to discard the paper.
- Check whether failures cluster on a specific starting pose. If P3 (close start) always fails, it may indicate the pipeline needs more than ~150mm of approach distance to converge.
- Check the debug dumps for failed trials. Look at `detections.png` to see if SAM3 detected the wrong box or produced a bad mask. If the perception is correct but the servo diverges, the issue is in the controller, not the perception pipeline.

---

## Minimum Viable Experiment for a First Pass

If you want to validate the protocol before committing to 180+ trials:

1. Pick one box (B1), one configuration (C1), one starting pose (P1)
2. Run 5 trials following the full protocol
3. Compute alignment error and grasp success
4. If the numbers look reasonable (error < 10mm, grasp > 80%), proceed to the full matrix
5. If not, debug using the checklist above before scaling up

This pilot run takes about 30 minutes and will save you from discovering protocol issues after 100 trials.