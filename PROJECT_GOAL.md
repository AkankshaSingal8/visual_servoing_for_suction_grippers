# Project Goal

## Objective

Develop a provably robust, high-accuracy visual servoing system for suction cup grippers that can be pulled off the shelf and deployed without task-specific re-training. The system is being developed for submission to the **IEEE Robotics and Automation Letters (RAL)** journal.

## Problem Statement

Suction cup grasping of boxes in unstructured environments requires two things to work reliably: finding a good grasp point on the target object, and closing the distance to that point accurately enough to make contact. Visual servoing — using continuous visual feedback to guide the end effector — is a natural fit, but existing approaches tend to be brittle to changes in appearance, background, or viewpoint. This project asks: can a system built on modern foundation models be robust enough to serve as a drop-in solution across real-world deployment conditions?

## Current Hardware Setup

The robot arm is an xArm with a ZED Mini stereo camera mounted on the end effector in place of the gripper. This camera-on-EE configuration lets us develop and evaluate the full visual servoing loop — perception, grasp point estimation, and approach trajectory — without requiring a physical suction cup during the research phase. The system is designed so that the camera mount can be replaced by a suction gripper with minimal changes to the pipeline.

## Robustness Requirements

A key claim of the paper is that the system is robust across the following axes. Each must be demonstrated empirically:

- **Object appearance variation** — different box types, sizes, and surface textures.
- **Background / environment variation** — cluttered shelves, varying lighting, different table surfaces.
- **Distractors** — other boxes with different textures, objects of similar shape, boxes with similar texture to the target (hard negatives).
- **3D rotation** — the target box may be tilted or rotated in all three axes; the system must still find a valid grasp point and approach it correctly.

## Reference Image Interface

The user provides a single reference image of the target object. The system uses this image to identify the target in the scene and estimate where to grasp it. No fine-tuning or additional annotation is required. This one-shot interface is a core usability requirement for the off-the-shelf claim.

## Grasp Point Estimation

For boxes, the natural grasp point for a suction cup is the face center. However, when the box is rotated in 3D, the visible face may be tilted relative to the camera, which means:

1. The grasp point should be the center of the visible face in 3D (not just the 2D image centroid).
2. The approach direction should be aligned with the surface normal of that face, not the camera Z axis.

The pipeline estimates the surface normal via RANSAC plane fitting on depth data and uses this both to identify the correct grasp point and to detect angular misalignment (tilt error) that the controller should correct before final approach.

## Visual Servoing Hypothesis

The central hypothesis is that visual feedback should produce a progressively more accurate grasp point estimate as the end effector approaches the target — the system should get better the closer it gets. This is expected but not yet proven: at close range, depth measurements become noisier, the box may exit the camera field of view, and the foundation models may behave differently on close-up crops than on full-scene images. Quantifying whether and when the error decreases with distance is an open research question this project aims to answer.

## Pipeline Comparison

Multiple perception architectures are implemented and compared to identify which foundation model combination gives the best accuracy and robustness:

- **Detection**: GroundingDINO, OWLv2
- **Segmentation / tracking**: SAM2, SAM3 (Segment Anything Model 3)
- **Feature matching**: DINOv2 best-buddy correspondences
- **Point tracking**: CoTracker3
- **Depth**: ZED stereo depth, Depth Anything V2

The system is structured as a state machine (FAR → NEAR → TERMINAL) that transitions between different signal sources as the robot approaches. In FAR state, SAM3 is the primary detector. In NEAR state, the box may exit the frame and the system hands off to point tracking. In TERMINAL state, depth-based plane fitting drives the final correction.

## Evaluation Metrics

Two evaluation regimes are needed:

### Image-Space Metrics
- Centroid error (px): distance between the estimated grasp point and ground-truth centroid in the image.
- Mask IoU: overlap between the predicted segmentation mask and the ground-truth mask.
- Tilt error (°): angular deviation of the estimated surface normal from the true box face normal.
- Signal consistency: agreement between the different signal sources (A/B/C/D) used in fusion.

### Real-World Metrics
- Final 3D position error (mm): distance between the EE tip and the target grasp point at the moment the approach terminates.
- Final angular error (°): misalignment between the EE approach axis and the box surface normal.
- Success rate: fraction of trials where the EE is within a threshold of the grasp point (e.g., < 5 mm, < 3°).

### Error Over Time / Distance
- Centroid error vs. EE-to-target distance: does the error decrease monotonically as the robot approaches?
- Tilt error vs. depth: does surface normal estimation improve at closer range?
- These trajectories are logged per-frame in JSONL format (`frames.jsonl`) for offline analysis.

## Deliverables

- A reproducible codebase that runs on any xArm + ZED setup with a single reference image as input.
- Quantitative comparison of pipeline variants across the robustness axes above.
- Per-frame metrics logged for every run, enabling post-hoc analysis of error-over-distance curves.
- A RAL paper reporting the results.
