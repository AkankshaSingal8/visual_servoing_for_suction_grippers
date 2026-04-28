# Last-Mile Visual Servoing: Research & Approach

This document surveys the state of the art in close-range visual servoing for suction grasping, evaluates
whether the current pipeline should be replaced or augmented, and gives a concrete recommendation.

---

## Problem Statement

**SAM3 gets the robot close to the box (~150–400 mm away), but the last centimeters are the hardest.**

Three things make this regime uniquely difficult:

1. **Object exits the camera frame.** As the EE descends, the box occupies more and more of the image until
   portions of it, or the entire top face, fall outside the field of view. Detectors trained on fully-framed
   objects fail here.

2. **Depth noise increases.** Stereo depth degrades at very short baselines. The ZED's accuracy drops
   significantly below ~100 mm, making direct depth-to-contact estimation unreliable.

3. **Orientation errors compound.** A 2–3° tilt in the approach angle becomes a significant surface offset
   at contact. Suction cups have limited compliance; a misaligned approach kills the seal. Any servoing
   method that only corrects XY centroid without correcting roll/pitch can still fail at contact.

---

## Current Pipeline: SemVS (Semantic + Geometric Fusion)

### State Machine

```
FAR → LOCK → NEAR → TERMINAL
```

| State | Trigger condition | Action |
|-------|------------------|--------|
| FAR | Always (default) | Run SAM3 per frame; detect + classify object |
| LOCK | Object area 5–25%, depth 150–400 mm, 3 frames sustained | Freeze 3D anchor snapshot |
| NEAR | Immediately after LOCK | Run 4-signal fusion each frame |
| TERMINAL | Depth < 30 mm OR centroid error < 8 px | Plane-fit surface; trigger contact |

### Four Fusion Signals (NEAR state)

| Signal | Method | Weight logic | Strength | Weakness |
|--------|--------|-------------|----------|----------|
| A — Geometric anchor | Reproject locked 3D centroid to current camera frame using EE pose + hand-eye cal | 1.0 if calibrated, else 0.0 | Deterministic; no vision needed | 3-DOF centroid only; no orientation; drifts with hand-eye cal error |
| B — CoTracker3 | Dense point tracking from uniform mask seed (80 pts) | Visibility fraction × per-point confidence | Handles partial framing; unbiased mean even off-frame | Long-range drift; fails if all points exit frame |
| C — SAM2 propagation | Mask propagation from locked hint | SAM2 score × (1 − border fraction), capped 0.4 | Watchdog sanity check | Hallucinates at borders; intentionally weak weight |
| D — DINOv2 best-buddy | Cosine-similarity patch correspondence: ref crop ↔ live frame | Mean similarity × (matches / 30) | Appearance-robust; no tracking drift | ~2 cm floor from ViT patch resolution; degrades when EE occludes box |

**Fusion rule**: Weighted geometric median of all signals. If result is > 80 px from Signal A, override with A
(geometry wins).

**Current gap**: Surface normal / tilt estimation only runs at TERMINAL (< 30 mm depth). Roll/pitch errors
accumulate silently throughout the NEAR approach.

---

## Surveyed Alternative Methods

### 1. ViT-VS — Vision Transformer Feature IBVS (IROS 2025)

**Paper**: [ViT-VS: On the Applicability of Pretrained Vision Transformer Features for Generalizable Visual
Servoing](https://arxiv.org/abs/2503.04545) — Scherl et al., IROS 2025.
[GitHub](https://github.com/AlessandroScherl/ViT-VS)

**Method**: Extracts DINOv2 patch embeddings from a reference image and current frame, finds best-buddy
correspondences via cyclic cosine similarity, then runs a classical IBVS control law using the interaction
(Jacobian) matrix with depth from RGB-D.

**Results**: Full convergence on unperturbed scenarios; +31.2% over classical SIFT-based IBVS in perturbed
scenarios. Validated on industrial box manipulation.

**Relation to current pipeline**: Signal D already implements the core idea (DINOv2 best-buddy matching).
ViT-VS uses these features as the *sole* controller; the current pipeline uses it as one of four fusion inputs.

**Positioning error**: ~2 cm (hard floor from ViT patch resolution = 1/14 of input image).

**Does not address**: Close-range frame exit, EE occlusion, surface normal alignment.

**Verdict**: No improvement over current Signal D. Not recommended as a replacement.

---

### 2. SVM — Servoing with Vision Models (Feb 2025)

**Paper**: [A Training-Free Framework for Precise Mobile Manipulation of Small Everyday Objects](https://arxiv.org/abs/2502.13964) — Feb 2025.

**Method**: Open-vocabulary detector + point tracker generate a 3D target from an RGB-D wrist camera. A
closed-loop VS loop drives the EE to that target. Crucially, the paper uses **out-painting** to fill in
the EE-occluded region when the end-effector enters the frame — this prevents the tracker from latching
onto the EE rather than the target.

**Results**: 71% zero-shot success on 72 novel object instances across 10 environments; +42% over open-loop
control; +50% over an imitation learning baseline trained on 1000+ demonstrations.

**Relation to current pipeline**: Their 3D target loop is analogous to Signal A (geometric anchor projection).
Their out-painting idea is novel relative to the current pipeline.

**Key insight to adopt**: **EE occlusion masking / out-painting for Signal D**. When the suction gripper
enters the frame, DINOv2 may match EE surface patches to box patches. Masking the EE silhouette from the
current frame before computing correspondences would prevent this.

**Verdict**: Does not replace the current architecture. EE masking idea is directly adoptable.

---

### 3. PBVS with 6-DOF Pose Estimation (FoundationPose / box plane)

**Paper**: [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://nvlabs.github.io/FoundationPose/) —
CVPR 2024 Highlight. [GitHub](https://github.com/NVlabs/FoundationPose)

**Also relevant**: [Rapid Deployment Pipeline Using FoundationPose + SAM](https://arxiv.org/abs/2604.17258) — shows
FoundationPose driven by SAM-generated meshes, very close to this pipeline.

**Method**: Full 6-DOF pose estimation (position + orientation) from RGB-D stream + a 3D mesh or reference
image. Runs at ~30 fps on GPU. Once pose is known, a Position-Based Visual Servoing (PBVS) control law
drives all 6 DOF to a target pose.

**For boxes specifically**: A full 3D mesh is not required. A box face is a flat plane. RANSAC plane
fitting on the ZED depth map gives the face normal and centroid — enough for 6-DOF PBVS without any CAD model.

**Key benefit over current Signal A**: Signal A gives only XY centroid correction (3-DOF). A 6-DOF PBVS
signal also corrects roll, pitch, and Z stand-off distance throughout the NEAR approach, not just at TERMINAL.

**For suction grippers this matters most**: A 2° tilt at 100 mm range = ~3.5 mm offset at the contact patch.
Typical suction cup compliance is 1–2 mm. Correcting tilt early rather than late (TERMINAL) reduces failed
seals significantly.

**Requirement**: ZED depth (already available) + box face RANSAC plane fit (already done at TERMINAL).

**Verdict**: Highest potential impact. Does not require any new models for box-grasping specifically. The
plane + rectangle estimator can be moved from TERMINAL-only to every NEAR frame.

---

### 4. Dense Depth-Based Surface Normal Estimation (in NEAR)

This is a targeted, lightweight augmentation rather than a complete alternative.

**What**: Run RANSAC plane fitting on the ZED depth map ROI centered on `best_centroid` every frame during
NEAR state. The current code already does this at TERMINAL; running it throughout NEAR costs minimal compute.

**Output**: Surface normal vector in camera frame → tilt correction for robot controller.

**Why it matters**: The NEAR phase spans roughly 30–150 mm of depth. Tilt errors that accumulate silently
during NEAR are only corrected at TERMINAL (< 30 mm), which may be too late.

**Cost**: No new models. Uses existing ZED depth + the plane-fitting code already in `servo_lastmile.py`.

**Verdict**: Best effort-to-impact ratio. Should be the first addition to the current pipeline.

---

### 5. Learning-Based Approaches: Diffusion Policy / ACT

**Papers**:
- [Diffusion Policy (Chi et al., IJRR 2025)](https://diffusion-policy.cs.columbia.edu/)
- [3D Diffusion Policy / DP3 (RSS 2024)](https://3d-diffusion-policy.github.io/)
- [ACT: Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/)

**What**: A visuomotor policy (neural network) takes wrist camera images + robot state → delta EE action.
Trained on 50–200 teleoperation demonstrations per task. Implicitly learns all geometry, tilt, and occlusion
handling.

**Strengths**:
- No hand-eye calibration required; learned end-to-end
- Implicitly handles all close-range edge cases that appear in training data
- Strong results for precise manipulation (peg insertion, cup stacking) in recent benchmarks

**Weaknesses for this setting**:

| Issue | Detail |
|-------|--------|
| Data collection cost | Requires 50–200 real-robot demos *per box type* or heavy augmentation; hours of teleoperation |
| Generalization gap | Different box sizes, textures, and surface finishes are out-of-distribution without specific augmentation or meta-learning |
| Inference latency | Diffusion Policy: 50–200 ms/step; ACT: 30–80 ms/step. Fine servoing at ≥ 10 Hz is marginal |
| No graceful degradation | Black box — no per-signal diagnostics, no watchdog, no geometric fallback |
| Close-range vs far-range split | A policy trained for last-mile (30–150 mm) still needs the upstream SAM3/SemVS pipeline to arrive there |

**Verdict**: Premature for an open-world multi-box setting. The current non-learning pipeline is already
competitive with imitation learning baselines (SVM showed +50% over IL with 1000 demos using a similar
geometric closed-loop approach). **Do not recommend replacing the current architecture with IL.** If
demonstrations are ever collected systematically for a specific box, fine-tuning a residual policy on top
of the current controller is a reasonable future direction.

---

## Comparison Table

| Method | Training needed | New models | XY centroid | Surface normal | Close-range frame exit | Box generalizes | Recommended |
|--------|----------------|------------|-------------|----------------|----------------------|-----------------|-------------|
| Current SemVS (4-signal) | None | SAM3, SAM2, DINOv2, CoTracker | Yes | Terminal only | Partial (CoTracker) | Yes | Baseline |
| ViT-VS | None | DINOv2 | Yes | No | No | Yes | No — already in Signal D |
| SVM | None | Detector + tracker | Yes | No | No (out-painting helps) | Yes | Adopt EE masking idea |
| PBVS + plane estimator | None | None (depth only) | Yes | **Yes (NEAR)** | Via Signal A | Yes | **Yes — Signal E** |
| Diffusion Policy | Yes (demos) | Policy network | Implicitly | Implicitly | Implicitly | Uncertain | Not yet |

---

## Recommendation

**Keep the 4-signal fusion architecture. Add two targeted improvements.**

### Improvement 1 (High impact, low cost): Surface Normal Feedback in NEAR

Run the existing RANSAC plane fitting on every NEAR frame, not just TERMINAL. Use the surface normal from
the depth map ROI around `best_centroid` to estimate the box face orientation. Feed this as a tilt-correction
signal to the robot controller alongside the existing XY centroid signal.

- Uses ZED depth already available at every frame
- Reuses the plane-fitting code already in `servo_lastmile.py` (TERMINAL state)
- Eliminates the "approaching at an angle" failure mode that currently goes undetected until < 30 mm depth

### Improvement 2 (Medium impact, medium cost): EE Occlusion Masking for Signal D

As the robot descends, the suction gripper enters the camera frame. Signal D (DINOv2 best-buddy matching)
may then match patches on the gripper body to patches on the box, pulling the fused centroid toward the
gripper. Masking the predicted EE silhouette from the current frame before computing correspondences prevents
this.

- EE silhouette can be estimated from known EE geometry + EE pose
- Inspired by SVM's out-painting; masking is simpler and sufficient
- Particularly important below ~100 mm depth where the gripper is large in frame

### Optional Improvement 3 (High impact, higher implementation cost): Full 6-DOF Box Pose

Replace the 3-DOF Signal A (centroid projection) with a full 6-DOF box face pose estimator:

1. Segment the visible box face using the SAM2/SAM3 mask
2. RANSAC plane fit on that mask's depth points → centroid + normal
3. PnP or direct PBVS control law that drives all 6 DOF to a target pose (EE normal to face, centered)

This moves tilt correction from TERMINAL to the entire NEAR approach. Most effective for boxes with clear
flat faces visible to the camera.

---

## References

- [ViT-VS: On the Applicability of Pretrained Vision Transformer Features for Generalizable Visual Servoing](https://arxiv.org/abs/2503.04545) — Scherl et al., IROS 2025
- [A Training-Free Framework for Precise Mobile Manipulation of Small Everyday Objects (SVM)](https://arxiv.org/abs/2502.13964) — Feb 2025
- [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://nvlabs.github.io/FoundationPose/) — NVLabs, CVPR 2024 Highlight
- [A Rapid Deployment Pipeline for Autonomous Humanoid Grasping Based on Foundation Models](https://arxiv.org/abs/2604.17258) — FoundationPose + SAM pipeline
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) — Chi et al., IJRR 2025
- [3D Diffusion Policy (DP3)](https://3d-diffusion-policy.github.io/) — RSS 2024
- [GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence](https://github.com/nv-nguyen/gigapose) — CVPR 2024
- [High-Precision Transformer-Based Visual Servoing for Humanoid Robots in Aligning Tiny Objects](https://arxiv.org/abs/2503.04862) — 2025
- [Robot Closed-Loop Grasping Based on Deep Visual Servoing Feature Network (DVSFN)](https://www.mdpi.com/2076-0825/14/1/25) — Jan 2025
- [Visual Servoing — Wikipedia overview of IBVS vs PBVS](https://en.wikipedia.org/wiki/Visual_servoing)
