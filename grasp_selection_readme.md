# README: Reference-Image-Based Box Segmentation, Grasp Point Selection, and Surface Normal Estimation

## Goal

Build a standalone perception script that uses:

```text
1 reference image from masked_objects/
+
live RGB-D input from ZED Mini stereo camera
+
SAM3 segmentation
```

to identify the target box in the scene, segment it correctly, estimate a suction grasp point, estimate the surface normal at that grasp point, and visualize/verify the result.

This script is only for **perception validation**. It should not move the robot yet.

---

## 1. Problem Setup

### Input

The script should take:

```text
1. Reference image of the target object
   Path: masked_objects/<target_reference>.png

2. Live RGB image from ZED Mini

3. Live depth map from ZED Mini

4. ZED camera intrinsics

5. Optional camera-to-robot-base transform
```

The target object is a **box placed in the scene**.

The user gives only **one reference image**, so the reference image should be used mainly for:

```text
target identity selection
```

not for full 6D pose estimation.

The geometric outputs should come from:

```text
SAM3 mask + ZED depth
```

---

## 2. Desired Output

For each frame, the script should output:

```text
1. Selected target mask
2. Target object centroid in image
3. Target object 3D point cloud
4. Fitted visible box plane
5. Grasp point in image coordinates
6. Grasp point in camera-frame 3D coordinates
7. Surface normal in camera frame
8. Surface normal visualization
9. Quality/confidence metrics
10. Debug overlay image/video
11. JSON/JSONL log
```

Example final per-frame result:

```text
target_found: true
selected_mask_id: 2
reference_similarity: 0.81
centroid_2d: [421, 286]
grasp_point_2d: [418, 291]
grasp_point_cam: [0.04, -0.03, 0.31]
normal_cam: [-0.12, 0.04, -0.99]
plane_inlier_ratio: 0.72
depth_valid_ratio: 0.86
normal_valid: true
```

---

## 3. High-Level Pipeline

```text
Load reference image from masked_objects/
        ↓
Extract reference DINOv2 embedding (on pre-masked reference)
        ↓
Capture RGB-D frame from ZED Mini (or load saved frame for offline testing)
        ↓
Run SAM3 on live RGB frame (Path A: specific prompt / Path B: generic prompt)
        ↓
Generate candidate object masks
        ↓
Compare each candidate masked crop with reference embedding (DINOv2 similarity)
        ↓
Select target mask (highest similarity above threshold)
        ↓
Clean mask → produce raw_selected_mask / clean_mask / eroded_mask
        ↓
Backproject eroded_mask pixels using depth → 3D point cloud
        ↓
Fit RANSAC plane to 3D points (fit_plane_ransac) → (normal, centroid) or None
        ↓
Normal sign correction → dot(normal, -grasp_point_cam) > 0
        ↓
Normal sanity checks → Z-component, tilt angle, centroid location
        ↓
[Video/live only] Temporal EMA smoothing on normal (Section 12.5)
        ↓
Grasp point = plane inlier centroid (project_3d_to_pixel for 2D)
        ↓
Visualize: 2D overlay (OpenCV) + 3D cloud (Plotly offline / Open3D live)
        ↓
Save overlay + logs
```

---

## 4. Folder Structure

Recommended structure:

```text
project/
│
├── masked_objects/
│   └── target_box.png
│
├── data/
│   ├── rgb/
│   ├── depth/
│   └── calibration/
│
├── outputs/
│   ├── overlays/
│   ├── masks/
│   ├── pointclouds/
│   ├── videos/
│   └── logs/
│
├── scripts/
│   └── check_grasp_normal.py
│
└── README.md
```

For this stage, the main script should be conceptually:

```text
scripts/check_grasp_normal.py
```

but the README should not depend on a fixed name.

---

## 5. Stage 1: Load Reference Image

### Purpose

The reference image tells the system:

```text
Which box/object should I select?
```

It should not be used as the only source of object pose or normal estimation.

### Steps

1. Load the reference image from:

```text
masked_objects/
```

2. Preprocess the reference image.

3. Extract a visual feature/embedding using a feature model such as:

```text
DINOv2
CLIP
SAM-compatible image features
```

Recommended for your setup:

```text
DINOv2 for reference-image matching
SAM3 for segmentation
```

### Output

```text
reference_embedding
reference_image_debug_view
```

### Important Notes

Since there is only one reference image:

```text
Do not expect robust pose estimation from the reference image alone.
Use it only to select the correct object mask.
```

If the live view is very different from the reference view, similarity may drop. That is okay as long as the geometry module uses depth after the correct mask is selected.

---

## 6. Stage 2: Capture RGB-D Frame from ZED Mini

### Purpose

Use the ZED Mini stereo camera to provide:

```text
RGB image
depth image
camera intrinsics
```

### Required Data

For each frame:

```text
RGB image: H × W × 3
Depth map: H × W
Camera intrinsics:
    fx, fy, cx, cy
```

The depth should be aligned to the RGB image.

### Checks

Before using the frame, verify:

```text
1. RGB and depth have the same resolution or are properly aligned.
2. Depth values are in known units: meters or millimeters.
3. Invalid depth pixels are marked clearly.
4. The box is visible in the RGB image.
5. The box has enough valid depth pixels.
```

### Intrinsics Loading

Two modes depending on whether ZED is attached:

**1. Static / offline testing (no ZED attached):**

Use `DEFAULT_INTRINSICS` from `foundation_model/servo_lastmile.py`:

```text
fx=700.0, fy=700.0, cx=640.0, cy=360.0, width=1280, height=720
```

These match a typical ZED Mini at HD720 resolution.
Override with `--intrinsics path/to/intrinsics.json` if available.

**2. Live ZED mode:**

```text
cam_info = sl.Camera().get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
K = {
    "fx": calib.left_cam.fx,
    "fy": calib.left_cam.fy,
    "cx": calib.left_cam.cx,
    "cy": calib.left_cam.cy,
}
```

ZED depth is aligned to the left camera by default.

Always log the intrinsics used at run start. Never hard-code values in the script body.

### Common ZED Depth Issues

Watch out for:

```text
missing depth on shiny surfaces
noisy depth at object edges
holes near box boundaries
depth mismatch if RGB/depth are not aligned
scale confusion between meters and millimeters
```

---

## 7. Stage 3: Run SAM3 to Generate Candidate Masks

### Purpose

SAM3 (Grounding DINO + SAM2) segments objects in the live RGB frame.

SAM3 behavior depends on the prompt type. There are two operating paths:

### Path A — Specific Prompt (default for initial testing)

```text
prompt = "<object name>"   e.g., "protein_bar", "cheez_it_box"
    → Grounding DINO returns single best-scoring detection bbox
    → SAM2 segments that bbox
    → DINOv2 similarity used as acceptance gate (threshold: > 0.6)
    → If similarity < 0.6 → reject frame, log status = LOW_REF_SIMILARITY
```

This is the default path. It is simpler and works well when the object
name is specific and the object is the most prominent match in the scene.

### Path B — Generic Prompt (for harder scenes)

```text
prompt = "box" or "object"
    → Grounding DINO may return multiple candidate bboxes
    → SAM2 segments each candidate
    → DINOv2 similarity ranks all candidates
    → Select candidate with highest similarity above threshold
```

Use Path B when:
- The object name is too generic for Grounding DINO to localize reliably
- The scene is cluttered with similar-looking objects

### Comparison experiment (to choose between Path A and B):

Run both paths on 5–10 test frames and compare:

```text
metric                    Path A     Path B
correct object selected?  manual     manual
reference_similarity      log        log
frames rejected           count      count
```

Pick Path A unless it fails on more than 20% of frames.

### After SAM3: Per-mask Descriptors

For each candidate mask, compute:

```text
mask area
bounding box
mask centroid
masked crop (for DINOv2 matching — see Stage 4)
depth statistics
border contact
```

### Basic Mask Rejection Rules

Reject masks if:

```text
mask area is too small
mask area is too large
mask touches too much of the image border
mask has too little valid depth
mask is mostly background/table
mask is disconnected or fragmented
```

For your setup, useful first thresholds:

```text
area: 5% to 45% of image depending on distance
depth: target should be within expected workspace range
valid depth ratio: > 50%
```

These thresholds can be adjusted after visual debugging.

---

## 8. Stage 4: Match SAM3 Candidate Masks to Reference Image

### Purpose

Because the user gives a reference image, the script should decide:

```text
Which SAM3 mask corresponds to the reference object?
```

### Matching Pipeline

For every SAM3 mask:

```text
candidate mask
    ↓
crop RGB region around the mask bounding box
    ↓
zero out (or white-fill) pixels outside the mask [REQUIRED — not optional]
    ↓
extract DINOv2 candidate feature from masked crop
    ↓
compare with reference feature (extracted from pre-masked reference image)
    ↓
compute cosine similarity score
```

**Why masking the crop is required:**
DINOv2 patch tokens are spatially sensitive. Background pixels in the crop
shift the embedding away from the object and degrade similarity scores.
The reference images in `masked_objects/` already have white backgrounds.
The live crop must be masked to match.

**Comparison experiment (masked vs. unmasked crop):**
On 5 test frames, extract similarity with and without masking and compare:

```text
frame     masked_sim     unmasked_sim     correct_selection?
1         0.82           0.67             masked=yes, unmasked=yes
2         0.79           0.58             masked=yes, unmasked=maybe
...
```

Use masked crop in all final runs.

### Selection Rule

Choose:

```text
selected_mask = candidate mask with highest reference similarity
```

The selected mask should also satisfy basic validity constraints.

### Recommended Score

Use a combined score:

```text
final_score =
    reference_similarity
    + mask_quality_score
    + depth_quality_score
    - border_penalty
    - size_penalty
```

For the first version, you can simply use:

```text
highest reference similarity among valid masks
```

### Output

```text
selected_mask
selected_mask_id
reference_similarity
candidate_scores
```

### Verification

Visualize all candidate masks with their similarity scores.

This is important because if the wrong mask is selected, every later step will be wrong.

---

## 9. Stage 5: Clean the Selected Mask

### Purpose

The SAM3 mask may include noisy edges, holes, or extra parts. Clean it before depth backprojection.

### Cleaning Operations

Apply simple mask cleanup:

```text
remove small disconnected components
fill small holes
smooth jagged edges
keep largest connected component if needed
```

### Why Keep Two Masks?

The pipeline uses two different masks for two different purposes:

```text
raw_selected_mask → visualization only (outline on RGB overlay)
eroded_mask       → backprojection + plane fitting
```

Never feed the raw mask into plane fitting — boundary pixels have noisy
ZED depth and will degrade the normal estimate.

### Erosion Parameters

```text
kernel_size = 7   # starting value; tune based on observed ZED edge noise width
```

Apply morphological erosion to `clean_mask` to get `eroded_mask`.
If the eroded mask loses too much area (< 30% of clean_mask), reduce kernel_size.
If depth on eroded mask is still noisy at edges, increase kernel_size.

### Output (three masks, not one)

```text
raw_selected_mask   — original SAM3 output; used for visualization overlay only
clean_mask          — morphologically cleaned (hole fill + largest component)
eroded_mask         — clean_mask eroded by kernel_size; used for all depth/plane steps
mask_centroid_2d    — centroid of clean_mask (for display only)
mask_area           — pixel count of clean_mask
```

---

## 10. Stage 6: Backproject Mask Pixels to 3D

### Purpose

Convert mask pixels into a 3D point cloud using ZED depth.

For each pixel inside the selected mask:

```text
u, v, depth
    ↓
x, y, z in camera frame
```

### Camera Backprojection Concept

Given:

```text
u, v = image pixel
z = depth
fx, fy, cx, cy = camera intrinsics
```

The 3D point is:

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth
```

### Filtering Rules

Use `eroded_mask` (not raw_selected_mask or clean_mask) for backprojection.
This excludes boundary pixels where ZED stereo depth is unreliable.

Keep only points that satisfy:

```text
pixel is inside eroded_mask
depth is valid (not NaN, not 0)
depth is finite
depth is within workspace range (e.g., 100mm – 1500mm)
```

### Output

```text
target_point_cloud_cam
valid_depth_ratio
number_of_3d_points
```

### Verification

Visualize:

```text
selected mask over RGB image
valid depth pixels inside mask
3D point cloud in Open3D
```

---

## 11. Stage 7: Fit the Visible Box Plane

### Purpose

A box has flat faces. For suction, we want the visible planar face.

Use the 3D points inside the mask to fit a dominant plane.

### Plane Model

Represent the plane as:

```text
a x + b y + c z + d = 0
```

The surface normal is:

```text
n = [a, b, c]
```

### Recommended Method

Use `fit_plane_ransac` from `foundation_model/servo_lastmile.py`.

```text
fit_plane_ransac(points_xyz, thresh_mm, max_iter, rng) → (normal, centroid) or None
```

Input: all valid 3D points from `eroded_mask` backprojection.
RANSAC is appropriate because the mask may still contain:

```text
depth noise
edge points
partial background
table pixels
non-planar artifacts
```

### RANSAC Parameters (starting values)

```text
thresh_mm = 8.0        # ZED Mini stereo noise ~3-5mm at 0.5m
max_iter = 200
min_inlier_ratio = 0.35
min_inlier_count = 50
```

**Tuning experiment:** Run with `thresh_mm` in {5, 8, 12} on the same test frames.
Record `plane_inlier_ratio` and `normal_delta_deg` for each value.
Pick the smallest thresh_mm that achieves inlier_ratio > 0.5 consistently.

### Failure Handling (fit_plane_ransac returns None)

`fit_plane_ransac` returns `None` when:

```text
fewer than 3 valid 3D points in input
inlier_ratio < min_inlier_ratio after all iterations
inlier count < min_inlier_count
```

On failure, the pipeline must:

```text
1. Set plane_found = False
2. Set normal_valid = False
3. Set grasp_point_valid = False
4. Log status = BAD_PLANE
5. Skip normal arrow in visualization
6. Do NOT fall back to mask centroid as grasp point
   (centroid without plane fit gives wrong normal → unsafe for suction)
7. Optional: retry once with thresh_mm * 1.5; log if retry succeeds
```

### Plane Fitting Output

```text
plane_found          — bool
normal               — unit vector (3,) if plane_found else None
centroid             — 3D mean of inliers (3,) if plane_found else None
plane_inlier_points  — array of inlier 3D points
plane_inlier_ratio   — fraction of input points that are inliers
plane_residual_error — mean distance of inliers to fitted plane (mm)
```

### Plane Quality Checks

Accept the plane only if:

```text
plane_inlier_ratio > 0.35   (hard minimum; target > 0.5)
number_of_inliers > 50
plane_residual_error < 5mm
```

These thresholds should be tuned using your ZED depth quality.

---

## 12. Stage 8: Estimate Surface Normal

### Purpose

The surface normal tells the controller:

```text
How should the suction cup orient itself?
```

For suction, the gripper should approach approximately perpendicular to the box face.

### Surface Normal

From the fitted plane:

```text
normal = [a, b, c]
```

Normalize:

```text
normal = normal / ||normal||
```

### Normal Sign Ambiguity

Plane fitting gives two valid normals:

```text
n
-n
```

Both are mathematically correct, but only one is useful for suction.

### Recommended Sign Convention

Force the normal to point toward the camera.

For a camera-frame point:

```text
p_grasp_cam
```

the direction from the grasp point to the camera is:

```text
view_direction = camera_origin - p_grasp_cam
```

Since camera origin is usually:

```text
[0, 0, 0]
```

then:

```text
view_direction = -p_grasp_cam
```

If the estimated normal points away from the camera, flip it:

```text
if dot(normal_cam, -grasp_point_cam) < 0:
    normal_cam = -normal_cam
```

### Normal Sanity Checks (required before accepting)

Run these after sign correction:

**1. Z-component check:**

```text
# In camera frame, Z+ points into the scene.
# A normal pointing toward the camera must have negative Z.
if abs(normal_cam[2]) < 0.1:
    # Normal is nearly perpendicular to optical axis
    # → likely fitted the table or a near-vertical side face
    → set normal_valid = False, status = BAD_PLANE
```

**2. Tilt angle check:**

```text
tilt_deg = degrees(arccos(abs(dot(normal_cam, [0, 0, -1]))))

if tilt_deg > 45:
    → log status = STEEP_TILT (warning, do not hard-reject)
    → suction still possible up to ~30° tilt
    → if tilt_deg > 60, also log LOW_SUCTION_CONFIDENCE
```

**3. Plane centroid location check:**

```text
project centroid (from fit_plane_ransac) back to 2D with project_3d_to_pixel()
if centroid_2d is outside raw_selected_mask bounding box:
    → plane was fitted to background points
    → set normal_valid = False, status = BAD_PLANE
```

### Output

```text
surface_normal_cam    — unit vector (3,), sign-corrected and sanity-checked
normal_valid          — bool
tilt_deg              — float, degrees from optical axis
normal_confidence     — "GOOD" | "STEEP_TILT" | "LOW_SUCTION_CONFIDENCE" | "BAD_PLANE"
```

### Interpretation

If the visible face is facing the camera, the normal should point approximately toward the camera.

If the box is tilted, the normal should tilt with the visible face.

If the top face is visible, the normal should point outward from the top surface.

---

## 12.5 Temporal Normal Smoothing (live/video mode only)

Skip this section for single-frame static testing. Activate when processing
video sequences or running with live ZED input.

### EMA Filter

Apply an Exponential Moving Average on the surface normal after sign correction
and sanity checks:

```text
alpha = 0.3   # smoothing factor: 0 = fully frozen, 1 = no smoothing

normal_smooth = alpha * normal_new + (1 - alpha) * normal_prev
normal_smooth = normalize(normal_smooth)
```

Apply before grasp point projection.

### Update Rules

Skip EMA update (reuse previous normal) if:

```text
plane_found is False
plane_inlier_ratio < min_inlier_ratio
normal_delta_deg > 30°   (outlier frame)
```

Log whether EMA was applied or skipped each frame.

### Tuning Experiment

Run with `alpha` in {0.2, 0.3, 0.5} on a 30-frame video of a stationary box.
Plot `normal_delta_deg` per frame for each alpha.
Pick the smallest alpha that keeps delta < 3° on a stationary object.

---

## 13. Stage 9: Select the Suction Grasp Point

### Purpose

The grasp point should be a point on the visible box face where the suction cup can make stable contact.

It should be:

```text
on the target box
on a flat planar region
away from edges
inside the mask
with valid depth
near the center of the visible face
large enough for the suction cup
```

### Grasp Point Definition (v1)

```text
grasp_point_3d = centroid returned by fit_plane_ransac()
```

`fit_plane_ransac()` in `foundation_model/servo_lastmile.py` returns
`(normal, centroid)` where `centroid = mean of inlier 3D points`.
Use that directly — no recomputation needed.

```text
grasp_point_2d = project_3d_to_pixel(grasp_point_3d, K)
```

Use `project_3d_to_pixel()` from `foundation_model/servo_lastmile.py`.

**Why centroid and not median:**
Centroid is the geometric center of the observed planar patch and is more
stable under ZED noise than the median on asymmetric inlier sets.

**Future improvement (Phase 2):**
Suction affordance scoring over candidate points (distance from boundary,
local flatness, suction cup radius clearance). Not needed for v1.

### Output

```text
grasp_point_2d
grasp_point_3d_cam
grasp_point_valid
grasp_quality_score
```

---

## 14. Stage 10: Optional Transform to Robot Base Frame

For the first visualization script, camera-frame output is enough.

But if you want to prepare for control, transform:

```text
grasp_point_cam → grasp_point_base
normal_cam → normal_base
```

using the camera-to-robot-base transform.

### Position Transform

```text
p_base = R_base_cam * p_cam + t_base_cam
```

### Normal Transform

Normals use rotation only:

```text
n_base = R_base_cam * n_cam
```

Do not add translation to normals.

### Output

```text
grasp_point_base
surface_normal_base
```

---

## 15. Visualizations to Add

You should add multiple visualization modes because each one catches different errors.

### Visualization Backends

Two backends are used depending on operating mode:

| Mode           | Backend | When to use                                         |
|----------------|---------|-----------------------------------------------------|
| Offline/static | Plotly  | Testing with saved RGB-D images, no ZED attached    |
| Live/robot     | Open3D  | ZED camera attached, preparing for servo integration|

**Plotly (offline):**
- Produces static interactive HTML — open in browser, no display server required
- Best for: initial debugging, sharing results, comparing across frames
- Use for Visualization 9 (3D point cloud) and Visualization 10 (plane patch)

**Open3D (live):**
- Real-time 3D window that updates per frame
- Requires display (X11/Wayland)
- Use for Visualization 9 and 10 in live mode

**2D overlay (Visualization 7, 8, 11):**
Always rendered with OpenCV regardless of backend. Both modes produce the same overlay PNG.

---

### Visualization 1: Reference Image Display

Show:

```text
reference image
reference mask if available
reference embedding status
```

Purpose:

```text
Verify the script loaded the correct reference object.
```

---

### Visualization 2: SAM3 Candidate Masks

Show all SAM3 candidate masks over the RGB frame.

For each mask, display:

```text
mask id
area percentage
reference similarity
valid depth ratio
selected/not selected
```

Purpose:

```text
Verify that SAM3 generated the correct target mask.
Verify that reference matching selected the correct object.
```

Expected result:

```text
The target box mask should have the highest similarity score.
```

---

### Visualization 3: Selected Mask Overlay

Overlay the selected mask on the RGB image.

Display:

```text
selected mask outline
mask centroid
mask area
reference similarity
```

Purpose:

```text
Verify the target object is segmented correctly.
```

Correct behavior:

```text
mask covers the visible box surface/object
mask does not include large background/table areas
mask is not missing most of the box
```

---

### Visualization 4: Mask Stages Overlay

Show three masks side by side: raw / clean / eroded.

```text
Panel 1: raw_selected_mask   — original SAM3 output
Panel 2: clean_mask          — after hole fill + largest component
Panel 3: eroded_mask         — after erosion (kernel_size=7px)
```

Purpose:

```text
Verify that cleanup removes noise without destroying the object mask.
Verify erosion removes boundary pixels without losing the face interior.
```

Correct behavior:

```text
raw mask: box body + possibly small fragments
clean mask: box body only
eroded mask: box body interior, visibly smaller by ~kernel_size pixels at each edge
```

---

### Visualization 5: Valid Depth Pixels Inside Mask

Show only pixels inside the mask that have valid depth.

Purpose:

```text
Verify ZED depth is usable for the target object.
```

Correct behavior:

```text
most of the visible box face should have valid depth
missing depth should not dominate the selected face
```

Failure cases:

```text
too many holes
depth on background instead of object
bad depth around object boundaries
```

---

### Visualization 6: Plane Inlier Overlay on Image

Project the RANSAC plane inlier points back onto the image.

Show:

```text
cyan pixels = plane inliers
red pixels = rejected/outlier mask points
```

Purpose:

```text
Verify the plane is fitted to the correct visible box face.
```

Correct behavior:

```text
inliers should lie on one flat face of the box
outliers should mostly be edges/noisy pixels/background
```

Bad behavior:

```text
inliers spread across table and box
inliers lie on wrong object
inliers are mostly on box edge
```

---

### Visualization 7: Grasp Point Overlay

Draw the grasp point on the RGB image.

Recommended colors:

```text
blue dot = grasp point
red cross = mask centroid
yellow outline = selected mask
```

Purpose:

```text
Verify the selected suction point is reasonable.
```

Correct behavior:

```text
grasp point is inside the box mask
grasp point is near the center of a flat face
grasp point is away from edges
grasp point is not on background/table
```

---

### Visualization 8: Surface Normal Arrow on RGB Image

Draw the normal as an arrow projected into image space.

**Arrow scale (named parameter, not magic number):**

```text
arrow_length_mm = 60   # tune if arrow appears too short (<30px) or too long (>200px)

tip_3d = grasp_point_3d + normal_cam * arrow_length_mm
grasp_point_2d = project_3d_to_pixel(grasp_point_3d, K)
tip_2d          = project_3d_to_pixel(tip_3d, K)

cv2.arrowedLine(img, grasp_point_2d, tip_2d,
                color=(0, 255, 0), thickness=3, tipLength=0.2)
```

Use `project_3d_to_pixel()` from `foundation_model/servo_lastmile.py`.

At 300mm working distance with fx≈700, a 60mm Z-shift projects to ~140px — visible
but not overwhelming. Scale up to 80mm if the object is further away.

Recommended colors:

```text
green arrow  = surface normal  (0, 255, 0) BGR
red cross    = grasp point     (0, 0, 255) BGR   [cv2.drawMarker MARKER_CROSS]
yellow outline = selected mask boundary
```

Purpose:

```text
Verify the normal visually points perpendicular to the visible face.
```

Correct behavior:

```text
front face → arrow points toward camera
tilted face → arrow tilts with face
top face → arrow points out from top surface
```

Bad behavior:

```text
arrow lies along the box surface
arrow points into the object when it should point outward
arrow flips direction frame to frame
arrow does not move consistently when the box rotates
```

---

### Visualization 9: 3D Point Cloud with Normal

Use a 3D viewer such as Open3D.

Display:

```text
target point cloud
plane inlier points
grasp point sphere
normal arrow
camera frame
```

Purpose:

```text
Best way to verify surface normal and plane fitting.
```

Correct behavior:

```text
normal arrow is perpendicular to the visible point cloud plane
grasp point lies on the plane
plane inliers form a flat patch on the box
```

---

### Visualization 10: Plane Patch Visualization

Visualize the fitted plane as a semi-transparent patch in 3D.

Show:

```text
box point cloud
fitted plane patch
normal arrow
grasp point
```

Purpose:

```text
Verify the plane model matches the visible box face.
```

---

### Visualization 11: Per-Frame Debug Text

Overlay text on the image:

```text
target_found: true/false
ref_similarity: 0.81
mask_area: 12.4%
valid_depth_ratio: 0.86
plane_inlier_ratio: 0.72
mean_plane_error: 0.003 m
normal_cam: [nx, ny, nz]
grasp_point_cam: [x, y, z]
normal_valid: true/false
```

Purpose:

```text
Quick debugging without opening logs.
```

---

### Visualization 12: Temporal Stability Plot

If processing video, plot over time:

```text
reference similarity
plane inlier ratio
grasp point x/y/z
normal nx/ny/nz
normal angle change between frames
valid depth ratio
```

Purpose:

```text
Verify that the grasp point and normal are stable.
```

A good result:

```text
normal does not jump suddenly
grasp point does not jitter badly
plane quality remains high
```

---

## 16. Verification Methods

You should verify both:

```text
grasp point correctness
surface normal correctness
```

### A. Verifying Correct Object Segmentation

Check:

```text
1. Is the selected mask on the reference object?
2. Does the mask cover the visible box?
3. Does the mask exclude the table/background?
4. Is the mask stable across frames?
5. Does the selected mask remain the same if other objects are nearby?
```

Use:

```text
candidate mask visualization
selected mask overlay
reference similarity scores
```

---

### B. Verifying Grasp Point Correctness

The grasp point is correct if:

```text
1. It lies inside the selected mask.
2. It lies on the fitted plane inliers.
3. It is not near the mask boundary.
4. It is on a flat visible surface.
5. It is reachable by the suction cup.
6. It has valid depth.
7. Its surrounding patch is large enough for the suction cup.
```

Visual checks:

```text
blue dot on RGB overlay
blue dot in 3D point cloud
distance from mask boundary
local planar patch visualization
```

Failure cases:

```text
grasp point on edge
grasp point on background
grasp point on non-flat region
grasp point on noisy depth hole
grasp point jumps between frames
```

---

### C. Verifying Surface Normal Correctness

The surface normal is correct if:

```text
1. It is perpendicular to the fitted plane.
2. It points outward toward the camera/gripper.
3. It is stable over time.
4. It changes consistently when the box orientation changes.
5. It agrees with visual intuition from the RGB image.
```

Visual checks:

```text
green arrow on image
green arrow in 3D point cloud
normal vector printed in overlay
normal angle over time
```

Manual check examples:

```text
If the box front face faces camera:
    normal should point roughly toward camera.

If the box is rotated left:
    normal should tilt left/right according to the visible face.

If the top face is selected:
    normal should point upward/outward from the top face.

If the box is slanted:
    normal should be perpendicular to the slanted surface.
```

---

### D. Verifying Normal Sign

Since both `n` and `-n` are mathematically valid, verify sign separately.

For suction, the normal should usually point:

```text
from box surface outward toward camera/gripper
```

Check:

```text
dot(normal, view_direction) > 0
```

where:

```text
view_direction = direction from grasp point to camera
```

If this is negative, flip the normal.

Visual sign check:

```text
The green arrow should come out of the box face, not go into it.
```

---

### E. Verifying Plane Quality

Plane quality is good if:

```text
1. Plane inliers cover most of the visible box face.
2. Plane residual error is low.
3. Inlier ratio is high.
4. Normal is stable.
5. Plane does not include table/background.
```

Useful metrics:

```text
plane_inlier_ratio
mean_plane_error
median_plane_error
number_of_inlier_points
depth_valid_ratio
normal_change_deg
```

---

## 17. Quality Metrics to Log

For every frame, log:

```text
frame_id
timestamp
target_found
selected_mask_id
reference_similarity
mask_area_percent
mask_centroid_2d
valid_depth_ratio
num_3d_points
plane_found
plane_inlier_ratio
plane_mean_error
plane_equation
grasp_point_2d
grasp_point_cam
surface_normal_cam
normal_flipped
normal_valid
normal_confidence
```

If robot calibration is available, also log:

```text
grasp_point_base
surface_normal_base
```

If processing video, also log temporal metrics:

```text
grasp_point_delta_mm
normal_delta_deg
reference_similarity_delta
plane_quality_delta
```

---

## 18. Acceptance Criteria for First Script

The first version is successful if:

```text
1. The reference image is loaded correctly.
2. SAM3 generates candidate masks.
3. The correct target box mask is selected using the reference image.
4. The selected mask is visually correct.
5. Valid depth points are extracted inside the mask.
6. A dominant plane is fitted to the visible box face.
7. The surface normal is visualized as an arrow.
8. The grasp point is visualized as a point on the box face.
9. The normal points roughly perpendicular to the face.
10. The results are saved as overlay images/video and JSON logs.
```

Do not require robot control yet.

---

## 19. Suggested Development Order

Implement and test in this order:

```text
1. Load one reference image from masked_objects/.
2. Capture or load one RGB-D frame from ZED Mini.
3. Run SAM3 and visualize all candidate masks.
4. Add reference-image matching to select the target mask.
5. Visualize selected target mask and reference similarity.
6. Clean selected mask.
7. Visualize valid depth pixels inside selected mask.
8. Backproject mask pixels into 3D point cloud.
9. Visualize target point cloud in 3D.
10. Fit RANSAC plane to the target point cloud.
11. Visualize plane inliers on the RGB image.
12. Estimate surface normal from fitted plane.
13. Fix normal sign toward the camera.
14. Select grasp point from plane inliers.
15. Draw grasp point on RGB image.
16. Draw surface normal arrow on RGB image.
17. Visualize point cloud + grasp point + normal in 3D.
18. Save per-frame JSON output.
19. Run the script on multiple box orientations.
20. Check temporal stability on a short video sequence.
```

---

## 20. Testing Scenarios

Use a simple staged test set.

### Test 1: Single Box, Front Face Visible

Expected:

```text
mask selects box
grasp point near center of front face
normal points toward camera
```

### Test 2: Box Rotated Left/Right

Expected:

```text
mask selects box
grasp point on visible face
normal tilts according to visible face
```

### Test 3: Box Tilted

Expected:

```text
plane follows tilted surface
normal remains perpendicular to tilted face
```

### Test 4: Box Near Image Boundary

Expected:

```text
mask may be rejected if too close to border
plane quality may decrease
script should report low confidence
```

### Test 5: Multiple Objects in Scene

Expected:

```text
SAM3 may produce multiple masks
reference matching should select the target box
```

### Test 6: Poor Depth Region

Expected:

```text
valid_depth_ratio decreases
plane fit confidence decreases
script should not output high-confidence normal
```

### Test 7: Different Box Orientation from Reference Image

Expected:

```text
reference similarity may drop
but if correct mask is selected, depth-plane fitting should still estimate normal
```

---

## 21. Failure Modes and Debugging

### Failure: Wrong Mask Selected

Likely causes:

```text
reference embedding not discriminative
SAM3 candidate masks poor
object view too different from reference
background included in crop
```

Debug with:

```text
candidate mask visualization
similarity scores
masked crop display
```

Possible fixes:

```text
use masked crop instead of bounding-box crop
combine reference similarity with depth/area constraints
use DINOv2 features instead of color-only features
allow user click initialization if needed
```

---

### Failure: Mask Includes Table

Likely causes:

```text
SAM3 over-segmentation
box/table boundary unclear
shadow or texture confusion
```

Fixes:

```text
erode mask before depth backprojection
remove points with table-plane geometry
use largest object component
filter by depth discontinuity
```

---

### Failure: Plane Fits the Table Instead of Box

Likely causes:

```text
mask includes table
box depth is missing
table has more inliers than box face
```

Fixes:

```text
clean mask
remove background/table points
use mask interior only
reject planes with normal similar to table normal if known
require plane centroid to lie near mask center
```

---

### Failure: Normal Points Opposite Direction

Cause:

```text
plane normal sign ambiguity
```

Fix:

```text
flip normal to point toward camera/gripper
```

Verification:

```text
green arrow should come out of the visible box face
```

---

### Failure: Normal Jitters Between Frames

Likely causes:

```text
depth noise
plane fitting instability
mask flicker
normal sign flips
```

Fixes:

```text
normal sign correction
temporal smoothing
require minimum plane confidence
reuse last reliable normal when current estimate is poor
```

---

### Failure: Grasp Point Is Near Edge

Likely causes:

```text
using raw centroid of irregular mask
plane inliers concentrated near edge
partial view
```

Fixes:

```text
use distance transform inside mask
prefer points far from boundary
require suction cup radius clearance
use center of largest planar patch
```

---

## 22. Recommended Confidence Rules

The script should mark output as low confidence if:

```text
reference_similarity < threshold
valid_depth_ratio < threshold
plane_inlier_ratio < threshold
number_of_plane_inliers too small
grasp point near mask boundary
normal changes too much from previous frame
mask touches image border
```

Example status labels:

```text
GOOD
LOW_REF_SIMILARITY
BAD_DEPTH
BAD_PLANE
GRASP_NEAR_EDGE
NORMAL_UNSTABLE
NO_TARGET
```

---

## 23. Final Output Files

For each run, save:

```text
outputs/overlays/frame_XXXX_overlay.png
outputs/masks/frame_XXXX_mask.png
outputs/pointclouds/frame_XXXX_target.ply
outputs/logs/run.jsonl
outputs/videos/debug_overlay.mp4
```

Each overlay should include:

```text
selected mask
grasp point
surface normal arrow
plane inliers
debug text
```

---

## 24. What This Script Should Not Do Yet

Do not add robot motion in this script.

Avoid:

```text
servoing
suction activation
robot approach
real-time control
closed-loop manipulation
```

This script should only answer:

```text
Can I correctly identify the target box?
Can I segment it?
Can I choose a reasonable suction grasp point?
Can I estimate and visualize the surface normal?
Can I verify that the normal is correct?
```

Once this works reliably, then integrate it into the full FAR/NEAR/TERMINAL servoing pipeline.

---

## 25. Summary

The first script should implement:

```text
one reference image
    ↓
SAM3 candidate segmentation
    ↓
reference-image mask selection
    ↓
mask cleanup
    ↓
ZED depth backprojection
    ↓
3D target point cloud
    ↓
RANSAC visible-plane fitting
    ↓
surface normal estimation
    ↓
suction grasp point selection
    ↓
2D and 3D visualization
    ↓
quality metrics and logs
```

The reference image identifies **which object**.

The SAM3 mask identifies **where the object is**.

The ZED depth gives **3D geometry**.

The plane fit gives **surface normal**.

The planar patch center gives **suction grasp point**.

The visualizations verify whether the result is correct before using it for robot control.

Version 1 (current):
Full-mask RANSAC on eroded mask interior. Grasp point = plane inlier centroid.

Version 2 (future):
RANSAC dominant plane + restrict grasp point to inliers far from mask boundary.

Version 3 (future):
Patch-based local plane fitting around candidate suction points.

Version 4 (future):
Best suction patch chosen by affordance score.

---

## 25.5 Integration Interface (for future servo_lastmile integration)

When this perception module is integrated into the TERMINAL state of
`full_system_pipeline/pipeline/servo_lastmile_v2_ext.py`, it must expose
a clean callable interface so integration is a copy-paste, not a rewrite.

### GraspResult

```python
@dataclass
class GraspResult:
    target_found: bool
    grasp_point_2d: tuple | None        # (u, v) in pixels
    grasp_point_cam: np.ndarray | None  # shape (3,) in mm, camera frame
    normal_cam: np.ndarray | None       # shape (3,) unit vector
    plane_inlier_ratio: float
    tilt_deg: float
    status: str                         # confidence label (see Section 22)
```

### compute_grasp

```python
def compute_grasp(
    frame_bgr: np.ndarray,          # H×W×3
    depth_mm: np.ndarray,           # H×W, aligned to frame_bgr
    K: dict,                        # fx, fy, cx, cy
    reference_embedding,            # precomputed DINOv2 embedding
    arrow_length_mm: float = 60,
    erode_kernel: int = 7,
    ransac_thresh_mm: float = 8.0,
    alpha_ema: float = 0.3,
) -> GraspResult:
    ...
```

### Design Rules

```text
No ZED SDK calls inside compute_grasp — caller handles capture
No robot control calls — caller handles motion
Returns GraspResult even on failure (target_found=False)
Overlay drawing is separate from compute_grasp (pass result to a draw function)
All parameters have defaults matching the values documented in this README
```

Define this interface at the top of the final script and use it even in
standalone static-testing mode, so the TERMINAL integration is a single import.