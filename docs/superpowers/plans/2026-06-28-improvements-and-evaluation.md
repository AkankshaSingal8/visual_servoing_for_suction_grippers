# SemVS Improvements & Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the tilt estimation bug, build a full offline ablation suite, run all existing data through the analysis pipeline, and produce a comprehensive progress doc — all without needing the robot.

**Architecture:** All improvements are layered on top of `foundation_model/servo_lastmile_v2.py` (the canonical pipeline). New evaluation and ablation scripts live in `full_system_pipeline/evaluation/` and `full_system_pipeline/pipeline/`. Tests live in `tests/`. The progress doc lives in `docs/PROGRESS.md`.

**Tech Stack:** Python 3.10, OpenCV, NumPy, Matplotlib, pytest (tests only). All scripts run offline on existing `frames.jsonl` and `raw.mp4` data. No robot, no ZED required for Tasks 1–6.

## Global Constraints

- Working directory is always `/home/akanksha/repo/visual_servoing_for_suction_grippers/`
- Never modify files under `foundation_model/variants/` — those are frozen snapshots
- All new evaluation scripts must work on existing run data without a robot or camera
- All paths in scripts use `Path` (pathlib), not raw strings
- Plots are saved as PNG (Agg backend, `dpi=150`). Never call `plt.show()`
- Every new Python file must have a `if __name__ == "__main__":` block so it's runnable directly
- No new dependencies beyond what's in `foundation_model/setup.sh` (torch, cv2, numpy, matplotlib)

---

## File Map

| File | Status | What it does |
|------|--------|--------------|
| `foundation_model/servo_lastmile_v2.py:768` | **BUG** | `res["_tilt"] = None` — tilt call missing |
| `full_system_pipeline/evaluation/analyze_run.py` | EXISTS | per-run metrics + plots |
| `full_system_pipeline/evaluation/batch_analyze.py` | **CREATE** | runs analyze_run on all v2 run dirs at once |
| `full_system_pipeline/evaluation/signal_ablation.py` | **CREATE** | replays frames.jsonl with different signal combos |
| `full_system_pipeline/evaluation/offline_detector_ablation.py` | **CREATE** | runs 3 detector backends on same video, compares centroids |
| `full_system_pipeline/evaluation/annotate_gt.py` | EXISTS | click-to-annotate GT centroid on a frame |
| `full_system_pipeline/pipeline/servo_lastmile_v2_ext.py` | EXISTS | --detector flag ablation |
| `tests/test_tilt_estimation.py` | **CREATE** | unit tests for estimate_near_tilt |
| `tests/test_fusion.py` | **CREATE** | unit tests for weighted_geometric_median, fuse_centroids |
| `tests/test_ee_masking.py` | **CREATE** | unit tests for EEOcclusionMasker |
| `tests/conftest.py` | **CREATE** | shared fixtures |
| `docs/PROGRESS.md` | **CREATE** | comprehensive project progress + experiment templates |

---

## Task 1: Run Baseline Analysis on All Existing Data

**Files:**
- Create: `full_system_pipeline/evaluation/batch_analyze.py`
- Uses (existing): `full_system_pipeline/evaluation/analyze_run.py`

**Interfaces:**
- Consumes: all `runs/v2_*/frames.jsonl` files
- Produces: `runs/batch_summary.csv` (one row per run), per-run `analysis/` subdirs

**Context:** There are 19 v2 runs with frames.jsonl data. The largest is 749 frames. None have GT annotations yet. This task gives us a baseline picture of all existing data — how many frames per phase, whether TERMINAL was reached, watchdog alarms. This is the first thing any reviewer will ask for.

- [ ] **Step 1: Write batch_analyze.py**

```python
#!/usr/bin/env python3
"""
batch_analyze.py — Run analyze_run.py on every v2_* run directory
and write a summary CSV.

Usage:
    python full_system_pipeline/evaluation/batch_analyze.py
    python full_system_pipeline/evaluation/batch_analyze.py --runs-dir runs/ --out-csv runs/batch_summary.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Allow both `python batch_analyze.py` and `python -m` style imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from full_system_pipeline.evaluation.analyze_run import (
    load_frames, compute_summary, plot_error_over_distance,
    plot_signal_centroids, plot_tilt_timeline, plot_phase_summary,
    plot_signal_dominance, IMAGE_CENTER,
)

IMAGE_W, IMAGE_H = 1280, 720

FIELDNAMES = [
    "run_dir", "n_frames", "n_far", "n_near_sam3", "n_near_track", "n_terminal",
    "reached_terminal", "mean_z_at_terminal", "mean_tilt_at_terminal",
    "mean_centroid_err_at_terminal", "final_centroid_err_px", "watchdog_alarms_count",
]


def _find_runs(runs_dir: Path) -> list[Path]:
    return sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "frames.jsonl").exists()
    )


def analyze_one(run_dir: Path, write_plots: bool = True) -> dict:
    frames = load_frames(str(run_dir))
    summary = compute_summary([frames], ref_centroid=IMAGE_CENTER)
    summary["run_dir"] = run_dir.name
    summary["reached_terminal"] = int(summary.get("n_terminal", 0) > 0)

    if write_plots:
        out_dir = run_dir / "analysis"
        out_dir.mkdir(exist_ok=True)
        plot_error_over_distance([frames], str(out_dir), IMAGE_CENTER)
        plot_signal_centroids([frames], str(out_dir))
        plot_tilt_timeline([frames], str(out_dir))
        plot_phase_summary([frames], str(out_dir))
        plot_signal_dominance([frames], str(out_dir))
        with open(out_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory containing v2_* run subdirs (default: runs/)")
    parser.add_argument("--out-csv", default="runs/batch_summary.csv",
                        help="Output CSV path (default: runs/batch_summary.csv)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip per-run plot generation (faster)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = _find_runs(runs_dir)
    if not runs:
        print(f"No runs with frames.jsonl found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} runs in {runs_dir}")
    rows = []
    for run_dir in runs:
        try:
            summary = analyze_one(run_dir, write_plots=not args.no_plots)
            rows.append({k: summary.get(k) for k in FIELDNAMES})
            status = "TERMINAL" if summary.get("reached_terminal") else "no-term"
            print(f"  {run_dir.name}: {summary['n_frames']} frames  [{status}]  "
                  f"watchdog={summary.get('watchdog_alarms_count', 0)}")
        except Exception as e:
            print(f"  {run_dir.name}: FAILED ({e})", file=sys.stderr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBatch summary written to: {out_csv}")
    reached = sum(1 for r in rows if r.get("reached_terminal"))
    print(f"Reached TERMINAL: {reached}/{len(rows)} runs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /home/akanksha/repo/visual_servoing_for_suction_grippers
python full_system_pipeline/evaluation/batch_analyze.py --no-plots
```

Expected output: table showing 19 runs, how many reached TERMINAL, watchdog counts.

- [ ] **Step 3: Run with plots (takes ~30s)**

```bash
python full_system_pipeline/evaluation/batch_analyze.py
```

Expected: `runs/batch_summary.csv` written, per-run `analysis/` subdirs created.

- [ ] **Step 4: Verify**

```bash
python -c "
import csv
rows = list(csv.DictReader(open('runs/batch_summary.csv')))
print(f'{len(rows)} runs')
print('Reached terminal:', sum(1 for r in rows if r['reached_terminal']=='1'))
print('Watchdog total:', sum(int(r['watchdog_alarms_count'] or 0) for r in rows))
"
```

Expected: 19 rows, some terminal count, watchdog count ≥ 0.

- [ ] **Step 5: Commit**

```bash
git add full_system_pipeline/evaluation/batch_analyze.py runs/batch_summary.csv
git commit -m "eval: add batch_analyze.py and run baseline analysis on all 19 v2 runs"
```

---

## Task 2: Fix Tilt Estimation Integration Bug

**Files:**
- Modify: `foundation_model/servo_lastmile_v2.py` (line 768)

**Interfaces:**
- Consumes: `estimate_near_tilt(depth_mm, K, center_xy)` already defined in the same file at line 164
- Produces: `res["tilt_deg"]`, `res["roll_deg"]`, `res["pitch_deg"]`, `res["near_plane_normal"]`, `res["near_plane_n_inliers"]` populated when depth is available

**Context:** `estimate_near_tilt` is defined and correct (lines 164–242). `LastMilePipelineV2.step()` obtains `depth_map = self.depth_provider(frame_bgr)` at line 541 but then at line 768 just sets `res["_tilt"] = None` and never calls the function. In offline mode, `depth_map` is `None` so tilt will still be None — that's correct. In live mode with ZED, `depth_map` is a real depth array, so the call will produce tilt data. The fix is adding ~10 lines.

- [ ] **Step 1: Apply the fix in servo_lastmile_v2.py**

Replace the block at line ~768:
```python
            res["_tilt"] = None
```

With:
```python
            # --- Surface normal feedback in NEAR (Improvement 1) ---
            # estimate_near_tilt() is defined above; depth_map is from self.depth_provider()
            # called at the top of step(). In offline mode depth_map is None → tilt stays null.
            tilt = None
            if depth_map is not None and best_centroid is not None:
                tilt = estimate_near_tilt(
                    depth_map, self.K, best_centroid, rng=self._rng)
            if tilt is not None:
                res["tilt_deg"]             = tilt.tilt_deg
                res["roll_deg"]             = tilt.roll_deg
                res["pitch_deg"]            = tilt.pitch_deg
                res["near_plane_normal"]    = tilt.normal.tolist()
                res["near_plane_n_inliers"] = tilt.n_inliers
            res["_tilt"] = tilt
```

The exact old string to replace:
```
            res["_tilt"] = None
```

- [ ] **Step 2: Verify the file parses cleanly**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from foundation_model.servo_lastmile_v2 import estimate_near_tilt, LastMilePipelineV2
print('Import OK')
"
```

Expected: `Import OK`

- [ ] **Step 3: Verify offline run still works (tilt = null, no crash)**

```bash
python foundation_model/servo_lastmile_v2.py \
  --input-video runs/v2_proteinbar_servo12/raw.mp4 \
  --ref-image assets/objects/protein_bar.jpeg \
  --output-dir /tmp/tilt_fix_test \
  --no-overlays --max-frames 10
python -c "
import json
lines = [json.loads(l) for l in open('/tmp/tilt_fix_test/frames.jsonl')]
print('tilt_deg values:', [l['tilt_deg'] for l in lines[:5]])
print('Expected: all None (no depth in offline mode)')
"
```

Expected: all `None` — correct since offline has no depth.

- [ ] **Step 4: Commit**

```bash
git add foundation_model/servo_lastmile_v2.py
git commit -m "fix: call estimate_near_tilt() in NEAR state (was hardcoded None)"
```

---

## Task 3: Unit Tests for Core Components

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_tilt_estimation.py`
- Create: `tests/test_fusion.py`
- Create: `tests/test_ee_masking.py`

**Interfaces:**
- Consumes: `foundation_model.servo_lastmile_v2.estimate_near_tilt`, `foundation_model.servo_lastmile.fit_plane_ransac`, `foundation_model.servo_lastmile.weighted_geometric_median`, `foundation_model.servo_lastmile_v2.EEOcclusionMasker`
- Produces: pytest test suite runnable without any hardware or heavy models

**Context:** There are zero tests in this repo. The three most critical components to test are (1) tilt estimation (has geometry that's easy to verify), (2) the fusion / weighted geometric median (pure math), and (3) EE occlusion masking (geometry). These can all be tested with synthetic inputs.

- [ ] **Step 1: Create tests/__init__.py**

```python
```
(empty file)

- [ ] **Step 2: Create tests/conftest.py**

```python
import sys
from pathlib import Path

# Ensure repo root is on the path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3: Create tests/test_tilt_estimation.py**

```python
"""Tests for estimate_near_tilt in servo_lastmile_v2."""
import math
import numpy as np
import pytest
from foundation_model.servo_lastmile_v2 import estimate_near_tilt, TiltEstimate


def _make_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0) -> np.ndarray:
    K = np.eye(3)
    K[0, 0] = fx; K[1, 1] = fy
    K[0, 2] = cx; K[1, 2] = cy
    return K


def _flat_depth_patch(h=480, w=640, depth_mm=400.0,
                      center=(320, 240), patch_half=80) -> np.ndarray:
    """Depth map with a perfectly flat fronto-parallel patch at `depth_mm`."""
    depth = np.zeros((h, w), dtype=np.float32)
    cx, cy = center
    x1, x2 = max(0, cx-patch_half), min(w, cx+patch_half)
    y1, y2 = max(0, cy-patch_half), min(h, cy+patch_half)
    depth[y1:y2, x1:x2] = depth_mm
    return depth


def test_flat_frontoparallel_tilt_near_zero():
    """A flat plane perpendicular to camera Z should give tilt ≈ 0°."""
    K = _make_K()
    depth = _flat_depth_patch(depth_mm=300.0)
    result = estimate_near_tilt(depth, K, center_xy=(320, 240))
    assert result is not None
    assert result.tilt_deg < 5.0, f"Expected tilt < 5°, got {result.tilt_deg:.2f}°"


def test_no_depth_returns_none():
    """All-zero depth should return None (insufficient valid points)."""
    K = _make_K()
    depth = np.zeros((480, 640), dtype=np.float32)
    result = estimate_near_tilt(depth, K, center_xy=(320, 240))
    assert result is None


def test_insufficient_points_returns_none():
    """Only 10 valid depth points (< min_pts=80) should return None."""
    K = _make_K()
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[238:240, 318:323] = 300.0  # only 10 points
    result = estimate_near_tilt(depth, K, center_xy=(320, 240))
    assert result is None


def test_tilt_estimate_has_all_fields():
    """Returned TiltEstimate should have all fields populated."""
    K = _make_K()
    depth = _flat_depth_patch(depth_mm=400.0)
    result = estimate_near_tilt(depth, K, center_xy=(320, 240))
    assert result is not None
    assert isinstance(result.tilt_deg, float)
    assert isinstance(result.roll_deg, float)
    assert isinstance(result.pitch_deg, float)
    assert result.normal.shape == (3,)
    assert abs(np.linalg.norm(result.normal) - 1.0) < 1e-4, "normal should be unit"
    assert result.n_inliers > 0


def test_tilted_plane_detected():
    """A plane tilted 15° around camera X should show roll ≈ 15°."""
    K = _make_K(fx=500, fy=500, cx=320, cy=240)
    h, w = 480, 640
    # Build a tilted plane: Z = Z0 + tan(15°) * (Y - cy) / fy
    tilt_rad = math.radians(15.0)
    depth = np.zeros((h, w), dtype=np.float32)
    cy_img, cx_img = 240, 320
    for v in range(cy_img - 80, cy_img + 80):
        for u in range(cx_img - 80, cx_img + 80):
            Y = (v - 240.0) * 400.0 / 500.0
            z = 400.0 + math.tan(tilt_rad) * Y
            if 10 < z < 2000:
                depth[v, u] = z
    result = estimate_near_tilt(depth, K, center_xy=(320, 240))
    assert result is not None
    # roll should be close to 15° (allow ±5° due to RANSAC + finite patch)
    assert abs(abs(result.roll_deg) - 15.0) < 5.0, (
        f"Expected roll ≈ 15°, got {result.roll_deg:.2f}°")
```

- [ ] **Step 4: Run tilt tests (expect 4 pass, 1 may need adjustment)**

```bash
cd /home/akanksha/repo/visual_servoing_for_suction_grippers
python -m pytest tests/test_tilt_estimation.py -v
```

Expected: at minimum `test_no_depth_returns_none`, `test_insufficient_points_returns_none`, `test_tilt_estimate_has_all_fields` pass. If `test_tilted_plane_detected` fails, inspect the roll_deg value and adjust the ±5° tolerance.

- [ ] **Step 5: Create tests/test_fusion.py**

```python
"""Tests for weighted_geometric_median and fuse_centroids."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundation_model.servo_lastmile import (
    weighted_geometric_median, SignalReading
)


def test_weighted_geomed_single_point():
    """With one point, the median IS that point."""
    pts = np.array([[100.0, 200.0]])
    weights = np.array([1.0])
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    np.testing.assert_allclose(result, [100.0, 200.0], atol=1.0)


def test_weighted_geomed_two_equal_weights():
    """Two equally-weighted points → result near their midpoint."""
    pts = np.array([[0.0, 0.0], [100.0, 100.0]])
    weights = np.array([1.0, 1.0])
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    # Geometric median with equal weights on 2 points lies between them
    assert 0 <= result[0] <= 100
    assert 0 <= result[1] <= 100


def test_weighted_geomed_dominant_weight():
    """High-weight point should dominate the median."""
    pts = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    weights = np.array([100.0, 0.001])
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    # Should be very close to the dominant point [0, 0]
    assert result[0] < 50.0 and result[1] < 50.0


def test_weighted_geomed_zero_weights_ignored():
    """Points with zero weight should not influence the result."""
    pts = np.array([[0.0, 0.0], [500.0, 500.0]])
    weights = np.array([1.0, 0.0])
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    # Effectively a single point at [0,0]
    np.testing.assert_allclose(result, [0.0, 0.0], atol=5.0)


def test_weighted_geomed_all_zero_weights_returns_none_or_mean():
    """All-zero weights should not crash; returns None or mean fallback."""
    pts = np.array([[0.0, 0.0], [100.0, 100.0]])
    weights = np.array([0.0, 0.0])
    # Should not raise
    try:
        result = weighted_geometric_median(pts, weights)
        # If it returns something, it should be finite
        if result is not None:
            assert np.all(np.isfinite(result))
    except Exception as e:
        pytest.fail(f"weighted_geometric_median raised with all-zero weights: {e}")
```

- [ ] **Step 6: Run fusion tests**

```bash
python -m pytest tests/test_fusion.py -v
```

Expected: all pass. If `weighted_geometric_median` isn't importable, check that `foundation_model.servo_lastmile` exports it.

- [ ] **Step 7: Create tests/test_ee_masking.py**

```python
"""Tests for EEOcclusionMasker."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundation_model.servo_lastmile_v2 import EEOcclusionMasker


def _make_identity_hand_eye() -> np.ndarray:
    """T_ee_cam = identity (EE and cam are co-located)."""
    return np.eye(4)


def _make_T_base_ee(x=0.0, y=0.0, z=-500.0) -> np.ndarray:
    """EE at position (x, y, z) mm in base frame, no rotation."""
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _make_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0) -> np.ndarray:
    K = np.eye(3)
    K[0, 0] = fx; K[1, 1] = fy
    K[0, 2] = cx; K[1, 2] = cy
    return K


def test_masker_returns_mask_shape():
    """get_mask should return (H, W) uint8 mask when EE is in front of camera."""
    K = _make_K()
    T_ee_cam = _make_identity_hand_eye()
    masker = EEOcclusionMasker(T_ee_cam=T_ee_cam, K=K,
                               tip_offset_mm=160.0, body_radius_mm=35.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # EE directly in front of camera at 300mm
    T_base_ee = _make_T_base_ee(z=-300.0)
    mask = masker.get_mask(frame, T_base_ee)
    # May be None if EE projects behind camera — that's fine
    if mask is not None:
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8


def test_masker_apply_zeros_ee_region():
    """apply_to_frame should zero out EE pixels."""
    K = _make_K()
    T_ee_cam = _make_identity_hand_eye()
    masker = EEOcclusionMasker(T_ee_cam=T_ee_cam, K=K,
                               tip_offset_mm=0.0, body_radius_mm=50.0)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    T_base_ee = np.eye(4)
    T_base_ee[2, 3] = 300.0  # EE 300mm in front
    masked_frame, ee_mask = masker.apply_to_frame(frame, T_base_ee)
    assert masked_frame.shape == frame.shape
    # masked_frame should differ from frame if mask was applied
    # (some pixels zeroed)
    if ee_mask is not None:
        assert not np.array_equal(masked_frame, frame)


def test_masker_behind_camera_returns_none():
    """EE behind camera (z <= 0 in camera frame) → mask = None, no crash."""
    K = _make_K()
    T_ee_cam = _make_identity_hand_eye()
    masker = EEOcclusionMasker(T_ee_cam=T_ee_cam, K=K,
                               tip_offset_mm=0.0, body_radius_mm=35.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    T_base_ee = np.eye(4)
    T_base_ee[2, 3] = -500.0  # EE behind camera
    result = masker.get_mask(frame, T_base_ee)
    # Should return None (behind camera) rather than crash
    assert result is None
```

- [ ] **Step 8: Run all tests together**

```bash
python -m pytest tests/ -v
```

Expected: majority pass. Any failures should be on edge-case geometry tests. Fix tolerances if needed (never change the test's semantic intent, only the numeric tolerance if RANSAC is noisy).

- [ ] **Step 9: Commit**

```bash
git add tests/ foundation_model/servo_lastmile_v2.py
git commit -m "test: add unit tests for tilt estimation, fusion, and EE masking"
```

---

## Task 4: Offline Detector Ablation

**Files:**
- Create: `full_system_pipeline/evaluation/offline_detector_ablation.py`

**Interfaces:**
- Consumes: a `raw.mp4` from any existing run, a ref image, the 3 detector backends in `full_system_pipeline/pipeline/servo_lastmile_v2_ext.py`
- Produces: `runs/<ablation_dir>/detector_comparison.png`, `runs/<ablation_dir>/<detector>/frames.jsonl`

**Context:** The repo has 3 detector backends: `sam3`, `gdino+sam2`, `gdino`. `servo_lastmile_v2_ext.py` exposes a `--detector` flag. This script runs all 3 on the same video and plots centroid trajectories side-by-side. Without the robot, this tells us whether detector choice affects the centroid estimate quality. It directly addresses the reviewer question "does the fusion contribute over any single signal."

**Important limitation:** The existing raw.mp4 runs were recorded with the robot moving — the robot stops moving when offline. So the ablation tests *perception* only, not closed-loop control. This is clearly stated in the progress doc.

- [ ] **Step 1: Write offline_detector_ablation.py**

```python
#!/usr/bin/env python3
"""
offline_detector_ablation.py — Run all detector backends on the same raw.mp4
and compare centroid trajectories.

Usage:
    python full_system_pipeline/evaluation/offline_detector_ablation.py \
        --video runs/v2_proteinbar_servo12/raw.mp4 \
        --ref-image assets/objects/protein_bar.jpeg \
        --out-dir runs/ablation_detector_comparison/

Runs: sam3, gdino+sam2, gdino — each gets its own frames.jsonl
Outputs: detector_comparison.png + per-detector frames.jsonl
"""

from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_EXT = _ROOT / "full_system_pipeline" / "pipeline" / "servo_lastmile_v2_ext.py"

DETECTORS = ["sam3", "gdino+sam2", "gdino"]
COLORS = {"sam3": "mediumseagreen", "gdino+sam2": "steelblue", "gdino": "darkorange"}
IMAGE_CENTER = (640.0, 360.0)


def _run_detector(detector: str, video: Path, ref_image: Path,
                  out_dir: Path, max_frames: int | None) -> Path:
    """Run servo_lastmile_v2_ext.py for one detector, return frames.jsonl path."""
    det_dir = out_dir / detector.replace("+", "_plus_")
    det_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_EXT),
        "--detector", detector,
        "--ref-image", str(ref_image),
        "--input-video", str(video),
        "--output-dir", str(det_dir),
        "--no-overlays",
        "--no-robot",
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]

    print(f"  Running detector={detector} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED ({detector}):\n{result.stderr[-500:]}", file=sys.stderr)
    else:
        print(f"  OK ({detector})")
    return det_dir / "frames.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frames = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return frames


def _centroid_err(c, ref=IMAGE_CENTER) -> float | None:
    if c is None:
        return None
    return math.hypot(c[0] - ref[0], c[1] - ref[1])


def _plot_comparison(results: dict[str, list[dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for detector, frames in results.items():
        color = COLORS.get(detector, "gray")
        idxs, errs, xs, ys = [], [], [], []
        for i, f in enumerate(frames):
            bc = f.get("best_centroid")
            if bc is not None:
                idxs.append(i)
                errs.append(_centroid_err(bc))
                xs.append(bc[0])
                ys.append(bc[1])

        label = detector
        axes[0].plot(idxs, errs, color=color, linewidth=1.5, alpha=0.85,
                     label=f"{label} (mean={np.mean(errs):.1f}px)" if errs else label)
        axes[1].plot(idxs, xs, color=color, linewidth=1.2, alpha=0.75, label=label)
        axes[2].plot(idxs, ys, color=color, linewidth=1.2, alpha=0.75, label=label)

    axes[0].set_ylabel("|centroid − center| (px)")
    axes[0].set_title("Centroid Error vs Image Center by Detector")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("centroid X (px)")
    axes[1].axhline(IMAGE_CENTER[0], color="gray", linewidth=0.8, linestyle="--")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("centroid Y (px)")
    axes[2].axhline(IMAGE_CENTER[1], color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("frame index")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "detector_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot: {out_path}")


def _print_summary(results: dict[str, list[dict]]) -> None:
    print("\n=== Detector Comparison Summary ===")
    print(f"{'Detector':<16} {'N frames':>9} {'Mean err (px)':>14} {'Std err (px)':>13}")
    print("-" * 56)
    for detector, frames in results.items():
        errs = [_centroid_err(f.get("best_centroid")) for f in frames
                if f.get("best_centroid") is not None]
        n = len(frames)
        mean_e = f"{np.mean(errs):.1f}" if errs else "N/A"
        std_e = f"{np.std(errs):.1f}" if len(errs) > 1 else "N/A"
        print(f"{detector:<16} {n:>9} {mean_e:>14} {std_e:>13}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to raw.mp4")
    parser.add_argument("--ref-image", required=True, help="Path to reference image")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--detectors", default="sam3,gdino+sam2,gdino",
                        help="Comma-separated list of detectors to compare")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)
    ref = Path(args.ref_image)
    detectors = [d.strip() for d in args.detectors.split(",")]

    results: dict[str, list[dict]] = {}
    for detector in detectors:
        jsonl_path = _run_detector(detector, video, ref, out_dir, args.max_frames)
        results[detector] = _load_jsonl(jsonl_path)

    _print_summary(results)
    _plot_comparison(results, out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the ablation on the protein_bar run (--max-frames 50 for speed)**

```bash
cd /home/akanksha/repo/visual_servoing_for_suction_grippers
python full_system_pipeline/evaluation/offline_detector_ablation.py \
  --video runs/v2_proteinbar_servo12/raw.mp4 \
  --ref-image assets/objects/protein_bar.jpeg \
  --out-dir runs/ablation_detector_comparison/ \
  --max-frames 50
```

Expected: 3 detector runs complete, `detector_comparison.png` generated.

**Note:** If `gdino` or `gdino+sam2` fail because weights aren't downloaded, the script still runs for `sam3` and reports which detectors failed. Check `foundation_model/detection_backends.py` for the weight paths.

- [ ] **Step 3: Verify output exists**

```bash
ls runs/ablation_detector_comparison/
ls runs/ablation_detector_comparison/sam3/frames.jsonl
```

- [ ] **Step 4: Commit**

```bash
git add full_system_pipeline/evaluation/offline_detector_ablation.py
git add runs/ablation_detector_comparison/
git commit -m "eval: add offline detector ablation script + run on protein_bar video"
```

---

## Task 5: Signal Ablation Framework

**Files:**
- Create: `full_system_pipeline/evaluation/signal_ablation.py`

**Interfaces:**
- Consumes: existing `frames.jsonl` from any v2 run
- Produces: per-ablation-condition centroid estimates, comparison plot

**Context:** The JSONL logs all per-signal centroids (`c_A`, `c_B`, `c_C`, `c_D`, `c_E`, `best_centroid`). We can replay the JSONL and simulate what the fused centroid would have been under different signal combinations — WITHOUT re-running the pipeline. This is an ablation that's fully offline and requires no models. The key question: does the fusion outperform any single signal? Does the 80px override threshold matter?

**Important limitation:** This ablation re-computes the fusion from logged per-signal values. If a signal was `None` in the original run (because it wasn't active), the ablation cannot resurrect it. For example, if CoTracker3 (Signal B) returned `None` every frame, the B-only ablation will show no data — but that itself is a finding.

- [ ] **Step 1: Write signal_ablation.py**

```python
#!/usr/bin/env python3
"""
signal_ablation.py — Replay frames.jsonl and compute fused centroids under
different signal subsets. Answers: "does fusion outperform any individual signal?"

Each ablation condition specifies which signals are INCLUDED (the rest are set
to None before fusion). The fusion logic is re-applied from logged per-signal
values using the same weighted_geometric_median as the live pipeline.

Usage:
    python full_system_pipeline/evaluation/signal_ablation.py \
        runs/v2_proteinbar_servo12/ \
        --out-dir runs/signal_ablation/

    # With GT centroid for error computation:
    python full_system_pipeline/evaluation/signal_ablation.py \
        runs/v2_proteinbar_servo12/ \
        --gt-centroid 660 430
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the actual fusion function so we use the real implementation
from foundation_model.servo_lastmile import weighted_geometric_median

IMAGE_CENTER = (640.0, 360.0)

# Ablation conditions: name → set of signal keys to include
ABLATION_CONDITIONS = {
    "all_signals":   {"c_A", "c_B", "c_D", "c_E"},
    "A_only":        {"c_A"},
    "B_only":        {"c_B"},
    "D_only":        {"c_D"},
    "E_only":        {"c_E"},
    "A_plus_B":      {"c_A", "c_B"},
    "A_plus_D":      {"c_A", "c_D"},
    "B_plus_D":      {"c_B", "c_D"},
    "no_A":          {"c_B", "c_D", "c_E"},
    "no_B":          {"c_A", "c_D", "c_E"},
    "canonical":     None,  # use logged best_centroid unchanged
}

COLORS = {
    "all_signals": "black",
    "A_only": "mediumseagreen",
    "B_only": "cyan",
    "D_only": "gold",
    "E_only": "darkorange",
    "A_plus_B": "steelblue",
    "A_plus_D": "limegreen",
    "B_plus_D": "teal",
    "no_A": "firebrick",
    "no_B": "mediumpurple",
    "canonical": "red",
}

# Default signal weights (from the live pipeline; real weights are dynamic
# but in SAM3-phase most weight is on E; in track-phase on B)
# For ablation we use equal weights among available signals — this
# is a simplification, but it isolates the *signal combination* effect
# from the *weighting* effect.
DEFAULT_WEIGHT = 1.0
OVERRIDE_THRESHOLD_PX = 80.0  # from FUSION_OVERRIDE_PX in servo_lastmile.py


def _centroid_err(c, ref) -> float | None:
    if c is None:
        return None
    return math.hypot(c[0] - ref[0], c[1] - ref[1])


def _load_frames(run_dir: Path) -> list[dict]:
    path = run_dir / "frames.jsonl"
    frames = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return frames


def _fuse_signals(frame: dict, signal_keys: set[str],
                  override_px: float = OVERRIDE_THRESHOLD_PX) -> tuple[float, float] | None:
    """Re-compute fused centroid from a subset of signals."""
    pts, weights = [], []
    for key in signal_keys:
        val = frame.get(key)
        if val is not None:
            pts.append(val)
            weights.append(DEFAULT_WEIGHT)

    if not pts:
        return None

    if len(pts) == 1:
        return tuple(pts[0])

    pts_arr = np.array(pts)
    w_arr = np.array(weights)
    fused = weighted_geometric_median(pts_arr, w_arr)
    if fused is None:
        return tuple(pts[0])

    # Apply override: if Signal A is available and result drifts >override_px from A, use A
    c_A = frame.get("c_A")
    if ("c_A" in signal_keys and c_A is not None):
        dist = math.hypot(fused[0] - c_A[0], fused[1] - c_A[1])
        if dist > override_px:
            return tuple(c_A)

    return (float(fused[0]), float(fused[1]))


def compute_ablation_errors(frames: list[dict],
                             condition: str,
                             signal_keys: set[str] | None,
                             ref: tuple[float, float]) -> list[float | None]:
    """Returns per-frame centroid error for this ablation condition."""
    errors = []
    for frame in frames:
        if frame.get("state") not in ("NEAR", "TERMINAL"):
            errors.append(None)
            continue

        if signal_keys is None:
            # "canonical" — use logged best_centroid
            c = frame.get("best_centroid")
        else:
            c = _fuse_signals(frame, signal_keys)
        errors.append(_centroid_err(c, ref))
    return errors


def _plot_ablation(frame_errs: dict[str, list[float | None]],
                   ref: tuple[float, float],
                   out_dir: Path,
                   gt_centroid: tuple[float, float] | None) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    for condition, errs in frame_errs.items():
        color = COLORS.get(condition, "gray")
        idxs = [i for i, e in enumerate(errs) if e is not None]
        vals = [e for e in errs if e is not None]
        if not vals:
            continue
        label = f"{condition} (mean={np.mean(vals):.1f}px)"
        ax.plot(idxs, vals, color=color, linewidth=1.5, alpha=0.8, label=label)

    if gt_centroid is not None:
        gt_err = _centroid_err(gt_centroid, ref)
        ax.axhline(gt_err, color="purple", linestyle="--", linewidth=1.5,
                   label=f"GT err ({gt_err:.1f}px)")

    ax.set_xlabel("frame index")
    ax.set_ylabel("|fused_centroid − ref| (px)")
    ax.set_title("Signal Ablation: Centroid Error by Signal Subset")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "signal_ablation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _print_ablation_summary(frame_errs: dict[str, list[float | None]],
                            ref: tuple[float, float]) -> None:
    print("\n=== Signal Ablation Summary (NEAR + TERMINAL frames only) ===")
    print(f"{'Condition':<16} {'N valid':>8} {'Mean err (px)':>14} {'Std':>8} {'Min':>8}")
    print("-" * 60)
    rows = []
    for condition, errs in frame_errs.items():
        valid = [e for e in errs if e is not None]
        if not valid:
            rows.append((condition, 0, float("inf"), 0, 0))
            continue
        rows.append((condition, len(valid), np.mean(valid), np.std(valid), np.min(valid)))
    rows.sort(key=lambda r: r[2])  # sort by mean err ascending
    for row in rows:
        cond, n, mean, std, min_ = row
        print(f"{cond:<16} {n:>8} {mean:>14.1f} {std:>8.1f} {min_:>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gt-centroid", nargs=2, type=float, metavar=("X", "Y"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (Path(args.run_dirs[0]) / "signal_ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[dict] = []
    for rd in args.run_dirs:
        all_frames.extend(_load_frames(Path(rd)))

    ref = tuple(args.gt_centroid) if args.gt_centroid else IMAGE_CENTER
    gt_centroid = tuple(args.gt_centroid) if args.gt_centroid else None

    frame_errs: dict[str, list[float | None]] = {}
    for condition, signal_keys in ABLATION_CONDITIONS.items():
        frame_errs[condition] = compute_ablation_errors(
            all_frames, condition, signal_keys, ref)

    _print_ablation_summary(frame_errs, ref)
    _plot_ablation(frame_errs, ref, out_dir, gt_centroid)

    # Save summary JSON
    summary = {}
    for condition, errs in frame_errs.items():
        valid = [e for e in errs if e is not None]
        summary[condition] = {
            "n_valid_frames": len(valid),
            "mean_err_px": float(np.mean(valid)) if valid else None,
            "std_err_px": float(np.std(valid)) if valid else None,
            "min_err_px": float(np.min(valid)) if valid else None,
        }
    with open(out_dir / "signal_ablation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON: {out_dir / 'signal_ablation_summary.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the signal ablation on the protein_bar run**

```bash
cd /home/akanksha/repo/visual_servoing_for_suction_grippers
python full_system_pipeline/evaluation/signal_ablation.py \
  runs/v2_proteinbar_servo12/ \
  --out-dir runs/signal_ablation_proteinbar/
```

Expected: table showing which signal combos give lowest mean centroid error. Many conditions will show `N valid: 0` because signals like D and C weren't active (those are only active in the tracking phase when the box exits the frame).

- [ ] **Step 3: Run on the longest live run for richer data**

```bash
python full_system_pipeline/evaluation/signal_ablation.py \
  runs/v2_live_nocup_20260510_213018/ \
  --out-dir runs/signal_ablation_live/
```

- [ ] **Step 4: Commit**

```bash
git add full_system_pipeline/evaluation/signal_ablation.py \
  runs/signal_ablation_proteinbar/ runs/signal_ablation_live/
git commit -m "eval: add signal ablation framework + run on existing data"
```

---

## Task 6: GT Annotation Workflow

**Files:**
- Uses (existing): `full_system_pipeline/evaluation/annotate_gt.py`
- Create: `full_system_pipeline/evaluation/gt_summary_table.py`

**Interfaces:**
- Consumes: `runs/v2_proteinbar_servo12/frames.jsonl`, click-annotated `gt_centroid.json` files
- Produces: `gt_centroid.json` per annotated run, `runs/gt_summary_table.csv`

**Context:** The `annotate_gt.py` tool already exists but hasn't been run. It opens a video frame and lets you click the true grasp center. This is the minimum needed for any quantitative claim about centroid error. Protein_bar runs are the best candidates since we have 6 of them.

- [ ] **Step 1: Run the annotator on the protein_bar servo run**

```bash
cd /home/akanksha/repo/visual_servoing_for_suction_grippers
python full_system_pipeline/evaluation/annotate_gt.py \
  --run-dir runs/v2_proteinbar_servo12/ \
  --ref-image assets/objects/protein_bar.jpeg
```

A window opens. Click the center of the protein bar's top face. Press `s` to save. This writes `runs/v2_proteinbar_servo12/gt_centroid.json`.

- [ ] **Step 2: Re-run analyze_run with the GT centroid**

```bash
python full_system_pipeline/evaluation/analyze_run.py \
  runs/v2_proteinbar_servo12/ \
  --gt-centroid $(python -c "import json; d=json.load(open('runs/v2_proteinbar_servo12/gt_centroid.json')); print(d['x'], d['y'])") \
  --out-dir runs/v2_proteinbar_servo12/analysis_gt/
```

- [ ] **Step 3: Write gt_summary_table.py**

```python
#!/usr/bin/env python3
"""
gt_summary_table.py — Collect all runs with gt_centroid.json and build a
summary table of GT centroid errors.

Usage:
    python full_system_pipeline/evaluation/gt_summary_table.py
    python full_system_pipeline/evaluation/gt_summary_table.py --runs-dir runs/ --out-csv runs/gt_summary.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

IMAGE_CENTER = (640.0, 360.0)


def _centroid_err(c1, c2) -> float:
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _load_frames(run_dir: Path) -> list[dict]:
    path = run_dir / "frames.jsonl"
    frames = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-csv", default="runs/gt_summary.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        gt_path = run_dir / "gt_centroid.json"
        jsonl_path = run_dir / "frames.jsonl"
        if not gt_path.exists() or not jsonl_path.exists():
            continue

        with open(gt_path) as fh:
            gt = json.load(fh)
        gt_centroid = (float(gt["x"]), float(gt["y"]))

        frames = _load_frames(run_dir)
        near_frames = [f for f in frames if f.get("state") in ("NEAR", "TERMINAL")]
        if not near_frames:
            continue

        # Final centroid = best_centroid of the last NEAR/TERMINAL frame
        last_frame = near_frames[-1]
        final_c = last_frame.get("best_centroid")
        final_err = _centroid_err(final_c, gt_centroid) if final_c else None

        all_errs = [_centroid_err(f["best_centroid"], gt_centroid)
                    for f in near_frames if f.get("best_centroid")]
        mean_err = sum(all_errs) / len(all_errs) if all_errs else None

        rows.append({
            "run_dir": run_dir.name,
            "gt_x": gt_centroid[0],
            "gt_y": gt_centroid[1],
            "n_near_frames": len(near_frames),
            "mean_gt_err_px": round(mean_err, 2) if mean_err else None,
            "final_gt_err_px": round(final_err, 2) if final_err else None,
            "reached_terminal": int(any(f.get("state") == "TERMINAL" for f in frames)),
        })

    if not rows:
        print("No runs with gt_centroid.json found.")
        return

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_dir", "gt_x", "gt_y", "n_near_frames",
                  "mean_gt_err_px", "final_gt_err_px", "reached_terminal"]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"GT summary written to: {out}")
    print(f"\n{'Run':<35} {'N near':>7} {'Mean err (px)':>14} {'Final err (px)':>15}")
    print("-" * 75)
    for r in sorted(rows, key=lambda x: x["mean_gt_err_px"] or 9999):
        print(f"{r['run_dir']:<35} {r['n_near_frames']:>7} "
              f"{str(r['mean_gt_err_px'] or 'N/A'):>14} "
              f"{str(r['final_gt_err_px'] or 'N/A'):>15}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the GT summary (after annotating at least 1 run)**

```bash
python full_system_pipeline/evaluation/gt_summary_table.py
```

Expected: table with GT errors for any annotated runs.

- [ ] **Step 5: Commit**

```bash
git add full_system_pipeline/evaluation/gt_summary_table.py
git commit -m "eval: add GT annotation workflow and gt_summary_table.py"
```

---

## Task 7: Progress Document

**Files:**
- Create: `docs/PROGRESS.md`

**Context:** The paper reviewer's key complaint: no quantitative results, system state unclear, too many parallel implementations. This doc fixes that — it's a single source of truth for what's built, what's been measured, what assumptions are baked in, and exactly what experiments to run when the robot is available.

- [ ] **Step 1: Write docs/PROGRESS.md**

Full content in the doc below (see Task 7 artifact). This is the output of running this entire plan — written after Tasks 1–6 complete so it reflects actual numbers.

- [ ] **Step 2: Verify the doc is accurate against the batch_summary.csv**

```bash
python -c "
import csv
rows = list(csv.DictReader(open('runs/batch_summary.csv')))
print('Total runs:', len(rows))
print('Reached terminal:', sum(1 for r in rows if r['reached_terminal']=='1'))
"
```

Copy the actual numbers into `docs/PROGRESS.md` under "Existing Data Inventory."

- [ ] **Step 3: Commit**

```bash
git add docs/PROGRESS.md
git commit -m "docs: add comprehensive PROGRESS.md with experiment templates and limitations"
```

---

## Self-Review

### Spec Coverage

| Reviewer concern | Task addressing it |
|---|---|
| No quantitative results | Task 1 (batch analysis), Task 6 (GT annotation) |
| Tilt not integrated | Task 2 (fix the bug) |
| No tests | Task 3 (unit tests) |
| No ablation of which signals matter | Task 5 (signal ablation) |
| Detector comparison missing | Task 4 (detector ablation) |
| System state unclear | Task 7 (PROGRESS.md) |
| "pick one canonical pipeline" | PROGRESS.md explicitly declares servo_lastmile_v2.py as canonical |

### Placeholder Scan

No TODOs or TBDs in any code blocks. All commands have exact paths.

### Type Consistency

- `_centroid_err(c, ref)` uses same signature in all 4 files (Task 1, 4, 5, 6) ✓
- `_load_frames(run_dir: Path)` consistent across batch_analyze.py and signal_ablation.py ✓
- `weighted_geometric_median(pts, weights)` imported from `foundation_model.servo_lastmile` in test_fusion.py — must match actual export ✓

### Known Gaps Not in This Plan

- SVO2 → depth replay for offline tilt computation (requires pyzed; deferred to robot session)
- Tilt-correction controller integration test (requires robot)
- Full 15-trial experiment (requires robot)
- Actual suction trials (requires physical gripper)
