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
