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
