"""Unit tests for LastMilePipelineSimple — no robot, no SAM3 models required."""
import numpy as np
import pytest

from foundation_model.servo_lastmile import State
from foundation_model.servo_lastmile_simple import (
    LastMilePipelineSimple,
    _NoopSignal,
    _empty_result_simple,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _sam3_detect(cx=320.0, cy=240.0, score=0.95):
    """SAM3 runner that always detects a small centred box."""
    def runner(frame_bgr):
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        r = 80
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))
        mask[y0:y1, x0:x1] = 255
        return {"gdino_box": [x0, y0, x1, y1], "mask_np": mask, "sam_score": score}
    return runner


def _sam3_miss():
    """SAM3 runner that always returns empty."""
    def runner(frame_bgr):
        return {"gdino_box": None, "mask_np": None, "sam_score": 0.0}
    return runner


def _make_pipeline(runner):
    p = LastMilePipelineSimple(sam3_runner=runner)
    p.reset()
    return p


# ── tests ─────────────────────────────────────────────────────────────────────

def test_empty_result_has_required_fields():
    res = _empty_result_simple()
    required = (
        "state", "best_centroid", "z_mm", "tilt_deg", "roll_deg",
        "pitch_deg", "near_plane_n_inliers", "phase",
        "holding_last_centroid", "sam3_detected", "frame_idx",
    )
    for f in required:
        assert f in res, f"Missing field: {f}"


def test_noop_signal_interface():
    s = _NoopSignal()
    s.reset()                        # must not raise
    s.reset("arg1", kw=2)            # variadic — must not raise
    assert s.step() == (None, 0.0, None)
    assert s.step("a", "b") == (None, 0.0, None)


def test_far_state_no_detection():
    """First frame with no detection stays in FAR, centroid is None."""
    p = _make_pipeline(_sam3_miss())
    res = p.step(_blank())
    assert res["state"] == "FAR"
    assert res["best_centroid"] is None


def test_near_sam3_phase_updates_centroid():
    """In NEAR with box detected, centroid updates and phase is 'sam3'."""
    p = _make_pipeline(_sam3_detect(cx=320, cy=240))
    p.fsm.state = State.NEAR
    p._last_centroid = (320.0, 240.0)

    res = p.step(_blank())
    assert res["state"] == "NEAR"
    assert res["phase"] == "sam3"
    assert res["holding_last_centroid"] is False
    assert res["sam3_detected"] is True
    assert res["best_centroid"] is not None
    cx, cy = res["best_centroid"]
    assert abs(cx - 320) < 50
    assert abs(cy - 240) < 50


def test_near_hold_when_sam3_fails():
    """In NEAR with no detection, last centroid is held and phase is 'hold'."""
    p = _make_pipeline(_sam3_miss())
    p.fsm.state = State.NEAR
    p._last_centroid = (300.0, 250.0)

    res = p.step(_blank())
    assert res["state"] == "NEAR"
    assert res["phase"] == "hold"
    assert res["holding_last_centroid"] is True
    assert res["sam3_detected"] is False
    assert res["best_centroid"] == (300.0, 250.0)   # must NOT be None


def test_frame_idx_increments_each_step():
    p = _make_pipeline(_sam3_miss())
    frame = _blank()
    for i in range(5):
        res = p.step(frame)
    assert res["frame_idx"] == 5


def test_terminal_returns_image_center():
    """TERMINAL short-circuits with image-centre centroid (no servo correction)."""
    p = _make_pipeline(_sam3_miss())
    p.fsm.state = State.TERM
    res = p.step(_blank(h=480, w=640))
    assert res["state"] == State.TERM
    assert res["best_centroid"] == (320.0, 240.0)


def test_reset_clears_near_state():
    """reset() returns pipeline to FAR and clears tracking state."""
    p = _make_pipeline(_sam3_miss())
    p.fsm.state = State.NEAR
    p._last_centroid = (100.0, 200.0)
    p._holding_last_centroid = True

    p.reset()
    assert p.fsm.state == State.FAR
    assert p._last_centroid is None
    assert p._holding_last_centroid is False
