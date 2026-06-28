"""Tests for weighted_geometric_median and fuse_centroids."""
import numpy as np
import pytest

from foundation_model.servo_lastmile import weighted_geometric_median


def test_weighted_geomed_single_point():
    """With one point, the median IS that point."""
    pts = [(100.0, 200.0)]
    weights = [1.0]
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    np.testing.assert_allclose(result, (100.0, 200.0), atol=1.0)


def test_weighted_geomed_two_equal_weights():
    """Two equally-weighted points → result near their midpoint."""
    pts = [(0.0, 0.0), (100.0, 100.0)]
    weights = [1.0, 1.0]
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    # Geometric median with equal weights on 2 points lies between them
    assert 0 <= result[0] <= 100
    assert 0 <= result[1] <= 100


def test_weighted_geomed_dominant_weight():
    """High-weight point should dominate the median."""
    pts = [(0.0, 0.0), (1000.0, 1000.0)]
    weights = [100.0, 0.001]
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    # Should be very close to the dominant point [0, 0]
    assert result[0] < 50.0 and result[1] < 50.0


def test_weighted_geomed_zero_weights_ignored():
    """Points with zero weight should not influence the result."""
    pts = [(0.0, 0.0), (500.0, 500.0)]
    weights = [1.0, 0.0]
    result = weighted_geometric_median(pts, weights)
    assert result is not None
    # Effectively a single point at [0,0]
    np.testing.assert_allclose(result, (0.0, 0.0), atol=5.0)


def test_weighted_geomed_all_zero_weights_returns_none_or_mean():
    """All-zero weights should not crash; returns None or mean fallback."""
    pts = [(0.0, 0.0), (100.0, 100.0)]
    weights = [0.0, 0.0]
    # Should not raise
    try:
        result = weighted_geometric_median(pts, weights)
        # If it returns something, it should be finite
        if result is not None:
            assert all(np.isfinite(v) for v in result)
    except Exception as e:
        pytest.fail(f"weighted_geometric_median raised with all-zero weights: {e}")
