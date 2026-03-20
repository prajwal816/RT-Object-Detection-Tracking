"""Unit tests for the Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.filters.kalman_filter import KalmanBoxFilter, _xyxy_to_xsr, _xsr_to_xyxy


class TestCoordinateConversion:
    """Tests for xyxy ↔ xsr conversion functions."""

    def test_xyxy_to_xsr_basic(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float64)
        xsr = _xyxy_to_xsr(bbox)
        assert xsr[0] == pytest.approx(150.0)  # cx
        assert xsr[1] == pytest.approx(150.0)  # cy
        assert xsr[2] == pytest.approx(10000.0)  # area = 100*100
        assert xsr[3] == pytest.approx(1.0)  # aspect ratio

    def test_roundtrip(self):
        bbox = np.array([50, 30, 180, 220], dtype=np.float64)
        xsr = _xyxy_to_xsr(bbox)
        recovered = _xsr_to_xyxy(xsr)
        np.testing.assert_allclose(recovered, bbox, atol=0.5)

    def test_non_square(self):
        bbox = np.array([0, 0, 100, 50], dtype=np.float64)
        xsr = _xyxy_to_xsr(bbox)
        assert xsr[3] == pytest.approx(2.0)  # width/height = 100/50


class TestKalmanBoxFilter:
    """Tests for the Kalman filter tracking a bounding box."""

    def test_initialization(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float64)
        kf = KalmanBoxFilter(bbox)
        result = kf.bbox
        np.testing.assert_allclose(result, bbox, atol=1.0)

    def test_predict_moves_state(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float64)
        kf = KalmanBoxFilter(bbox)
        p1 = kf.predict()
        # After one prediction with no velocity, should stay roughly the same
        np.testing.assert_allclose(p1, bbox, atol=5.0)

    def test_update_corrects(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float64)
        kf = KalmanBoxFilter(bbox)
        kf.predict()
        # Shift the box
        new_bbox = np.array([110, 110, 210, 210], dtype=np.float64)
        result = kf.update(new_bbox)
        # Should be closer to the new measurement
        center_x = (result[0] + result[2]) / 2
        assert center_x > 150.0  # moved toward 160

    def test_velocity_estimation(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float64)
        kf = KalmanBoxFilter(bbox)
        # Move box right consistently
        for i in range(10):
            kf.predict()
            kf.update(bbox + np.array([i * 5, 0, i * 5, 0]))
        vel = kf.velocity
        assert vel[0] > 0  # should detect rightward motion

    def test_predict_after_multiple_updates(self):
        kf = KalmanBoxFilter(np.array([0, 0, 50, 50], dtype=np.float64))
        for i in range(20):
            kf.predict()
            kf.update(np.array([i * 2, 0, 50 + i * 2, 50], dtype=np.float64))
        predicted = kf.predict()
        # Should predict the box will continue moving right
        assert predicted[0] > 30
