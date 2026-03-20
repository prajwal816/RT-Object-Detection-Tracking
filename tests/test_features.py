"""Unit tests for feature extractors and optical flow."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.sift_extractor import SIFTExtractor
from src.features.orb_extractor import ORBExtractor
from src.features.optical_flow import LucasKanadeFlow


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a synthetic BGR frame with some texture."""
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return frame


class TestSIFTExtractor:
    """Tests for the SIFT extractor."""

    def test_extract_returns_keypoints(self):
        extractor = SIFTExtractor(max_keypoints=100)
        frame = _make_frame()
        kps, descs = extractor.extract(frame)
        assert len(kps) > 0
        assert descs is not None
        assert descs.shape[1] == 128

    def test_max_keypoints(self):
        extractor = SIFTExtractor(max_keypoints=10)
        frame = _make_frame()
        kps, _ = extractor.extract(frame)
        # OpenCV's SIFT nfeatures is approximate; allow small margin
        assert len(kps) <= 15

    def test_match_self(self):
        extractor = SIFTExtractor(max_keypoints=50)
        frame = _make_frame()
        _, desc = extractor.extract(frame)
        matches = extractor.match(desc, desc)
        assert len(matches) > 0

    def test_match_with_none(self):
        extractor = SIFTExtractor()
        matches = extractor.match(None, None)
        assert matches == []


class TestORBExtractor:
    """Tests for the ORB extractor."""

    def test_extract_returns_keypoints(self):
        extractor = ORBExtractor(max_keypoints=100)
        frame = _make_frame()
        kps, descs = extractor.extract(frame)
        assert len(kps) > 0
        assert descs is not None
        assert descs.shape[1] == 32  # ORB descriptors are 32 bytes

    def test_match_self(self):
        extractor = ORBExtractor(max_keypoints=50)
        frame = _make_frame()
        _, desc = extractor.extract(frame)
        matches = extractor.match(desc, desc, max_distance=100)
        assert len(matches) > 0


class TestOpticalFlow:
    """Tests for Lucas-Kanade optical flow."""

    def test_compute_first_frame(self):
        flow = LucasKanadeFlow()
        frame = _make_frame()
        points = np.array([[100, 100], [200, 200]], dtype=np.float32)
        new_pts, status, errors = flow.compute(frame, points)
        # First frame: no previous frame, should return empty
        assert len(new_pts) == 0

    def test_compute_two_frames(self):
        flow = LucasKanadeFlow()
        frame1 = _make_frame()
        frame2 = _make_frame()

        points = np.array([[100, 100], [200, 200]], dtype=np.float32)
        # Initialize with first frame
        flow.compute(frame1, points)
        # Track points to second frame
        new_pts, status, errors = flow.compute(frame2, points)
        assert len(new_pts) == 2

    def test_compute_from_boxes(self):
        flow = LucasKanadeFlow()
        frame1 = _make_frame()
        frame2 = _make_frame()
        boxes = np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float32)

        # Initialize
        flow.compute_from_boxes(frame1, boxes)
        # Compute displacement
        disp = flow.compute_from_boxes(frame2, boxes)
        assert disp.shape == (2, 2)

    def test_empty_boxes(self):
        flow = LucasKanadeFlow()
        frame = _make_frame()
        disp = flow.compute_from_boxes(frame, np.empty((0, 4), dtype=np.float32))
        assert len(disp) == 0

    def test_reset(self):
        flow = LucasKanadeFlow()
        flow.compute(_make_frame(), np.array([[100, 100]], dtype=np.float32))
        flow.reset()
        assert flow._prev_gray is None
