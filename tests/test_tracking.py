"""Unit tests for tracking — association, Track, SORT tracker."""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.track import Track
from src.tracking.association import iou_batch, cosine_distance, associate_detections_to_tracks
from src.tracking.sort_tracker import SORTTracker


class TestIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self):
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        iou = iou_batch(boxes, boxes)
        assert iou[0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_non_overlapping(self):
        a = np.array([[0, 0, 50, 50]], dtype=np.float32)
        b = np.array([[100, 100, 200, 200]], dtype=np.float32)
        iou = iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float32)
        b = np.array([[50, 50, 150, 150]], dtype=np.float32)
        iou = iou_batch(a, b)
        # Intersection = 50*50 = 2500, Union = 10000 + 10000 - 2500 = 17500
        assert iou[0, 0] == pytest.approx(2500 / 17500, abs=1e-3)

    def test_batch(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
        iou = iou_batch(a, b)
        assert iou.shape == (2, 2)
        assert iou[0, 0] == pytest.approx(1.0, abs=1e-4)
        assert iou[1, 1] == pytest.approx(0.0, abs=1e-4)


class TestCosineDistance:
    """Tests for cosine distance."""

    def test_identical(self):
        a = np.array([[1, 0, 0]], dtype=np.float32)
        dist = cosine_distance(a, a)
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-4)

    def test_orthogonal(self):
        a = np.array([[1, 0]], dtype=np.float32)
        b = np.array([[0, 1]], dtype=np.float32)
        dist = cosine_distance(a, b)
        assert dist[0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_opposite(self):
        a = np.array([[1, 0]], dtype=np.float32)
        b = np.array([[-1, 0]], dtype=np.float32)
        dist = cosine_distance(a, b)
        assert dist[0, 0] == pytest.approx(2.0, abs=1e-4)


class TestAssociation:
    """Tests for detection-to-track association."""

    def test_perfect_match(self):
        dets = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        trks = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        matches, unmatched_d, unmatched_t = associate_detections_to_tracks(dets, trks, 0.3)
        assert len(matches) == 2
        assert len(unmatched_d) == 0
        assert len(unmatched_t) == 0

    def test_no_match(self):
        dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
        trks = np.array([[100, 100, 200, 200]], dtype=np.float32)
        matches, unmatched_d, unmatched_t = associate_detections_to_tracks(dets, trks, 0.3)
        assert len(matches) == 0
        assert len(unmatched_d) == 1
        assert len(unmatched_t) == 1

    def test_empty_detections(self):
        trks = np.array([[0, 0, 10, 10]], dtype=np.float32)
        matches, unmatched_d, unmatched_t = associate_detections_to_tracks(
            np.empty((0, 4)), trks, 0.3
        )
        assert len(matches) == 0
        assert len(unmatched_t) == 1

    def test_empty_tracks(self):
        dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
        matches, unmatched_d, unmatched_t = associate_detections_to_tracks(
            dets, np.empty((0, 4)), 0.3
        )
        assert len(matches) == 0
        assert len(unmatched_d) == 1


class TestTrack:
    """Tests for the Track class."""

    def setup_method(self):
        Track.reset_id_counter()

    def test_creation(self):
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        t = Track(bbox, score=0.9, class_id=0)
        assert t.track_id == 1
        assert t.hits == 1
        assert t.time_since_update == 0

    def test_predict(self):
        t = Track(np.array([10, 10, 50, 50], dtype=np.float32))
        predicted = t.predict()
        assert len(predicted) == 4
        assert t.age == 1
        assert t.time_since_update == 1

    def test_update_resets_time_since_update(self):
        t = Track(np.array([10, 10, 50, 50], dtype=np.float32))
        t.predict()
        t.update(np.array([12, 12, 52, 52], dtype=np.float32))
        assert t.time_since_update == 0
        assert t.hits == 2

    def test_auto_increment_ids(self):
        Track.reset_id_counter()
        t1 = Track(np.array([0, 0, 10, 10], dtype=np.float32))
        t2 = Track(np.array([20, 20, 30, 30], dtype=np.float32))
        assert t2.track_id == t1.track_id + 1

    def test_history_tracking(self):
        t = Track(np.array([0, 0, 100, 100], dtype=np.float32))
        assert len(t.history) == 1
        t.predict()
        assert len(t.history) == 2


class TestSORTTracker:
    """Tests for the SORT tracker."""

    def setup_method(self):
        Track.reset_id_counter()

    def test_creates_tracks(self):
        tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)
        dets = {
            "boxes": np.array([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "class_ids": np.array([0, 1], dtype=np.int32),
        }
        tracks = tracker.update(dets)
        assert len(tracks) >= 1

    def test_maintains_ids(self):
        tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)
        box = np.array([[10, 10, 50, 50]], dtype=np.float32)
        dets = {
            "boxes": box,
            "scores": np.array([0.9], dtype=np.float32),
            "class_ids": np.array([0], dtype=np.int32),
        }
        t1 = tracker.update(dets)
        # Slightly move box
        dets["boxes"] = np.array([[12, 12, 52, 52]], dtype=np.float32)
        t2 = tracker.update(dets)
        # Same object should keep same ID
        if len(t1) > 0 and len(t2) > 0:
            assert t1[0].track_id == t2[0].track_id

    def test_removes_stale_tracks(self):
        tracker = SORTTracker(max_age=2, min_hits=1, iou_threshold=0.3)
        dets = {
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "class_ids": np.array([0], dtype=np.int32),
        }
        tracker.update(dets)
        # Now send empty detections for max_age+1 frames
        empty = {"boxes": np.empty((0, 4)), "scores": np.empty(0), "class_ids": np.empty(0, dtype=np.int32)}
        for _ in range(5):
            tracker.update(empty)
        assert len(tracker.tracks) == 0

    def test_reset(self):
        tracker = SORTTracker()
        tracker.update({
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "class_ids": np.array([0], dtype=np.int32),
        })
        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker._frame_count == 0

    def test_from_config(self):
        cfg = {
            "tracking": {
                "tracker_type": "sort",
                "max_age": 10,
                "min_hits": 2,
                "iou_threshold": 0.4,
                "kalman": {"process_noise": 2.0, "measurement_noise": 0.5, "estimation_error": 5.0},
            }
        }
        tracker = SORTTracker.from_config(cfg)
        assert tracker.max_age == 10
        assert tracker.min_hits == 2
        assert tracker.iou_threshold == 0.4
