"""Unit tests for MOT evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.mot_metrics import MOTAccumulator, evaluate_mot


class TestMOTAccumulator:
    """Tests for the MOT metrics accumulator."""

    def test_perfect_tracking(self):
        """All detections perfectly match GT."""
        acc = MOTAccumulator(iou_threshold=0.5)
        for i in range(10):
            gt_ids = np.array([1, 2])
            gt_boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float64)
            pred_ids = np.array([1, 2])
            pred_boxes = gt_boxes.copy()
            acc.update(gt_ids, gt_boxes, pred_ids, pred_boxes)

        metrics = acc.compute()
        assert metrics["MOTA"] == pytest.approx(1.0, abs=0.01)
        assert metrics["MOTP"] == pytest.approx(1.0, abs=0.01)
        assert metrics["ID_switches"] == 0

    def test_all_misses(self):
        """No predictions at all — all false negatives."""
        acc = MOTAccumulator(iou_threshold=0.5)
        for _ in range(5):
            acc.update(
                np.array([1]),
                np.array([[0, 0, 50, 50]], dtype=np.float64),
                np.empty(0, dtype=int),
                np.empty((0, 4), dtype=np.float64),
            )
        metrics = acc.compute()
        assert metrics["false_negatives"] == 5
        assert metrics["true_positives"] == 0
        assert metrics["MOTA"] < 0.0  # negative MOTA for all misses

    def test_all_false_positives(self):
        """No GT — all predictions are false positives."""
        acc = MOTAccumulator(iou_threshold=0.5)
        for _ in range(5):
            acc.update(
                np.empty(0, dtype=int),
                np.empty((0, 4), dtype=np.float64),
                np.array([1]),
                np.array([[0, 0, 50, 50]], dtype=np.float64),
            )
        metrics = acc.compute()
        assert metrics["false_positives"] == 5

    def test_id_switch_detection(self):
        """Simulate an ID switch."""
        acc = MOTAccumulator(iou_threshold=0.3)
        box = np.array([[10, 10, 50, 50]], dtype=np.float64)

        # Frame 1: GT=1 matched to Pred=10
        acc.update(np.array([1]), box, np.array([10]), box)
        # Frame 2: GT=1 matched to Pred=20 (ID switch!)
        acc.update(np.array([1]), box, np.array([20]), box)

        metrics = acc.compute()
        assert metrics["ID_switches"] == 1

    def test_idf1_range(self):
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        for _ in range(10):
            acc.update(np.array([1]), gt_boxes, np.array([1]), gt_boxes)

        metrics = acc.compute()
        assert 0.0 <= metrics["IDF1"] <= 1.0

    def test_mostly_tracked(self):
        acc = MOTAccumulator(iou_threshold=0.5)
        gt_boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        # Track 1 matched in all 10 frames
        for _ in range(10):
            acc.update(np.array([1]), gt_boxes, np.array([1]), gt_boxes)
        metrics = acc.compute()
        assert metrics["mostly_tracked_ratio"] == pytest.approx(1.0)

    def test_reset(self):
        acc = MOTAccumulator()
        acc.update(np.array([1]), np.array([[0, 0, 10, 10]]), np.array([1]), np.array([[0, 0, 10, 10]]))
        acc.reset()
        assert acc._frame_count == 0


class TestEvaluateMOT:
    """Tests for the convenience evaluate_mot function."""

    def test_basic_evaluation(self):
        gt = {
            0: {"ids": np.array([1, 2]), "boxes": np.array([[0, 0, 50, 50], [60, 60, 120, 120]])},
            1: {"ids": np.array([1, 2]), "boxes": np.array([[2, 2, 52, 52], [62, 62, 122, 122]])},
        }
        pred = {
            0: {"ids": np.array([1, 2]), "boxes": np.array([[0, 0, 50, 50], [60, 60, 120, 120]])},
            1: {"ids": np.array([1, 2]), "boxes": np.array([[2, 2, 52, 52], [62, 62, 122, 122]])},
        }
        metrics = evaluate_mot(gt, pred)
        assert metrics["MOTA"] > 0.9
        assert metrics["total_frames"] == 2

    def test_missing_frames(self):
        gt = {0: {"ids": np.array([1]), "boxes": np.array([[0, 0, 50, 50]])}}
        pred = {1: {"ids": np.array([1]), "boxes": np.array([[0, 0, 50, 50]])}}
        metrics = evaluate_mot(gt, pred)
        # Frame 0: GT present, no pred → FN
        # Frame 1: No GT, pred present → FP
        assert metrics["false_negatives"] >= 1
        assert metrics["false_positives"] >= 1
