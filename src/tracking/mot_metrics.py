"""MOT (Multiple Object Tracking) evaluation metrics.

Implements standard MOT metrics:
- MOTA  (Multiple Object Tracking Accuracy)
- MOTP  (Multiple Object Tracking Precision)
- IDF1  (ID F1 Score)
- ID Switches
- Mostly Tracked / Mostly Lost ratios

These operate on frame-level matched predictions vs ground truth annotations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


class MOTAccumulator:
    """Accumulates frame-level tracking results for MOT metric computation.

    Usage::

        acc = MOTAccumulator()
        for frame_id in range(num_frames):
            acc.update(
                gt_ids=gt_ids_this_frame,
                gt_boxes=gt_boxes_this_frame,
                pred_ids=pred_ids_this_frame,
                pred_boxes=pred_boxes_this_frame,
            )
        metrics = acc.compute()
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

        # Per-frame accumulation
        self._total_gt: int = 0        # total ground truth objects
        self._total_pred: int = 0      # total predicted objects
        self._total_matches: int = 0   # true positives
        self._total_fp: int = 0        # false positives
        self._total_fn: int = 0        # false negatives (misses)
        self._total_id_switches: int = 0
        self._total_iou_sum: float = 0.0  # for MOTP

        # ID tracking across frames
        self._prev_mapping: Dict[int, int] = {}  # gt_id → pred_id

        # Per ground-truth track stats
        self._gt_track_lengths: Dict[int, int] = defaultdict(int)
        self._gt_track_matched: Dict[int, int] = defaultdict(int)

        # For IDF1
        self._idtp: int = 0
        self._idfp: int = 0
        self._idfn: int = 0

        self._frame_count: int = 0

    def update(
        self,
        gt_ids: np.ndarray,
        gt_boxes: np.ndarray,
        pred_ids: np.ndarray,
        pred_boxes: np.ndarray,
    ) -> None:
        """Process one frame of GT and predicted tracks.

        Args:
            gt_ids: (G,) ground truth object IDs.
            gt_boxes: (G, 4) GT boxes ``[x1, y1, x2, y2]``.
            pred_ids: (P,) predicted track IDs.
            pred_boxes: (P, 4) predicted boxes ``[x1, y1, x2, y2]``.
        """
        self._frame_count += 1
        G = len(gt_ids)
        P = len(pred_ids)

        self._total_gt += G
        self._total_pred += P

        # Update GT track lengths
        for gid in gt_ids:
            self._gt_track_lengths[int(gid)] += 1

        if G == 0 and P == 0:
            return
        if G == 0:
            self._total_fp += P
            self._idfp += P
            return
        if P == 0:
            self._total_fn += G
            self._idfn += G
            return

        # ── IoU cost matrix ──────────────────────────────────────────────
        iou_matrix = np.zeros((G, P), dtype=np.float64)
        for i in range(G):
            for j in range(P):
                iou_matrix[i, j] = _iou(gt_boxes[i], pred_boxes[j])

        cost = 1.0 - iou_matrix
        gt_idx, pred_idx = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()
        current_mapping: Dict[int, int] = {}

        for gi, pi in zip(gt_idx, pred_idx):
            if iou_matrix[gi, pi] >= self.iou_threshold:
                matched_gt.add(gi)
                matched_pred.add(pi)

                gid = int(gt_ids[gi])
                pid = int(pred_ids[pi])
                current_mapping[gid] = pid

                self._total_matches += 1
                self._total_iou_sum += iou_matrix[gi, pi]
                self._gt_track_matched[gid] += 1

                # ID switch detection
                if gid in self._prev_mapping and self._prev_mapping[gid] != pid:
                    self._total_id_switches += 1

                self._idtp += 1

        # Unmatched
        fn = G - len(matched_gt)
        fp = P - len(matched_pred)
        self._total_fn += fn
        self._total_fp += fp
        self._idfn += fn
        self._idfp += fp

        self._prev_mapping = current_mapping

    def compute(self) -> Dict[str, Any]:
        """Compute final MOT metrics.

        Returns:
            Dictionary with MOTA, MOTP, IDF1, ID switches, MT/ML ratios,
            precision, recall, and F1.
        """
        tp = self._total_matches
        fp = self._total_fp
        fn = self._total_fn
        idsw = self._total_id_switches
        total_gt = self._total_gt

        # MOTA = 1 - (FN + FP + IDSW) / total_GT
        mota = 1.0 - (fn + fp + idsw) / max(total_gt, 1)

        # MOTP = sum(IoU of matches) / num_matches
        motp = self._total_iou_sum / max(tp, 1)

        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        idf1 = 2 * self._idtp / max(2 * self._idtp + self._idfp + self._idfn, 1)

        # Precision & Recall
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Mostly Tracked (>80% frames matched) / Mostly Lost (<20%)
        mt_count = 0
        ml_count = 0
        for gid, total_frames in self._gt_track_lengths.items():
            matched_frames = self._gt_track_matched.get(gid, 0)
            ratio = matched_frames / max(total_frames, 1)
            if ratio >= 0.8:
                mt_count += 1
            elif ratio <= 0.2:
                ml_count += 1

        num_gt_tracks = len(self._gt_track_lengths)
        mt_ratio = mt_count / max(num_gt_tracks, 1)
        ml_ratio = ml_count / max(num_gt_tracks, 1)

        metrics = {
            "MOTA": round(mota, 4),
            "MOTP": round(motp, 4),
            "IDF1": round(idf1, 4),
            "ID_switches": idsw,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "F1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_gt_objects": total_gt,
            "total_frames": self._frame_count,
            "num_gt_tracks": num_gt_tracks,
            "mostly_tracked_ratio": round(mt_ratio, 4),
            "mostly_lost_ratio": round(ml_ratio, 4),
        }

        logger.info(f"MOT metrics: MOTA={mota:.4f}, MOTP={motp:.4f}, IDF1={idf1:.4f}, IDSW={idsw}")
        return metrics

    def reset(self) -> None:
        """Reset all accumulated data."""
        self.__init__(self.iou_threshold)


def evaluate_mot(
    ground_truth: Dict[int, Dict[str, np.ndarray]],
    predictions: Dict[int, Dict[str, np.ndarray]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate MOT metrics over an entire sequence.

    Args:
        ground_truth: ``{frame_id: {"ids": (G,), "boxes": (G, 4)}}``
        predictions: ``{frame_id: {"ids": (P,), "boxes": (P, 4)}}``
        iou_threshold: IoU matching threshold.

    Returns:
        MOT metrics dictionary.
    """
    acc = MOTAccumulator(iou_threshold=iou_threshold)
    all_frames = sorted(set(ground_truth.keys()) | set(predictions.keys()))

    for fid in all_frames:
        gt = ground_truth.get(fid, {"ids": np.empty(0, dtype=int), "boxes": np.empty((0, 4))})
        pred = predictions.get(fid, {"ids": np.empty(0, dtype=int), "boxes": np.empty((0, 4))})

        acc.update(
            gt_ids=gt["ids"],
            gt_boxes=gt["boxes"],
            pred_ids=pred["ids"],
            pred_boxes=pred["boxes"],
        )

    return acc.compute()
