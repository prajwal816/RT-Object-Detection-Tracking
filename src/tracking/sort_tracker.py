"""SORT (Simple Online and Realtime Tracking) multi-object tracker.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.tracking.track import Track
from src.tracking.association import associate_detections_to_tracks
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SORTTracker:
    """Multi-object tracker using the SORT algorithm.

    Assigns detections to existing tracks via IoU + Hungarian matching.
    Un-matched detections create new tracks; stale tracks are removed.

    Args:
        max_age: Maximum frames a track can go unmatched before deletion.
        min_hits: Minimum matched frames before a track is returned as confirmed.
        iou_threshold: IoU gate for valid association.
        kalman_params: Dict with ``process_noise``, ``measurement_noise``,
            ``estimation_error`` forwarded to each Track's Kalman filter.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        kalman_params: Dict[str, float] | None = None,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._kf_params = kalman_params or {}
        self.tracks: List[Track] = []
        self._frame_count: int = 0
        self.id_switches: int = 0

    def update(self, detections: Dict[str, np.ndarray]) -> List[Track]:
        """Process a new frame of detections.

        Args:
            detections: Dictionary with keys ``boxes`` (N, 4), ``scores`` (N,),
                ``class_ids`` (N,).

        Returns:
            List of active (confirmed) Track objects.
        """
        self._frame_count += 1
        boxes = detections.get("boxes", np.empty((0, 4)))
        scores = detections.get("scores", np.empty((0,)))
        class_ids = detections.get("class_ids", np.empty((0,), dtype=np.int32))

        # ── 1. Predict existing tracks ───────────────────────────────────
        predicted_boxes = []
        for trk in self.tracks:
            trk.predict()
            predicted_boxes.append(trk.bbox)

        if len(predicted_boxes) > 0:
            predicted_boxes = np.array(predicted_boxes)
        else:
            predicted_boxes = np.empty((0, 4))

        # ── 2. Associate detections to tracks ────────────────────────────
        matches, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            boxes, predicted_boxes, self.iou_threshold,
        )

        # ── 3. Update matched tracks ────────────────────────────────────
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(
                boxes[det_idx],
                score=float(scores[det_idx]) if len(scores) > det_idx else 0.0,
                class_id=int(class_ids[det_idx]) if len(class_ids) > det_idx else 0,
            )

        # ── 4. Create new tracks for unmatched detections ───────────────
        for det_idx in unmatched_dets:
            new_track = Track(
                bbox=boxes[det_idx],
                score=float(scores[det_idx]) if len(scores) > det_idx else 0.0,
                class_id=int(class_ids[det_idx]) if len(class_ids) > det_idx else 0,
                **self._kf_params,
            )
            self.tracks.append(new_track)
            logger.debug(f"New track: ID {new_track.track_id}")

        # ── 5. Remove stale tracks ──────────────────────────────────────
        active: List[Track] = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                active.append(trk)
            else:
                logger.debug(f"Removed stale track: ID {trk.track_id}")
        self.tracks = active

        # ── 6. Return confirmed tracks ──────────────────────────────────
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits or self._frame_count <= self.min_hits]
        return confirmed

    def reset(self) -> None:
        """Reset all tracks and counters."""
        self.tracks.clear()
        self._frame_count = 0
        self.id_switches = 0
        Track.reset_id_counter()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SORTTracker":
        """Construct a SORT tracker from config dictionary."""
        trk_cfg = cfg.get("tracking", {})
        kalman_cfg = trk_cfg.get("kalman", {})
        return cls(
            max_age=trk_cfg.get("max_age", 30),
            min_hits=trk_cfg.get("min_hits", 3),
            iou_threshold=trk_cfg.get("iou_threshold", 0.3),
            kalman_params={
                "process_noise": kalman_cfg.get("process_noise", 1.0),
                "measurement_noise": kalman_cfg.get("measurement_noise", 1.0),
                "estimation_error": kalman_cfg.get("estimation_error", 10.0),
            },
        )
