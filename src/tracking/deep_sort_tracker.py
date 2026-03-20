"""DeepSORT tracker — SORT with appearance-based re-identification.

Extends SORT by incorporating an appearance embedding (from a lightweight CNN
or a simple crop histogram) to improve association accuracy and reduce
ID switches during occlusions.

Reference: Wojke et al., "Simple Online and Realtime Tracking with a
Deep Association Metric", ICIP 2017.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.track import Track
from src.tracking.association import iou_batch, cosine_distance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleEmbedder:
    """Lightweight appearance embedder using resized crop histograms.

    When no dedicated re-ID CNN is available, this provides a cheap
    appearance descriptor by computing a normalized colour histogram of
    each detection crop.

    Args:
        crop_size: Resize crops to this ``(width, height)``.
        hist_bins: Number of bins per colour channel.
    """

    def __init__(self, crop_size: Tuple[int, int] = (64, 128), hist_bins: int = 16) -> None:
        self.crop_size = crop_size
        self.hist_bins = hist_bins

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute appearance embeddings for each box.

        Args:
            frame: BGR image.
            boxes: (N, 4) ``[x1, y1, x2, y2]``.

        Returns:
            (N, D) normalized embedding vectors.
        """
        h, w = frame.shape[:2]
        embeddings = []
        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                embeddings.append(np.zeros(self.hist_bins * 3, dtype=np.float32))
                continue
            crop = cv2.resize(crop, self.crop_size)
            hist_parts = []
            for ch in range(3):
                hist = cv2.calcHist([crop], [ch], None, [self.hist_bins], [0, 256])
                hist_parts.append(hist.flatten())
            emb = np.concatenate(hist_parts).astype(np.float32)
            norm = np.linalg.norm(emb) + 1e-8
            embeddings.append(emb / norm)
        return np.array(embeddings, dtype=np.float32)


class DeepSORTTracker:
    """Multi-object tracker using the DeepSORT algorithm.

    Combines IoU-based gating with cosine distance on appearance embeddings
    for improved association under occlusions.

    Args:
        max_age: Maximum frames without a match before track deletion.
        min_hits: Minimum hits to consider a track confirmed.
        iou_threshold: IoU gate for valid association.
        max_cosine_distance: Cosine distance gate for appearance matching.
        nn_budget: Max gallery size per track for nearest-neighbor distance.
        embedding_model: Optional path to a re-ID CNN weights file.
        use_reid_cnn: If True, use MobileNetV2 Re-ID CNN even without pre-trained weights.
        device: Device for Re-ID CNN inference ("cpu" or "cuda").
        kalman_params: Kalman filter parameters.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
        embedding_model: Optional[str] = None,
        use_reid_cnn: bool = False,
        device: str = "cpu",
        kalman_params: Dict[str, float] | None = None,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self._kf_params = kalman_params or {}

        self.tracks: List[Track] = []
        self._frame_count: int = 0
        self.id_switches: int = 0

        # Appearance embedder — Re-ID CNN or histogram fallback
        if use_reid_cnn or embedding_model:
            try:
                from src.tracking.reid_embedder import ReIDEmbedder
                self._embedder = ReIDEmbedder(
                    model_path=embedding_model,
                    device=device,
                )
                self._embedding_dim = 128
                logger.info("Using Re-ID CNN embedder (MobileNetV2)")
            except ImportError:
                logger.warning("torch/torchvision not available — falling back to histogram embedder")
                self._embedder = SimpleEmbedder()
                self._embedding_dim = 48
        else:
            self._embedder = SimpleEmbedder()
            self._embedding_dim = 48

        # Gallery of embeddings per track ID
        self._gallery: Dict[int, List[np.ndarray]] = {}

    def _update_gallery(self, track_id: int, embedding: np.ndarray) -> None:
        """Add an embedding to the track's gallery (FIFO bounded by nn_budget)."""
        if track_id not in self._gallery:
            self._gallery[track_id] = []
        self._gallery[track_id].append(embedding)
        if len(self._gallery[track_id]) > self.nn_budget:
            self._gallery[track_id] = self._gallery[track_id][-self.nn_budget:]

    def _nn_cosine_distance(self, det_embeddings: np.ndarray, track_ids: List[int]) -> np.ndarray:
        """Compute nearest-neighbour cosine distance between detections and tracks."""
        N = len(det_embeddings)
        M = len(track_ids)
        dist = np.full((N, M), 1e5, dtype=np.float32)

        for j, tid in enumerate(track_ids):
            gallery = self._gallery.get(tid)
            if gallery is None or len(gallery) == 0:
                continue
            gallery_mat = np.array(gallery, dtype=np.float32)
            d = cosine_distance(det_embeddings, gallery_mat)  # (N, G)
            dist[:, j] = d.min(axis=1)  # nearest neighbour

        return dist

    def update(
        self,
        detections: Dict[str, np.ndarray],
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Process new detections for a frame.

        Args:
            detections: Dict with ``boxes``, ``scores``, ``class_ids``.
            frame: Current BGR frame (needed for appearance extraction).

        Returns:
            List of confirmed Track objects.
        """
        self._frame_count += 1
        boxes = detections.get("boxes", np.empty((0, 4)))
        scores = detections.get("scores", np.empty((0,)))
        class_ids = detections.get("class_ids", np.empty((0,), dtype=np.int32))

        # ── 1. Predict ───────────────────────────────────────────────────
        predicted = []
        for trk in self.tracks:
            trk.predict()
            predicted.append(trk.bbox)
        predicted = np.array(predicted) if predicted else np.empty((0, 4))

        # ── 2. Extract appearance embeddings ─────────────────────────────
        if frame is not None and len(boxes) > 0:
            det_embeddings = self._embedder.extract(frame, boxes)
        else:
            det_embeddings = np.zeros((len(boxes), self._embedding_dim), dtype=np.float32)

        # ── 3. Build combined cost matrix ────────────────────────────────
        if len(boxes) > 0 and len(self.tracks) > 0:
            # IoU cost
            iou_matrix = iou_batch(boxes, predicted)
            iou_cost = 1.0 - iou_matrix

            # Appearance cost
            track_ids = [t.track_id for t in self.tracks]
            app_cost = self._nn_cosine_distance(det_embeddings, track_ids)

            # Combined (weighted)
            cost = 0.5 * iou_cost + 0.5 * app_cost

            # Gates
            gate = (iou_matrix < self.iou_threshold) | (app_cost > self.max_cosine_distance)
            cost[gate] = 1e5

            # Hungarian
            det_idx, trk_idx = linear_sum_assignment(cost)
            matches, unmatched_dets, unmatched_trks = [], list(range(len(boxes))), list(range(len(self.tracks)))
            for d, t in zip(det_idx, trk_idx):
                if cost[d, t] < 1e5:
                    matches.append((d, t))
                    unmatched_dets.remove(d)
                    unmatched_trks.remove(t)
        else:
            matches = []
            unmatched_dets = list(range(len(boxes)))
            unmatched_trks = list(range(len(self.tracks)))

        # ── 4. Update matched tracks ────────────────────────────────────
        for d, t in matches:
            self.tracks[t].update(
                boxes[d],
                score=float(scores[d]) if len(scores) > d else 0.0,
                class_id=int(class_ids[d]) if len(class_ids) > d else 0,
            )
            self.tracks[t].embedding = det_embeddings[d]
            self._update_gallery(self.tracks[t].track_id, det_embeddings[d])

        # ── 5. Create new tracks ────────────────────────────────────────
        for d in unmatched_dets:
            t = Track(
                bbox=boxes[d],
                score=float(scores[d]) if len(scores) > d else 0.0,
                class_id=int(class_ids[d]) if len(class_ids) > d else 0,
                **self._kf_params,
            )
            t.embedding = det_embeddings[d]
            self._update_gallery(t.track_id, det_embeddings[d])
            self.tracks.append(t)

        # ── 6. Remove stale tracks ──────────────────────────────────────
        active = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                active.append(trk)
            else:
                self._gallery.pop(trk.track_id, None)
        self.tracks = active

        # ── 7. Return confirmed ─────────────────────────────────────────
        return [t for t in self.tracks if t.hits >= self.min_hits or self._frame_count <= self.min_hits]

    def reset(self) -> None:
        """Reset all tracks, gallery, and counters."""
        self.tracks.clear()
        self._gallery.clear()
        self._frame_count = 0
        self.id_switches = 0
        Track.reset_id_counter()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DeepSORTTracker":
        """Construct from config dictionary."""
        trk_cfg = cfg.get("tracking", {})
        ds_cfg = trk_cfg.get("deepsort", {})
        kalman_cfg = trk_cfg.get("kalman", {})
        return cls(
            max_age=trk_cfg.get("max_age", 30),
            min_hits=trk_cfg.get("min_hits", 3),
            iou_threshold=trk_cfg.get("iou_threshold", 0.3),
            max_cosine_distance=ds_cfg.get("max_cosine_distance", 0.4),
            nn_budget=ds_cfg.get("nn_budget", 100),
            embedding_model=ds_cfg.get("embedding_model"),
            kalman_params={
                "process_noise": kalman_cfg.get("process_noise", 1.0),
                "measurement_noise": kalman_cfg.get("measurement_noise", 1.0),
                "estimation_error": kalman_cfg.get("estimation_error", 10.0),
            },
        )
