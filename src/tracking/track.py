"""Single-object Track representation with Kalman filter state."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.filters.kalman_filter import KalmanBoxFilter


class Track:
    """Represents a single tracked object with Kalman-filtered state.

    Attributes:
        track_id: Unique track identifier.
        bbox: Current bounding box ``[x1, y1, x2, y2]``.
        score: Last detection confidence.
        class_id: Object class index.
        hits: Total number of matched detections.
        age: Frames since track creation.
        time_since_update: Consecutive frames without a matched detection.
        history: Deque of past center positions ``(cx, cy)`` for trail drawing.
    """

    _next_id: int = 1

    def __init__(
        self,
        bbox: np.ndarray,
        score: float = 0.0,
        class_id: int = 0,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        estimation_error: float = 10.0,
        track_id: Optional[int] = None,
    ) -> None:
        self.track_id = track_id if track_id is not None else Track._next_id
        if track_id is None:
            Track._next_id += 1

        self.score = score
        self.class_id = class_id
        self.hits: int = 1
        self.age: int = 0
        self.time_since_update: int = 0

        self._kf = KalmanBoxFilter(
            bbox,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            estimation_error=estimation_error,
        )
        self.bbox = bbox.copy().astype(np.float64)

        # Motion trail
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history: deque = deque(maxlen=60)
        self.history.append((cx, cy))

        # Optional appearance embedding
        self.embedding: Optional[np.ndarray] = None

    def predict(self) -> np.ndarray:
        """Advance the Kalman filter by one timestep and return the predicted bbox."""
        self.age += 1
        self.time_since_update += 1
        self.bbox = self._kf.predict()
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.history.append((cx, cy))
        return self.bbox

    def update(self, bbox: np.ndarray, score: float = 0.0, class_id: int = 0) -> None:
        """Update the track with a matched detection.

        Args:
            bbox: Measured bounding box ``[x1, y1, x2, y2]``.
            score: Detection confidence.
            class_id: Detection class index.
        """
        self.bbox = self._kf.update(bbox)
        self.score = score
        self.class_id = class_id
        self.hits += 1
        self.time_since_update = 0

    @property
    def is_confirmed(self) -> bool:
        """Track has been matched enough times to be considered valid."""
        return self.hits >= 3

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity [vx, vy] from the Kalman filter."""
        return self._kf.velocity

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the global ID counter (for testing / new sessions)."""
        cls._next_id = 1

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, hits={self.hits}, "
            f"age={self.age}, tsu={self.time_since_update})"
        )
