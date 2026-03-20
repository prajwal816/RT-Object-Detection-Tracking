"""Lucas-Kanade sparse optical flow for motion estimation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker

logger = get_logger(__name__)


class LucasKanadeFlow:
    """Sparse optical flow using the Lucas-Kanade method.

    Tracks selected points between consecutive frames to estimate motion
    vectors.  Useful for predicting object movement between detection frames.

    Args:
        win_size: Window size for the optical flow calculation.
        max_level: Maximum pyramid level.
        criteria_eps: Convergence epsilon.
        criteria_max_iter: Maximum iterations per pyramid level.
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (15, 15),
        max_level: int = 3,
        criteria_eps: float = 0.03,
        criteria_max_iter: int = 10,
    ) -> None:
        self.win_size = win_size
        self.max_level = max_level
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            criteria_max_iter,
            criteria_eps,
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._latency = LatencyTracker("optical_flow")

    def compute(
        self,
        frame: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute optical flow for *points* from the previous frame to *frame*.

        Args:
            frame: Current BGR frame.
            points: (N, 2) float32 array of points to track in the previous frame.

        Returns:
            Tuple of:
            - ``new_points``: (M, 2) tracked point positions in the current frame.
            - ``status``: (M,) uint8 — 1 if the flow for the point was found.
            - ``errors``: (M,) float32 — error measure for each point.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or len(points) == 0:
            self._prev_gray = gray
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)

        pts = points.reshape(-1, 1, 2).astype(np.float32)

        with self._latency:
            new_pts, status, errors = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                pts,
                None,
                winSize=self.win_size,
                maxLevel=self.max_level,
                criteria=self._criteria,
            )

        self._prev_gray = gray

        status = status.flatten()
        errors = errors.flatten()
        new_pts = new_pts.reshape(-1, 2)

        return new_pts, status, errors

    def compute_from_boxes(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """Estimate motion vectors for bounding box centers.

        Args:
            frame: Current BGR frame.
            boxes: (N, 4) array in ``[x1, y1, x2, y2]`` format.

        Returns:
            (N, 2) displacement vectors ``(dx, dy)`` for each box center.
        """
        if len(boxes) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Box centers
        centers = np.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ], axis=1).astype(np.float32)

        new_pts, status, _ = self.compute(frame, centers)

        displacements = np.zeros_like(centers)
        valid = status.astype(bool)
        if np.any(valid) and len(new_pts) == len(centers):
            displacements[valid] = new_pts[valid] - centers[valid]

        return displacements

    def reset(self) -> None:
        """Reset internal state (previous frame)."""
        self._prev_gray = None

    @property
    def latency_ms(self) -> float:
        return self._latency.last_ms

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LucasKanadeFlow":
        """Construct from config dictionary."""
        flow_cfg = cfg.get("features", {}).get("optical_flow", {})
        return cls(
            win_size=tuple(flow_cfg.get("win_size", [15, 15])),
            max_level=flow_cfg.get("max_level", 3),
            criteria_eps=flow_cfg.get("criteria_eps", 0.03),
            criteria_max_iter=flow_cfg.get("criteria_max_iter", 10),
        )
