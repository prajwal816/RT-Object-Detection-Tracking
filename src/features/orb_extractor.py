"""ORB feature extraction using OpenCV — faster alternative to SIFT."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker

logger = get_logger(__name__)


class ORBExtractor:
    """Oriented FAST and Rotated BRIEF (ORB) keypoint and descriptor extractor.

    ORB is patent-free and ~10× faster than SIFT, making it ideal for
    real-time applications on edge devices.

    Args:
        max_keypoints: Maximum number of features to retain.
        scale_factor: Pyramid decimation ratio (>1).
        n_levels: Number of pyramid levels.
    """

    def __init__(
        self,
        max_keypoints: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
    ) -> None:
        self.max_keypoints = max_keypoints
        self._orb = cv2.ORB_create(
            nfeatures=max_keypoints,
            scaleFactor=scale_factor,
            nlevels=n_levels,
        )
        self._latency = LatencyTracker("orb")
        logger.debug(f"ORBExtractor initialized (max_kp={max_keypoints})")

    def extract(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[list, Optional[np.ndarray]]:
        """Detect keypoints and compute binary descriptors.

        Args:
            frame: BGR input image.
            mask: Optional binary mask.

        Returns:
            Tuple of (keypoints, descriptors).
            Descriptors shape: ``(N, 32)`` uint8 or ``None``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._latency:
            keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        max_distance: int = 50,
    ) -> list:
        """Match binary descriptors using Hamming distance.

        Args:
            desc1: Descriptors from frame 1.
            desc2: Descriptors from frame 2.
            max_distance: Maximum Hamming distance for a valid match.

        Returns:
            List of ``cv2.DMatch`` objects below the distance threshold.
        """
        if desc1 is None or desc2 is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        return [m for m in matches if m.distance < max_distance]

    @property
    def latency_ms(self) -> float:
        return self._latency.last_ms
