"""SIFT feature extraction using OpenCV."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker

logger = get_logger(__name__)


class SIFTExtractor:
    """Scale-Invariant Feature Transform (SIFT) keypoint and descriptor extractor.

    SIFT is robust to scale and rotation changes, making it suitable for
    re-identification and appearance matching across frames.

    Args:
        max_keypoints: Maximum number of keypoints to retain.
        contrast_threshold: Contrast threshold for filtering weak features.
        edge_threshold: Edge threshold for filtering edge-like features.
    """

    def __init__(
        self,
        max_keypoints: int = 500,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
    ) -> None:
        self.max_keypoints = max_keypoints
        self._sift = cv2.SIFT_create(
            nfeatures=max_keypoints,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )
        self._latency = LatencyTracker("sift")
        logger.debug(f"SIFTExtractor initialized (max_kp={max_keypoints})")

    def extract(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[list, Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors.

        Args:
            frame: BGR input image.
            mask: Optional binary mask (same H×W) — keypoints only in white regions.

        Returns:
            Tuple of (keypoints, descriptors).
            Descriptors shape: ``(N, 128)`` float32 or ``None`` if no keypoints.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._latency:
            keypoints, descriptors = self._sift.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75,
    ) -> list:
        """Match descriptors using Lowe's ratio test.

        Args:
            desc1: Descriptors from frame 1.
            desc2: Descriptors from frame 2.
            ratio_threshold: Lowe's ratio test threshold.

        Returns:
            List of good ``cv2.DMatch`` objects.
        """
        if desc1 is None or desc2 is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_threshold * n.distance:
                    good.append(m)
        return good

    @property
    def latency_ms(self) -> float:
        return self._latency.last_ms
