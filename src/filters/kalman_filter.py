"""2-D Kalman filter for bounding-box tracking (constant-velocity model).

State vector: [cx, cy, s, r, vx, vy, vs, 0]
where:
    cx, cy = center position
    s      = scale (area)
    r      = aspect ratio (width / height)  — modeled as constant
    vx, vy = velocity of center
    vs     = velocity of scale
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import block_diag


def _xyxy_to_xsr(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # area
    r = w / max(h, 1e-6)  # aspect ratio
    return np.array([cx, cy, s, r], dtype=np.float64)


def _xsr_to_xyxy(state: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, s, r] to [x1, y1, x2, y2]."""
    cx, cy, s, r = state[:4]
    s = max(s, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ], dtype=np.float64)


class KalmanBoxFilter:
    """Linear Kalman filter for tracking a single bounding box.

    Implements a constant-velocity model in the ``[cx, cy, s, r]`` space
    (center-x, center-y, scale/area, aspect-ratio).

    Args:
        bbox: Initial bounding box ``[x1, y1, x2, y2]``.
        process_noise: Process noise gain.
        measurement_noise: Measurement noise gain.
        estimation_error: Initial estimation error covariance gain.
    """

    _DIM_X = 8  # state dimension
    _DIM_Z = 4  # measurement dimension

    def __init__(
        self,
        bbox: np.ndarray,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        estimation_error: float = 10.0,
    ) -> None:
        # State transition matrix (constant velocity)
        self.F = np.eye(self._DIM_X, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity * dt

        # Measurement matrix
        self.H = np.eye(self._DIM_Z, self._DIM_X, dtype=np.float64)

        # Process noise
        q_pos = np.diag([1, 1, 1, 1]).astype(np.float64) * process_noise
        q_vel = np.diag([0.01, 0.01, 0.0001, 0]).astype(np.float64) * process_noise
        self.Q = block_diag(q_pos, q_vel)

        # Measurement noise
        self.R = np.diag([1, 1, 10, 1]).astype(np.float64) * measurement_noise

        # State estimate and covariance
        z = _xyxy_to_xsr(bbox)
        self.x = np.zeros(self._DIM_X, dtype=np.float64)
        self.x[:4] = z
        self.P = np.eye(self._DIM_X, dtype=np.float64) * estimation_error
        # High uncertainty on velocities initially
        self.P[4:, 4:] *= 100.0

    def predict(self) -> np.ndarray:
        """Predict the next state and return the predicted bbox ``[x1, y1, x2, y2]``.

        Returns:
            Predicted bounding box ``(4,)`` float64.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Ensure area stays positive
        self.x[2] = max(self.x[2], 1e-6)
        return self.bbox

    def update(self, bbox: np.ndarray) -> np.ndarray:
        """Update the state with a new measurement.

        Args:
            bbox: Measured bounding box ``[x1, y1, x2, y2]``.

        Returns:
            Updated bounding box ``(4,)`` float64.
        """
        z = _xyxy_to_xsr(bbox)
        y = z - self.H @ self.x           # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(self._DIM_X) - K @ self.H
        self.P = I_KH @ self.P
        return self.bbox

    @property
    def bbox(self) -> np.ndarray:
        """Current state as ``[x1, y1, x2, y2]``."""
        return _xsr_to_xyxy(self.x)

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity ``[vx, vy]``."""
        return self.x[4:6].copy()
