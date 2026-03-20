"""Visualization helpers — draw bounding boxes, trails, IDs, and HUD overlays."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Color palette (BGR) ──────────────────────────────────────────────────────
_PALETTE = [
    (75, 25, 230),   # red
    (48, 130, 245),  # orange
    (25, 225, 255),  # yellow
    (60, 245, 210),  # lime
    (75, 180, 60),   # green
    (200, 130, 0),   # teal
    (240, 110, 70),  # blue
    (230, 50, 240),  # magenta
    (190, 75, 190),  # purple
    (80, 190, 230),  # gold
]


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic color for a given track ID."""
    return _PALETTE[track_id % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw detection bounding boxes on *frame* (in-place).

    Args:
        frame: BGR image (H, W, 3).
        boxes: (N, 4) array in ``[x1, y1, x2, y2]`` format.
        scores: (N,) confidence scores.
        class_ids: (N,) integer class indices.
        class_names: List mapping class index → readable name.
        color: Default box color (BGR).
        thickness: Line thickness in pixels.

    Returns:
        Annotated frame.
    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label_parts: list[str] = []
        if class_ids is not None:
            cid = int(class_ids[i])
            name = class_names[cid] if class_names and cid < len(class_names) else str(cid)
            label_parts.append(name)
        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return frame


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    trail_length: int = 30,
    thickness: int = 2,
) -> np.ndarray:
    """Draw tracked object boxes, IDs, and motion trails.

    Each element in *tracks* must expose ``.track_id``, ``.bbox``
    ``[x1, y1, x2, y2]``, and ``.history`` (list of ``(cx, cy)``).

    Args:
        frame: BGR image.
        tracks: List of Track objects.
        trail_length: Max points in trail.
        thickness: Box line thickness.

    Returns:
        Annotated frame.
    """
    for track in tracks:
        tid = track.track_id
        color = _color_for_id(tid)
        x1, y1, x2, y2 = map(int, track.bbox[:4])

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ID label
        label = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Trail
        history = list(track.history)[-trail_length:]
        for j in range(1, len(history)):
            alpha = j / len(history)
            pt1 = (int(history[j - 1][0]), int(history[j - 1][1]))
            pt2 = (int(history[j][0]), int(history[j][1]))
            line_thick = max(1, int(thickness * alpha))
            cv2.line(frame, pt1, pt2, color, line_thick, cv2.LINE_AA)

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float = 0.0,
    num_tracks: int = 0,
    latency_ms: float = 0.0,
) -> np.ndarray:
    """Draw a heads-up display overlay with performance metrics.

    Args:
        frame: BGR image.
        fps: Current frames per second.
        num_tracks: Number of active tracks.
        latency_ms: Inference latency in milliseconds.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (220, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"FPS:     {fps:.1f}",
        f"Tracks:  {num_tracks}",
        f"Latency: {latency_ms:.1f} ms",
    ]
    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (16, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1, cv2.LINE_AA,
        )
    return frame
