"""Association utilities — IoU computation and Hungarian matching."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_a: np.ndarray, bb_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bb_a: (N, 4) boxes in ``[x1, y1, x2, y2]`` format.
        bb_b: (M, 4) boxes in ``[x1, y1, x2, y2]`` format.

    Returns:
        (N, M) IoU matrix.
    """
    N = bb_a.shape[0]
    M = bb_b.shape[0]

    # Expand dims for broadcasting
    a = bb_a[:, np.newaxis, :]  # (N, 1, 4)
    b = bb_b[np.newaxis, :, :]  # (1, M, 4)

    # Intersection
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    # Union
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter

    return inter / np.maximum(union, 1e-6)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance between embedding matrices.

    Args:
        a: (N, D) embeddings.
        b: (M, D) embeddings.

    Returns:
        (N, M) cosine distance matrix (0 = identical, 2 = opposite).
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    similarity = a_norm @ b_norm.T
    return 1.0 - similarity


def associate_detections_to_tracks(
    detections: np.ndarray,
    tracks: np.ndarray,
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match detections to tracks using the Hungarian algorithm on IoU cost.

    Args:
        detections: (N, 4) detection boxes ``[x1, y1, x2, y2]``.
        tracks: (M, 4) predicted track boxes ``[x1, y1, x2, y2]``.
        iou_threshold: Minimum IoU for a valid match.

    Returns:
        Tuple of:
        - ``matches``: list of ``(detection_idx, track_idx)`` pairs.
        - ``unmatched_dets``: list of unmatched detection indices.
        - ``unmatched_trks``: list of unmatched track indices.
    """
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))
    if len(tracks) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = iou_batch(detections, tracks)
    cost_matrix = 1.0 - iou_matrix

    # Hungarian assignment
    det_indices, trk_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(tracks)))

    for d_idx, t_idx in zip(det_indices, trk_indices):
        if iou_matrix[d_idx, t_idx] >= iou_threshold:
            matches.append((d_idx, t_idx))
            unmatched_dets.remove(d_idx)
            unmatched_trks.remove(t_idx)

    return matches, unmatched_dets, unmatched_trks
