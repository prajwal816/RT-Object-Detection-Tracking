"""ONNX Runtime detector for edge-optimized YOLOv8 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker

logger = get_logger(__name__)


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return out


class ONNXDetector:
    """YOLOv8 detector using ONNX Runtime for edge inference.

    Args:
        onnx_path: Path to exported ``.onnx`` model.
        confidence: Minimum detection confidence.
        nms_thresh: NMS IoU threshold.
        input_size: Model input resolution ``(width, height)``.
        providers: ORT execution providers (e.g. ``["CUDAExecutionProvider"]``).
    """

    def __init__(
        self,
        onnx_path: str = "models/yolov8n.onnx",
        confidence: float = 0.45,
        nms_thresh: float = 0.50,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = onnx_path
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.input_size = input_size

        if providers is None:
            providers = ort.get_available_providers()

        logger.info(f"Loading ONNX model from [bold]{onnx_path}[/]")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._latency = LatencyTracker("onnx_detect")

        logger.info(f"ONNX session ready — providers: {providers}")

    # ── Preprocessing ────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and transpose a BGR frame to NCHW float32."""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, axis=0)   # → NCHW

    # ── Postprocessing ───────────────────────────────────────────────────
    def _postprocess(
        self,
        output: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> Dict[str, np.ndarray]:
        """Parse raw ONNX output into boxes, scores, class_ids."""
        # YOLOv8 output shape: (1, 84, 8400) → transpose to (8400, 84)
        predictions = output[0].T  # (8400, 84)

        # Split box + class scores
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Best class per detection
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = scores >= self.confidence
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "class_ids": np.empty((0,), dtype=np.int32),
            }

        # Convert to xyxy
        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        # Scale back to original image
        sx = orig_w / self.input_size[0]
        sy = orig_h / self.input_size[1]
        boxes_xyxy[:, [0, 2]] *= sx
        boxes_xyxy[:, [1, 3]] *= sy

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.confidence,
            self.nms_thresh,
        )
        if len(indices) > 0:
            indices = indices.flatten()
            return {
                "boxes": boxes_xyxy[indices].astype(np.float32),
                "scores": scores[indices].astype(np.float32),
                "class_ids": class_ids[indices].astype(np.int32),
            }

        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "class_ids": np.empty((0,), dtype=np.int32),
        }

    # ── Public API ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run detection on a single BGR frame.

        Returns:
            Dictionary with ``boxes``, ``scores``, ``class_ids``.
        """
        orig_h, orig_w = frame.shape[:2]
        blob = self._preprocess(frame)

        with self._latency:
            outputs = self.session.run(self.output_names, {self.input_name: blob})

        return self._postprocess(outputs[0], orig_h, orig_w)

    @property
    def latency_ms(self) -> float:
        return self._latency.last_ms

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ONNXDetector":
        """Construct from config dictionary."""
        det_cfg = cfg.get("detection", {})
        return cls(
            onnx_path=det_cfg.get("onnx_path", "models/yolov8n.onnx"),
            confidence=det_cfg.get("confidence_threshold", 0.45),
            nms_thresh=det_cfg.get("nms_threshold", 0.50),
            input_size=tuple(det_cfg.get("input_size", [640, 640])),
        )
