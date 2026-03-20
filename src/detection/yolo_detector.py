"""YOLOv8 detector using the Ultralytics PyTorch backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker

logger = get_logger(__name__)


class YOLODetector:
    """Wrapper around ``ultralytics.YOLO`` for real-time inference.

    Args:
        model_path: Path to YOLOv8 ``.pt`` weights.
        confidence: Minimum detection confidence.
        nms_thresh: NMS IoU threshold.
        input_size: Model input resolution ``(width, height)``.
        device: Inference device (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        classes: Optional list of class indices to filter.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence: float = 0.45,
        nms_thresh: float = 0.50,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cpu",
        classes: Optional[List[int]] = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.device = device
        self.classes = classes

        logger.info(f"Loading YOLOv8 model from [bold]{model_path}[/] on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self._latency = LatencyTracker("yolo_detect")

        # Cache class names
        self.class_names: List[str] = []
        if hasattr(self.model, "names"):
            names = self.model.names
            if isinstance(names, dict):
                self.class_names = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, (list, tuple)):
                self.class_names = list(names)

        logger.info(f"YOLOv8 ready — {len(self.class_names)} classes")

    def detect(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run detection on a single BGR frame.

        Args:
            frame: Input image ``(H, W, 3)`` in BGR format.

        Returns:
            Dictionary with keys:
            - ``boxes``: ``(N, 4)`` float32 ``[x1, y1, x2, y2]``
            - ``scores``: ``(N,)`` float32 confidence scores
            - ``class_ids``: ``(N,)`` int32 class indices
        """
        with self._latency:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.nms_thresh,
                imgsz=self.input_size[0],
                device=self.device,
                classes=self.classes,
                verbose=False,
            )

        # Parse results
        if results and len(results) > 0:
            result = results[0]
            boxes_tensor = result.boxes
            boxes = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)
            scores = boxes_tensor.conf.cpu().numpy().astype(np.float32)
            class_ids = boxes_tensor.cls.cpu().numpy().astype(np.int32)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)

        return {"boxes": boxes, "scores": scores, "class_ids": class_ids}

    @property
    def latency_ms(self) -> float:
        """Last inference latency in milliseconds."""
        return self._latency.last_ms

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "YOLODetector":
        """Construct a detector from a config dictionary (``detection`` section)."""
        det_cfg = cfg.get("detection", {})
        input_size = tuple(det_cfg.get("input_size", [640, 640]))
        return cls(
            model_path=det_cfg.get("model_path", "models/yolov8n.pt"),
            confidence=det_cfg.get("confidence_threshold", 0.45),
            nms_thresh=det_cfg.get("nms_threshold", 0.50),
            input_size=input_size,
            device=det_cfg.get("device", "cpu"),
            classes=det_cfg.get("classes"),
        )
