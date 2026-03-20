"""Pipeline configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PipelineConfig:
    """Typed configuration for the hybrid detection + tracking pipeline.

    Can be constructed directly or via :meth:`from_dict` using a parsed
    YAML configuration dictionary.
    """

    # ── Detection ────────────────────────────────────────────────────────
    model_path: str = "models/yolov8n.pt"
    onnx_path: str = "models/yolov8n.onnx"
    backend: str = "pytorch"  # pytorch | onnx
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.50
    input_size: Tuple[int, int] = (640, 640)
    device: str = "cpu"
    classes: Optional[List[int]] = None

    # ── Tracking ─────────────────────────────────────────────────────────
    tracker_type: str = "sort"  # sort | deepsort
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    process_noise: float = 1.0
    measurement_noise: float = 1.0
    estimation_error: float = 10.0
    max_cosine_distance: float = 0.4
    nn_budget: int = 100

    # ── Features ─────────────────────────────────────────────────────────
    feature_extractor: str = "orb"
    max_keypoints: int = 500
    optical_flow_enabled: bool = True

    # ── Pipeline ─────────────────────────────────────────────────────────
    source: str = "0"
    output_dir: str = "data/output"
    show_display: bool = True
    save_video: bool = False
    save_fps: int = 25
    log_level: str = "INFO"
    skip_frames: int = 0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PipelineConfig":
        """Construct from a nested config dictionary (as loaded from YAML)."""
        det = cfg.get("detection", {})
        trk = cfg.get("tracking", {})
        kal = trk.get("kalman", {})
        ds = trk.get("deepsort", {})
        feat = cfg.get("features", {})
        oflow = feat.get("optical_flow", {})
        pipe = cfg.get("pipeline", {})

        input_size = det.get("input_size", [640, 640])
        if isinstance(input_size, list):
            input_size = tuple(input_size)

        return cls(
            model_path=det.get("model_path", cls.model_path),
            onnx_path=det.get("onnx_path", cls.onnx_path),
            backend=det.get("backend", cls.backend),
            confidence_threshold=det.get("confidence_threshold", cls.confidence_threshold),
            nms_threshold=det.get("nms_threshold", cls.nms_threshold),
            input_size=input_size,
            device=det.get("device", cls.device),
            classes=det.get("classes"),
            tracker_type=trk.get("tracker_type", cls.tracker_type),
            max_age=trk.get("max_age", cls.max_age),
            min_hits=trk.get("min_hits", cls.min_hits),
            iou_threshold=trk.get("iou_threshold", cls.iou_threshold),
            process_noise=kal.get("process_noise", cls.process_noise),
            measurement_noise=kal.get("measurement_noise", cls.measurement_noise),
            estimation_error=kal.get("estimation_error", cls.estimation_error),
            max_cosine_distance=ds.get("max_cosine_distance", cls.max_cosine_distance),
            nn_budget=ds.get("nn_budget", cls.nn_budget),
            feature_extractor=feat.get("extractor", cls.feature_extractor),
            max_keypoints=feat.get("max_keypoints", cls.max_keypoints),
            optical_flow_enabled=oflow.get("enabled", cls.optical_flow_enabled),
            source=str(pipe.get("source", cls.source)),
            output_dir=pipe.get("output_dir", cls.output_dir),
            show_display=pipe.get("show_display", cls.show_display),
            save_video=pipe.get("save_video", cls.save_video),
            save_fps=pipe.get("save_fps", cls.save_fps),
            log_level=pipe.get("log_level", cls.log_level),
            skip_frames=pipe.get("skip_frames", cls.skip_frames),
        )
