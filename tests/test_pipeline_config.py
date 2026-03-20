"""Unit tests for the pipeline configuration."""

from __future__ import annotations

import pytest

from src.pipeline.pipeline_config import PipelineConfig
from src.utils.config import load_config


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.backend == "pytorch"
        assert cfg.tracker_type == "sort"
        assert cfg.confidence_threshold == 0.45
        assert cfg.input_size == (640, 640)

    def test_from_dict_basic(self):
        d = {
            "detection": {"backend": "onnx", "confidence_threshold": 0.6},
            "tracking": {"tracker_type": "deepsort", "max_age": 50},
        }
        cfg = PipelineConfig.from_dict(d)
        assert cfg.backend == "onnx"
        assert cfg.confidence_threshold == 0.6
        assert cfg.tracker_type == "deepsort"
        assert cfg.max_age == 50

    def test_from_yaml(self):
        """Load from actual default.yaml config."""
        raw = load_config()
        cfg = PipelineConfig.from_dict(raw)
        assert cfg.model_path == "models/yolov8n.pt"
        assert cfg.device == "cpu"
        assert cfg.show_display is True

    def test_input_size_list_to_tuple(self):
        d = {"detection": {"input_size": [320, 320]}}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.input_size == (320, 320)

    def test_kalman_params(self):
        d = {
            "tracking": {
                "kalman": {
                    "process_noise": 5.0,
                    "measurement_noise": 0.1,
                }
            }
        }
        cfg = PipelineConfig.from_dict(d)
        assert cfg.process_noise == 5.0
        assert cfg.measurement_noise == 0.1
