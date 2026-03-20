"""Unit tests for the utilities module."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest
import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import FPSCounter, LatencyTracker, profile
from src.utils.config import load_config, merge_configs


class TestLogger:
    """Tests for the logging utility."""

    def test_get_logger_returns_logger(self):
        logger = get_logger("test_logger_1")
        assert logger is not None
        assert logger.name == "test_logger_1"

    def test_get_logger_caches(self):
        logger1 = get_logger("test_cached")
        logger2 = get_logger("test_cached")
        assert logger1 is logger2

    def test_logger_with_file(self):
        tmpdir = tempfile.mkdtemp()
        try:
            log_file = os.path.join(tmpdir, "test.log")
            logger = get_logger("test_file_logger", log_file=log_file)
            logger.info("Test message")
            assert Path(log_file).exists()
        finally:
            # Close file handlers to avoid Windows file-lock errors
            for handler in logger.handlers[:]:
                if hasattr(handler, 'close'):
                    handler.close()
                logger.removeHandler(handler)
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFPSCounter:
    """Tests for the FPS counter."""

    def test_initial_fps_is_zero(self):
        counter = FPSCounter()
        assert counter.fps == 0.0

    def test_fps_after_ticks(self):
        counter = FPSCounter(window_size=10)
        for _ in range(5):
            counter.tick()
            time.sleep(0.01)
        assert counter.fps > 0.0

    def test_avg_fps(self):
        counter = FPSCounter()
        for _ in range(10):
            counter.tick()
            time.sleep(0.01)
        assert counter.avg_fps > 0.0

    def test_reset(self):
        counter = FPSCounter()
        counter.tick()
        counter.tick()
        counter.reset()
        assert counter.fps == 0.0
        assert counter.avg_fps == 0.0

    def test_repr(self):
        counter = FPSCounter()
        assert "FPSCounter" in repr(counter)


class TestLatencyTracker:
    """Tests for the latency context manager."""

    def test_context_manager(self):
        tracker = LatencyTracker("test")
        with tracker:
            time.sleep(0.01)
        assert tracker.last_ms > 0.0

    def test_avg_ms(self):
        tracker = LatencyTracker("test_avg")
        for _ in range(5):
            with tracker:
                time.sleep(0.005)
        assert tracker.avg_ms > 0.0

    def test_min_max(self):
        tracker = LatencyTracker("test_minmax")
        with tracker:
            time.sleep(0.01)
        with tracker:
            time.sleep(0.02)
        assert tracker.min_ms <= tracker.max_ms


class TestProfileDecorator:
    """Tests for the @profile decorator."""

    def test_profile_basic(self):
        @profile
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_profile_with_name(self):
        @profile(name="custom_op")
        def multiply(a, b):
            return a * b

        result = multiply(3, 4)
        assert result == 12


class TestConfig:
    """Tests for the config loader."""

    def test_load_default_config(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "detection" in cfg
        assert "tracking" in cfg
        assert "pipeline" in cfg

    def test_load_nonexistent_returns_empty(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg == {}

    def test_merge_flat(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        merged = merge_configs(base, override)
        assert merged == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        base = {"detection": {"confidence": 0.5, "device": "cpu"}}
        override = {"detection": {"confidence": 0.7}}
        merged = merge_configs(base, override)
        assert merged["detection"]["confidence"] == 0.7
        assert merged["detection"]["device"] == "cpu"

    def test_merge_does_not_mutate(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        merge_configs(base, override)
        assert "c" not in base["a"]
