"""FPS counter and latency profiling utilities."""

from __future__ import annotations

import functools
import time
from collections import deque
from typing import Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FPSCounter:
    """Rolling-window FPS counter for real-time performance monitoring.

    Args:
        window_size: Number of frames to average over.
    """

    def __init__(self, window_size: int = 60) -> None:
        self._window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._frame_count: int = 0
        self._start_time: Optional[float] = None

    def tick(self) -> None:
        """Record a frame timestamp."""
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now
        self._timestamps.append(now)
        self._frame_count += 1

    @property
    def fps(self) -> float:
        """Current rolling-window FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def avg_fps(self) -> float:
        """Average FPS since the first ``tick()``."""
        if self._start_time is None or self._frame_count < 2:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def frame_time_ms(self) -> float:
        """Last frame time in milliseconds."""
        if len(self._timestamps) < 2:
            return 0.0
        return (self._timestamps[-1] - self._timestamps[-2]) * 1000.0

    def reset(self) -> None:
        """Reset all counters."""
        self._timestamps.clear()
        self._frame_count = 0
        self._start_time = None

    def __repr__(self) -> str:
        return f"FPSCounter(fps={self.fps:.1f}, avg={self.avg_fps:.1f})"


class LatencyTracker:
    """Context manager and decorator for measuring execution latency.

    Usage::

        tracker = LatencyTracker("detection")
        with tracker:
            model.predict(frame)
        print(tracker.last_ms)
    """

    def __init__(self, name: str = "op") -> None:
        self.name = name
        self._start: float = 0.0
        self.last_ms: float = 0.0
        self._history: deque[float] = deque(maxlen=200)

    def __enter__(self) -> "LatencyTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.last_ms = (time.perf_counter() - self._start) * 1000.0
        self._history.append(self.last_ms)

    @property
    def avg_ms(self) -> float:
        """Average latency in ms."""
        return sum(self._history) / len(self._history) if self._history else 0.0

    @property
    def min_ms(self) -> float:
        return min(self._history) if self._history else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._history) if self._history else 0.0


def profile(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Decorator that logs function execution time.

    Args:
        func: Function to wrap.
        name: Optional label (defaults to function name).
    """

    def decorator(fn: Callable) -> Callable:
        label = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.debug(f"[bold cyan]{label}[/] took {elapsed_ms:.2f} ms")
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
