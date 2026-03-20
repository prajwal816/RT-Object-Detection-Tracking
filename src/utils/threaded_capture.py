"""Multi-threaded video capture for decoupled reading and processing.

Separates frame I/O from the inference pipeline to maximize GPU utilization
and reduce latency caused by blocking ``VideoCapture.read()`` calls.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreadedCapture:
    """Non-blocking video capture using a dedicated reader thread.

    Frames are continuously read in a background thread and buffered.
    The main thread can grab the latest frame without blocking on I/O.

    Args:
        source: Camera index (int) or video file path (str).
        queue_size: Not used (latest-frame mode). Kept for API compat.

    Usage::

        cap = ThreadedCapture("video.mp4")
        cap.start()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # process frame ...
        cap.stop()
    """

    def __init__(self, source: str | int = 0) -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._cap = cv2.VideoCapture(source)
        self._source = source

        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ThreadedCapture":
        """Start the background reader thread.

        Returns:
            Self for chaining.
        """
        if not self._cap.isOpened():
            logger.error(f"Cannot open source: {self._source}")
            return self

        self._stopped.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"ThreadedCapture started — {self.width}×{self.height} "
            f"@ {self.fps:.1f} FPS"
        )
        return self

    def _reader_loop(self) -> None:
        """Continuously read frames in the background."""
        while not self._stopped.is_set():
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                self._stopped.set()
                break

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab the latest frame (non-blocking).

        Returns:
            Tuple of ``(success, frame)``.
        """
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Stop the reader thread and release the capture device."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._cap.release()
        logger.debug("ThreadedCapture stopped.")

    def isOpened(self) -> bool:
        return self._cap.isOpened() and not self._stopped.is_set()

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    def __enter__(self) -> "ThreadedCapture":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()
