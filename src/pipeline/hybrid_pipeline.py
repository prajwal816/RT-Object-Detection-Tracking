"""Hybrid Detection + Tracking Pipeline.

Master pipeline that orchestrates:
    Frame capture → Detection → Tracking → Visualization → Metrics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.detection.yolo_detector import YOLODetector
from src.detection.onnx_detector import ONNXDetector
from src.features.optical_flow import LucasKanadeFlow
from src.tracking.sort_tracker import SORTTracker
from src.tracking.deep_sort_tracker import DeepSORTTracker
from src.pipeline.pipeline_config import PipelineConfig
from src.utils.logger import get_logger
from src.utils.timer import FPSCounter, LatencyTracker
from src.utils.visualization import draw_tracks, draw_hud
from src.utils.threaded_capture import ThreadedCapture


class HybridPipeline:
    """Real-time detection + tracking pipeline.

    Supports:
    - YOLOv8 (PyTorch) or ONNX Runtime detection backends
    - SORT or DeepSORT tracking
    - Optional optical flow for inter-frame motion estimation
    - Live display with HUD overlay
    - Video output recording

    Args:
        config: ``PipelineConfig`` instance or raw config dict.
    """

    def __init__(self, config: PipelineConfig | Dict[str, Any]) -> None:
        if isinstance(config, dict):
            config = PipelineConfig.from_dict(config)
        self.cfg = config

        self.logger = get_logger("pipeline", level=config.log_level)
        self.fps_counter = FPSCounter()
        self._det_latency = LatencyTracker("detection")
        self._trk_latency = LatencyTracker("tracking")

        # ── Detector ─────────────────────────────────────────────────────
        self.logger.info(f"Backend: [bold]{config.backend}[/]")
        if config.backend == "onnx":
            self.detector = ONNXDetector(
                onnx_path=config.onnx_path,
                confidence=config.confidence_threshold,
                nms_thresh=config.nms_threshold,
                input_size=config.input_size,
            )
        else:
            self.detector = YOLODetector(
                model_path=config.model_path,
                confidence=config.confidence_threshold,
                nms_thresh=config.nms_threshold,
                input_size=config.input_size,
                device=config.device,
                classes=config.classes,
            )

        # ── Tracker ──────────────────────────────────────────────────────
        kalman_params = {
            "process_noise": config.process_noise,
            "measurement_noise": config.measurement_noise,
            "estimation_error": config.estimation_error,
        }
        if config.tracker_type == "deepsort":
            self.tracker = DeepSORTTracker(
                max_age=config.max_age,
                min_hits=config.min_hits,
                iou_threshold=config.iou_threshold,
                max_cosine_distance=config.max_cosine_distance,
                nn_budget=config.nn_budget,
                kalman_params=kalman_params,
            )
        else:
            self.tracker = SORTTracker(
                max_age=config.max_age,
                min_hits=config.min_hits,
                iou_threshold=config.iou_threshold,
                kalman_params=kalman_params,
            )

        # ── Optical flow (optional) ──────────────────────────────────────
        self.optical_flow: Optional[LucasKanadeFlow] = None
        if config.optical_flow_enabled:
            self.optical_flow = LucasKanadeFlow()

        # ── Video writer ─────────────────────────────────────────────────
        self._writer: Optional[cv2.VideoWriter] = None

        self.logger.info(
            f"Pipeline ready — tracker=[bold]{config.tracker_type}[/], "
            f"source=[bold]{config.source}[/]"
        )

    # ── Run loop ─────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """Execute the pipeline loop.

        Opens the video source, processes frames, and returns metrics
        when the stream ends or the user presses 'q'.

        Returns:
            Dictionary with ``avg_fps``, ``total_frames``, ``id_switches``.
        """
        source = self.cfg.source
        use_threaded = getattr(self.cfg, 'use_threaded_capture', False)

        if use_threaded:
            cap = ThreadedCapture(source)
            cap.start()
            frame_w, frame_h = cap.width, cap.height
        else:
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not cap.isOpened():
                self.logger.error(f"Cannot open source: {source}")
                return {"avg_fps": 0, "total_frames": 0, "id_switches": 0}
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"Source opened: {frame_w}×{frame_h}")

        if self.cfg.save_video:
            out_path = Path(self.cfg.output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(out_path / "output.mp4"),
                fourcc, self.cfg.save_fps,
                (frame_w, frame_h),
            )

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of stream.")
                    break

                frame_idx += 1
                self.fps_counter.tick()

                # ── Detection ────────────────────────────────────────────
                run_detection = (
                    self.cfg.skip_frames == 0
                    or frame_idx % (self.cfg.skip_frames + 1) == 1
                )

                if run_detection:
                    with self._det_latency:
                        detections = self.detector.detect(frame)
                else:
                    detections = {
                        "boxes": np.empty((0, 4), dtype=np.float32),
                        "scores": np.empty((0,), dtype=np.float32),
                        "class_ids": np.empty((0,), dtype=np.int32),
                    }

                # ── Tracking ─────────────────────────────────────────────
                with self._trk_latency:
                    if isinstance(self.tracker, DeepSORTTracker):
                        active_tracks = self.tracker.update(detections, frame=frame)
                    else:
                        active_tracks = self.tracker.update(detections)

                # ── Visualization ────────────────────────────────────────
                display_frame = frame.copy()
                draw_tracks(display_frame, active_tracks)
                draw_hud(
                    display_frame,
                    fps=self.fps_counter.fps,
                    num_tracks=len(active_tracks),
                    latency_ms=self._det_latency.last_ms + self._trk_latency.last_ms,
                )

                if self.cfg.show_display:
                    cv2.imshow("RT Detection & Tracking", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        self.logger.info("User quit.")
                        break

                if self._writer is not None:
                    self._writer.write(display_frame)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            if use_threaded:
                cap.stop()
            else:
                cap.release()
            if self._writer is not None:
                self._writer.release()
            cv2.destroyAllWindows()

        metrics = {
            "avg_fps": round(self.fps_counter.avg_fps, 2),
            "total_frames": frame_idx,
            "id_switches": getattr(self.tracker, "id_switches", 0),
        }
        self.logger.info(f"Pipeline finished — {metrics}")
        return metrics

    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame (for programmatic use).

        Args:
            frame: BGR image.

        Returns:
            Tuple of ``(annotated_frame, active_tracks, detections)``.
        """
        self.fps_counter.tick()

        with self._det_latency:
            detections = self.detector.detect(frame)

        with self._trk_latency:
            if isinstance(self.tracker, DeepSORTTracker):
                active_tracks = self.tracker.update(detections, frame=frame)
            else:
                active_tracks = self.tracker.update(detections)

        display = frame.copy()
        draw_tracks(display, active_tracks)
        draw_hud(
            display,
            fps=self.fps_counter.fps,
            num_tracks=len(active_tracks),
            latency_ms=self._det_latency.last_ms + self._trk_latency.last_ms,
        )
        return display, active_tracks, detections
