#!/usr/bin/env python3
"""Benchmarking tool — measures FPS, latency, mAP@0.5 (simulated), and ID switches.

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --source video.mp4 --frames 500
    python benchmarks/benchmark.py --report benchmarks/results.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import numpy as np
from tabulate import tabulate

from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.utils.timer import FPSCounter, LatencyTracker
from src.pipeline.pipeline_config import PipelineConfig

logger = get_logger("benchmark")


def _simulate_map(num_detections: int, base_map: float = 0.88) -> float:
    """Simulate mAP@0.5 with slight random variance.

    In a full evaluation, this would compare predictions against ground truth
    annotations using the COCO evaluation protocol. Here we simulate realistic
    values for demonstration.
    """
    noise = random.gauss(0, 0.02)
    # Higher detection count → slightly better mAP (more true positives)
    bonus = min(num_detections * 0.001, 0.03)
    return max(0.0, min(1.0, base_map + noise + bonus))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--source", "-s", default="0", help="Video source (file or camera index).")
@click.option("--config", "-c", "config_path", default=None, help="Config file path.")
@click.option("--warmup", default=30, type=int, help="Warmup frames to skip.")
@click.option("--frames", "-n", default=300, type=int, help="Frames to benchmark.")
@click.option("--report", "-r", default="benchmarks/results.json", help="Output JSON report.")
@click.option("--tracker", type=click.Choice(["sort", "deepsort"]), default=None)
@click.option("--backend", type=click.Choice(["pytorch", "onnx"]), default=None)
def main(
    source: str,
    config_path: str | None,
    warmup: int,
    frames: int,
    report: str,
    tracker: str | None,
    backend: str | None,
) -> None:
    """Benchmark the detection + tracking pipeline."""
    import cv2
    from src.pipeline.hybrid_pipeline import HybridPipeline

    logger.info("[bold green]Pipeline Benchmark[/]")

    cfg = load_config(config_path)
    overrides: dict = {"pipeline": {"source": source, "show_display": False, "save_video": False}}
    if tracker:
        overrides["tracking"] = {"tracker_type": tracker}
    if backend:
        overrides["detection"] = {"backend": backend}
    cfg = merge_configs(cfg, overrides)

    pipeline = HybridPipeline(cfg)
    pcfg = PipelineConfig.from_dict(cfg)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        return

    fps_counter = FPSCounter()
    det_latency = LatencyTracker("det_bench")
    trk_latency = LatencyTracker("trk_bench")

    total_detections = 0
    frame_idx = 0
    phase = "warmup"

    logger.info(f"Warmup: {warmup} frames, Benchmark: {frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx <= warmup:
            # Warmup pass
            pipeline.detector.detect(frame)
            continue

        if frame_idx > warmup + frames:
            break

        if phase == "warmup":
            phase = "benchmark"
            logger.info("[bold]Benchmark phase started[/]")

        fps_counter.tick()

        with det_latency:
            dets = pipeline.detector.detect(frame)

        with trk_latency:
            if hasattr(pipeline.tracker, "update"):
                from src.tracking.deep_sort_tracker import DeepSORTTracker
                if isinstance(pipeline.tracker, DeepSORTTracker):
                    pipeline.tracker.update(dets, frame=frame)
                else:
                    pipeline.tracker.update(dets)

        total_detections += len(dets.get("boxes", []))

    cap.release()

    bench_frames = frame_idx - warmup
    if bench_frames <= 0:
        logger.warning("No benchmark frames processed.")
        return

    # ── Compute metrics ──────────────────────────────────────────────────
    avg_fps = fps_counter.avg_fps
    avg_det_ms = det_latency.avg_ms
    avg_trk_ms = trk_latency.avg_ms
    sim_map = _simulate_map(total_detections)
    id_switches = getattr(pipeline.tracker, "id_switches", 0)

    results: Dict[str, Any] = {
        "source": source,
        "backend": pcfg.backend,
        "tracker": pcfg.tracker_type,
        "device": pcfg.device,
        "benchmark_frames": bench_frames,
        "avg_fps": round(avg_fps, 2),
        "avg_detection_latency_ms": round(avg_det_ms, 2),
        "avg_tracking_latency_ms": round(avg_trk_ms, 2),
        "total_pipeline_latency_ms": round(avg_det_ms + avg_trk_ms, 2),
        "mAP_at_0.5_simulated": round(sim_map, 4),
        "total_detections": total_detections,
        "id_switches": id_switches,
    }

    # ── Display ──────────────────────────────────────────────────────────
    table = [[k, v] for k, v in results.items()]
    logger.info("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # ── Save report ──────────────────────────────────────────────────────
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved to [bold]{report_path}[/]")


if __name__ == "__main__":
    main()
