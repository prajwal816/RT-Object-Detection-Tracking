#!/usr/bin/env python3
"""CLI entry point for the Real-Time Object Detection & Tracking pipeline.

Usage examples:
    # Webcam (default)
    python scripts/run_pipeline.py

    # Video file with DeepSORT
    python scripts/run_pipeline.py --source video.mp4 --tracker deepsort

    # Custom config, save output
    python scripts/run_pipeline.py --config configs/default.yaml --save --no-show

    # ONNX backend
    python scripts/run_pipeline.py --backend onnx --source video.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.pipeline.hybrid_pipeline import HybridPipeline
from src.pipeline.pipeline_config import PipelineConfig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--source", "-s", default="0", help="Video file path or camera index (default: 0).")
@click.option("--config", "-c", "config_path", default=None, help="YAML config file path.")
@click.option("--tracker", "-t", type=click.Choice(["sort", "deepsort"]), default=None, help="Tracker type.")
@click.option("--backend", "-b", type=click.Choice(["pytorch", "onnx"]), default=None, help="Detection backend.")
@click.option("--model", "-m", default=None, help="Model weights path (.pt or .onnx).")
@click.option("--confidence", default=None, type=float, help="Detection confidence threshold.")
@click.option("--device", "-d", default=None, help="Device (cpu / cuda / cuda:0).")
@click.option("--show/--no-show", default=True, help="Display output window.")
@click.option("--save", is_flag=True, default=False, help="Save output video.")
@click.option("--output-dir", default=None, help="Output directory for saved video.")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None)
def main(
    source: str,
    config_path: str | None,
    tracker: str | None,
    backend: str | None,
    model: str | None,
    confidence: float | None,
    device: str | None,
    show: bool,
    save: bool,
    output_dir: str | None,
    log_level: str | None,
) -> None:
    """Real-Time Object Detection & Tracking Pipeline.

    Combines YOLOv8 detection with SORT/DeepSORT tracking for
    real-time multi-object tracking on video streams or webcam.
    """
    # Load base config
    cfg = load_config(config_path)

    # Apply CLI overrides
    overrides: dict = {}
    if source:
        overrides.setdefault("pipeline", {})["source"] = source
    if tracker:
        overrides.setdefault("tracking", {})["tracker_type"] = tracker
    if backend:
        overrides.setdefault("detection", {})["backend"] = backend
    if model:
        key = "onnx_path" if backend == "onnx" else "model_path"
        overrides.setdefault("detection", {})[key] = model
    if confidence is not None:
        overrides.setdefault("detection", {})["confidence_threshold"] = confidence
    if device:
        overrides.setdefault("detection", {})["device"] = device
    if not show:
        overrides.setdefault("pipeline", {})["show_display"] = False
    if save:
        overrides.setdefault("pipeline", {})["save_video"] = True
    if output_dir:
        overrides.setdefault("pipeline", {})["output_dir"] = output_dir
    if log_level:
        overrides.setdefault("pipeline", {})["log_level"] = log_level

    cfg = merge_configs(cfg, overrides)

    logger = get_logger("cli", level=cfg.get("pipeline", {}).get("log_level", "INFO"))
    logger.info("[bold green]Real-Time Object Detection & Tracking[/]")
    logger.info(f"Source: {cfg['pipeline']['source']}")

    pipeline = HybridPipeline(cfg)
    metrics = pipeline.run()

    logger.info(f"[bold]Final metrics:[/] {metrics}")


if __name__ == "__main__":
    main()
