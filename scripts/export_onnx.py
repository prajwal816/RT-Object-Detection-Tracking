#!/usr/bin/env python3
"""Export YOLOv8 model to ONNX format.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/yolov8s.pt --output models/yolov8s.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from src.detection.exporter import export_to_onnx
from src.utils.logger import get_logger

logger = get_logger("export")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", "-m", default="models/yolov8n.pt", help="YOLOv8 .pt model path.")
@click.option("--output", "-o", default="models/yolov8n.onnx", help="Output ONNX path.")
@click.option("--imgsz", default=640, type=int, help="Export image size.")
@click.option("--opset", default=17, type=int, help="ONNX opset version.")
@click.option("--simplify/--no-simplify", default=True, help="Run onnx-simplifier.")
@click.option("--half", is_flag=True, default=False, help="Export in FP16.")
@click.option("--dynamic", is_flag=True, default=False, help="Dynamic batch size.")
def main(
    model: str,
    output: str,
    imgsz: int,
    opset: int,
    simplify: bool,
    half: bool,
    dynamic: bool,
) -> None:
    """Export YOLOv8 PyTorch model to ONNX format."""
    logger.info("[bold green]YOLOv8 → ONNX Export[/]")
    result = export_to_onnx(
        model_path=model,
        output_path=output,
        input_size=(imgsz, imgsz),
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=dynamic,
    )
    logger.info(f"Done: {result}")


if __name__ == "__main__":
    main()
