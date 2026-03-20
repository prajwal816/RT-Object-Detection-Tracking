"""Export YOLOv8 PyTorch model to ONNX format."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model_path: str = "models/yolov8n.pt",
    output_path: str = "models/yolov8n.onnx",
    input_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
    simplify: bool = True,
    half: bool = False,
    dynamic: bool = False,
) -> str:
    """Export a YOLOv8 ``.pt`` model to ONNX.

    Args:
        model_path: Path to YOLOv8 PyTorch weights.
        output_path: Destination ONNX file path.
        input_size: Export input resolution ``(width, height)``.
        opset: ONNX opset version.
        simplify: Run ``onnx-simplifier`` after export.
        half: Export in FP16.
        dynamic: Enable dynamic batch size.

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO

    logger.info(f"Exporting [bold]{model_path}[/] → ONNX")
    model = YOLO(model_path)

    out = model.export(
        format="onnx",
        imgsz=input_size[0],
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=dynamic,
    )

    # Ultralytics returns the path; move to desired location if different
    exported_path = Path(out) if out else Path(model_path).with_suffix(".onnx")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if exported_path.resolve() != target.resolve():
        import shutil
        shutil.move(str(exported_path), str(target))

    logger.info(f"ONNX model saved to [bold]{target}[/]")
    return str(target)
