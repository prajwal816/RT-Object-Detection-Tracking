"""Lightweight Re-ID CNN embedder for DeepSORT appearance matching.

Uses a small MobileNetV2-based feature extractor to produce 128-D
appearance embeddings from detection crops, replacing the histogram
fallback for more robust re-identification across occlusions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.timer import LatencyTracker

logger = get_logger(__name__)


class ReIDEmbedder:
    """MobileNetV2-based appearance embedder for detection crops.

    Extracts a compact 128-D embedding vector for each detected bounding box.
    Uses PyTorch for inference with optional GPU support.

    Args:
        model_path: Path to saved re-ID model weights. If ``None``, builds
            a fresh (untrained) lightweight feature extractor.
        input_size: Crop resize dimensions ``(width, height)``.
        device: Inference device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (64, 128),
        device: str = "cpu",
    ) -> None:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        self.device = torch.device(device)
        self.input_size = input_size
        self._latency = LatencyTracker("reid")

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size[::-1]),  # (H, W)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Build model — MobileNetV2 backbone + embedding head
        backbone = models.mobilenet_v2(weights=None)
        # Remove the classifier head
        features = backbone.features
        self.model = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
        )

        # Load pre-trained weights if available
        if model_path:
            try:
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info(f"Re-ID model loaded from [bold]{model_path}[/]")
            except Exception as e:
                logger.warning(f"Could not load Re-ID weights: {e}. Using random init.")

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"ReIDEmbedder ready on {device} (input: {input_size})")

    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute 128-D appearance embeddings for each bounding box.

        Args:
            frame: BGR image (H, W, 3).
            boxes: (N, 4) array in ``[x1, y1, x2, y2]`` format.

        Returns:
            (N, 128) float32 L2-normalized embedding vectors.
        """
        import torch

        if len(boxes) == 0:
            return np.empty((0, 128), dtype=np.float32)

        h, w = frame.shape[:2]
        crops = []

        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                # Create a blank crop if the box is degenerate
                crop = np.zeros(
                    (self.input_size[1], self.input_size[0], 3), dtype=np.uint8
                )

            # BGR → RGB for torchvision
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb)
            crops.append(tensor)

        batch = torch.stack(crops).to(self.device)

        with self._latency:
            with torch.no_grad():
                embeddings = self.model(batch)

        # L2 normalize
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        return embeddings

    @property
    def latency_ms(self) -> float:
        return self._latency.last_ms
