"""Imágenes de debug con las máscaras superpuestas."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scene.detection.debug import draw_detections
from scene.segmentation.sam2 import SegmentedObject


def draw_masks(
    image: np.ndarray, objs: list[SegmentedObject], path: str | Path, alpha: float = 0.45
) -> Path:
    """Pinta cada máscara con un color fijo por índice y dibuja su caja encima."""
    out = image.astype(np.float32).copy()
    rng = np.random.default_rng(0)
    colors = rng.integers(40, 255, size=(max(len(objs), 1), 3))
    for o, color in zip(objs, colors, strict=False):
        out[o.mask] = (1 - alpha) * out[o.mask] + alpha * color
    blended = out.clip(0, 255).astype(np.uint8)
    return draw_detections(blended, [o.detection for o in objs], Path(path))
