"""Imágenes de debug con las cajas dibujadas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scene.detection.base import Detection


def draw_detections(
    image: np.ndarray, dets: list[Detection], path: str | Path, max_boxes: int = 50
) -> Path:
    """Dibuja las `max_boxes` cajas de mayor score con su clase y score, y guarda un PNG."""
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in sorted(dets, key=lambda d: -d.score)[:max_boxes]:
        x1, y1, x2, y2 = d.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)
        label = f"{d.category_name} {d.score:.2f}"
        tx, ty = x1 + 2, max(0, y1 - 12)
        draw.rectangle(draw.textbbox((tx, ty), label), fill=(255, 40, 40))
        draw.text((tx, ty), label, fill=(255, 255, 255))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
