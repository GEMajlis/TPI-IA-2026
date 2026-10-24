"""Post-proceso de máscaras: recorte a la caja con margen, componente principal y relleno de huecos.

Cada paso puede apagarse desde `cfg['segmentation']['postprocess']` para medir su aporte (E1-12).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def crop_to_box(
    mask: np.ndarray, box_xyxy: tuple[float, float, float, float], margin: float
) -> np.ndarray:
    """Anula lo que cae fuera de la caja agrandada `margin` (fracción del ancho y alto)."""
    h, w = mask.shape
    x1, y1, x2, y2 = box_xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    c1, r1 = max(0, int(np.floor(x1 - mx))), max(0, int(np.floor(y1 - my)))
    c2, r2 = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
    out = np.zeros_like(mask)
    out[r1:r2, c1:c2] = mask[r1:r2, c1:c2]
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Se queda con la componente conexa (8-vecinos) de mayor área."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:  # fondo + a lo sumo una componente
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def fill_holes(mask: np.ndarray, max_hole_ratio: float) -> np.ndarray:
    """Rellena huecos (fondo que no toca el borde) de área ≤ `max_hole_ratio` × área de máscara."""
    area = int(mask.sum())
    if area == 0 or max_hole_ratio <= 0:
        return mask
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, hole_area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and hole_area <= max_hole_ratio * area:
            out[labels == i] = True
    return out


def postprocess_mask(
    mask: np.ndarray, box_xyxy: tuple[float, float, float, float], pp_cfg: dict[str, Any]
) -> np.ndarray:
    """Aplica los pasos activos en orden: recorte → componente principal → huecos.

    Con `apply_below_area_ratio` (E1-30) solo se post-procesan las máscaras crudas cuya área
    relativa es menor que ese valor; las demás se devuelven tal cual las da SAM2.
    """
    out = mask.astype(bool)
    limit = pp_cfg.get("apply_below_area_ratio")
    if limit is not None and out.mean() >= limit:
        return out
    if pp_cfg.get("crop_to_box", True):
        out = crop_to_box(out, box_xyxy, pp_cfg["box_margin"])
    if pp_cfg.get("largest_component", True):
        out = largest_component(out)
    if pp_cfg.get("fill_holes", True):
        out = fill_holes(out, pp_cfg["max_hole_ratio"])
    return out
