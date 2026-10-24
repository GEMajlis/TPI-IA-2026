"""Propiedades geométricas derivadas de la máscara: área, área relativa, centroide, posición 3×3,
tamaño y caja de la máscara. Convenciones en `docs/schema_notes.md` §6.

Se calculan siempre sobre la máscara que va al SceneJSON (la decodificada del RLE), así lo que se
entrega es consistente por construcción. Si la máscara está vacía se usa la caja rasterizada
como máscara y se marca `source="box_fallback"`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from scene.masks import rle
from scene.scene.schema import POSITIONS


@dataclass(frozen=True)
class GeometricProperties:
    area: int  # píxeles de la máscara
    area_ratio: float  # área / (H·W), en (0, 1]
    centroid: tuple[float, float]  # [x, y] en píxeles: media de los índices de la máscara
    position: str  # celda de la grilla 3×3 que contiene el centroide
    size: str  # small | medium | large según area_ratio
    mask_bbox: tuple[int, int, int, int]  # xyxy con borde exclusivo
    source: str  # mask | box_fallback

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid"] = list(self.centroid)
        d["mask_bbox"] = list(self.mask_bbox)
        return d


def position_3x3(cx: float, cy: float, width: int, height: int) -> str:
    """Celda por tercios iguales; un punto sobre un borde va a la celda de la derecha/abajo."""
    col = min(max(int(cx * 3 // width), 0), 2)
    row = min(max(int(cy * 3 // height), 0), 2)
    return POSITIONS[row * 3 + col]


def size_label(area_ratio: float, small_max: float, medium_max: float) -> str:
    if area_ratio < small_max:
        return "small"
    if area_ratio < medium_max:
        return "medium"
    return "large"


def box_to_mask(box_xyxy: tuple[float, float, float, float], height: int, width: int) -> np.ndarray:
    """Rasteriza una caja (al menos 1 píxel, dentro de la imagen)."""
    x1, y1, x2, y2 = box_xyxy
    c1 = min(max(int(np.floor(x1)), 0), width - 1)
    r1 = min(max(int(np.floor(y1)), 0), height - 1)
    c2 = min(max(int(np.ceil(x2)), c1 + 1), width)
    r2 = min(max(int(np.ceil(y2)), r1 + 1), height)
    mask = np.zeros((height, width), dtype=bool)
    mask[r1:r2, c1:c2] = True
    return mask


def ensure_nonempty(
    mask: np.ndarray, box_xyxy: tuple[float, float, float, float]
) -> tuple[np.ndarray, str]:
    """Devuelve la máscara y su origen; si está vacía, la reemplaza por la caja rasterizada."""
    if mask.any():
        return mask, "mask"
    return box_to_mask(box_xyxy, *mask.shape), "box_fallback"


def properties_from_mask(
    mask: np.ndarray, small_max: float, medium_max: float, source: str = "mask"
) -> GeometricProperties:
    """Propiedades de una máscara no vacía (bool HxW)."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Máscara vacía: usar ensure_nonempty antes")
    area = int(len(xs))
    area_ratio = area / (height * width)
    cx, cy = float(xs.mean()), float(ys.mean())
    return GeometricProperties(
        area=area,
        area_ratio=area_ratio,
        centroid=(cx, cy),
        position=position_3x3(cx, cy, width, height),
        size=size_label(area_ratio, small_max, medium_max),
        mask_bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        source=source,
    )


def properties_from_rle(
    mask_rle: dict, small_max: float, medium_max: float, source: str = "mask"
) -> GeometricProperties:
    """Igual que `properties_from_mask`, decodificando antes el RLE del SceneJSON."""
    return properties_from_mask(rle.decode(mask_rle), small_max, medium_max, source)


def properties_from_config(
    mask: np.ndarray, cfg: dict, source: str = "mask"
) -> GeometricProperties:
    t = cfg["scene"]["size_thresholds"]
    return properties_from_mask(mask, t["small_max"], t["medium_max"], source)
