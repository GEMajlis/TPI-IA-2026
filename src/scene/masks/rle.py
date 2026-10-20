"""RLE no comprimido en orden column-major (formato COCO), tal como lo pide el schema.

Formato: {"size": [H, W], "counts": [c0, c1, ...]}. Se recorre la máscara por columnas
(`flatten(order="F")`) y las corridas alternan ceros y unos, empezando siempre por ceros
(la primera corrida puede valer 0 si el primer píxel es 1).
"""

from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> dict:
    """Codifica una máscara binaria HxW en RLE no comprimido column-major."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"Se espera una máscara 2D no vacía, llegó shape={mask.shape}")
    h, w = mask.shape
    flat = mask.astype(bool).ravel(order="F")
    # Posiciones donde cambia el valor → límites de cada corrida.
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0, *counts]
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode(rle: dict) -> np.ndarray:
    """Decodifica un RLE no comprimido a una máscara bool HxW."""
    h, w = (int(v) for v in rle["size"])
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or (counts < 0).any():
        raise ValueError("counts debe ser una lista no vacía de enteros >= 0")
    if counts.sum() != h * w:
        raise ValueError(f"La suma de counts ({counts.sum()}) no coincide con H*W ({h * w})")
    values = np.arange(counts.size) % 2 == 1  # corridas pares = 0, impares = 1
    flat = np.repeat(values, counts)
    return flat.reshape((h, w), order="F")


def area(rle: dict) -> int:
    """Cantidad de píxeles en 1 sin decodificar (suma de las corridas impares)."""
    return int(sum(rle["counts"][1::2]))
