"""Relaciones espaciales entre objetos, derivadas de centroides, cajas y máscaras.

- `relations` (enum del schema): `left_of`, `right_of`, `above`, `below`, `overlaps`.
- `extra_relations` (D-04, fuera del enum): `near` y `on_top_of`.

Las relaciones direccionales se emiten en los dos sentidos (A left_of B ⇔ B right_of A) y
`overlaps`/`near` son simétricas. Las operaciones con máscaras solo se hacen sobre pares cuyas
cajas se tocan, y se recortan a la unión de las cajas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RelObject:
    id: str
    centroid: tuple[float, float]  # [x, y]
    box: tuple[int, int, int, int]  # caja de la máscara, xyxy exclusivo
    mask: np.ndarray = field(repr=False, compare=False)
    confidence: float = 1.0


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RelationsResult:
    relations: list[Relation]
    extra_relations: list[Relation]
    stats: dict[str, Any]


def directional(
    a: RelObject, b: RelObject, width: int, height: int, h_margin: float, v_margin: float
) -> list[str]:
    """Predicados direccionales de `a` respecto de `b` por centroides (con margen mínimo)."""
    preds = []
    dx = b.centroid[0] - a.centroid[0]
    dy = b.centroid[1] - a.centroid[1]
    if dx > h_margin * width:
        preds.append("left_of")
    elif dx < -h_margin * width:
        preds.append("right_of")
    if dy > v_margin * height:
        preds.append("above")
    elif dy < -v_margin * height:
        preds.append("below")
    return preds


def _inverse(pred: str) -> str:
    return {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above"}[pred]


def box_gap(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Distancia euclídea entre dos cajas (0 si se tocan o se solapan)."""
    gx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(gx, gy))


def _union_crop(a: RelObject, b: RelObject, pad: int) -> tuple[slice, slice]:
    h, w = a.mask.shape
    x1, y1 = max(0, min(a.box[0], b.box[0]) - pad), max(0, min(a.box[1], b.box[1]) - pad)
    x2, y2 = min(w, max(a.box[2], b.box[2]) + pad), min(h, max(a.box[3], b.box[3]) + pad)
    return slice(y1, y2), slice(x1, x2)


def mask_iou(a: RelObject, b: RelObject) -> float:
    rows, cols = _union_crop(a, b, 0)
    ma, mb = a.mask[rows, cols], b.mask[rows, cols]
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def masks_touch(a: RelObject, b: RelObject, contact_px: int) -> bool:
    """True si la máscara de `a` dilatada `contact_px` toca la de `b`."""
    rows, cols = _union_crop(a, b, contact_px)
    ma = a.mask[rows, cols].astype(np.uint8)
    kernel = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)
    return bool(np.logical_and(cv2.dilate(ma, kernel) > 0, b.mask[rows, cols]).any())


def on_top_of(a: RelObject, b: RelObject, min_x_overlap: float, contact_px: int) -> bool:
    """`a` apoyado sobre `b`: más arriba, con solape horizontal y contacto entre máscaras."""
    if not (a.centroid[1] < b.centroid[1] and a.box[3] <= b.box[3]):
        return False
    x_overlap = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
    if x_overlap < min_x_overlap * max(a.box[2] - a.box[0], 1):
        return False
    return masks_touch(a, b, contact_px)


def _pair_relations(
    a: RelObject, b: RelObject, width: int, height: int, r: dict[str, Any], stats: dict
) -> tuple[list[Relation], list[Relation]]:
    """Todas las relaciones del par no ordenado {a, b}, en los dos sentidos."""
    rels = []
    for p in directional(a, b, width, height, r["h_margin"], r["v_margin"]):
        rels += [Relation(a.id, p, b.id), Relation(b.id, _inverse(p), a.id)]
    extra = []
    gap = box_gap(a.box, b.box)
    if gap <= r["contact_px"]:
        stats["n_mask_ops"] += 1
        if gap == 0 and mask_iou(a, b) >= r["overlap_min_iou"]:
            rels += [Relation(a.id, "overlaps", b.id), Relation(b.id, "overlaps", a.id)]
        for s, o in ((a, b), (b, a)):
            if on_top_of(s, o, r["on_top_min_x_overlap"], r["contact_px"]):
                extra.append(Relation(s.id, "on_top_of", o.id))
    if gap <= r["near_max_gap"] * np.hypot(width, height):
        extra += [Relation(a.id, "near", b.id), Relation(b.id, "near", a.id)]
    return rels, extra


def compute_relations(
    objs: list[RelObject], width: int, height: int, rel_cfg: dict[str, Any]
) -> RelationsResult:
    """Relaciones de todos los pares, priorizando los de mayor confianza hasta `max_relations`.

    Un par entra completo o no entra, así nunca queda una relación sin su inversa.
    """
    pairs = sorted(
        combinations(range(len(objs)), 2),
        key=lambda ij: (-min(objs[ij[0]].confidence, objs[ij[1]].confidence), ij),
    )
    stats: dict[str, Any] = {"n_objects": len(objs), "n_pairs": len(pairs), "n_mask_ops": 0}
    relations: list[Relation] = []
    extra: list[Relation] = []
    truncated = 0
    for i, j in pairs:
        rels, ext = _pair_relations(objs[i], objs[j], width, height, rel_cfg, stats)
        if len(relations) + len(rels) > rel_cfg["max_relations"]:
            truncated += 1
            continue
        relations += rels
        extra += ext
    stats["pairs_truncated"] = truncated
    stats["n_relations"] = len(relations)
    stats["n_extra_relations"] = len(extra)
    return RelationsResult(relations, extra, stats)
