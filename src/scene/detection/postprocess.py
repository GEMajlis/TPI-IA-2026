"""Post-proceso de detecciones: umbral, NMS por clase, NMS entre clases y tope por imagen.

Cada filtro informa cuántas detecciones elimina, para el embudo de la traza (E1-31).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scene.detection.base import Detection


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU entre dos conjuntos de cajas xyxy (N×4 y M×4) → N×M."""
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """NMS greedy: índices que sobreviven, ordenados por score descendente."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    iou = box_iou_matrix(boxes, boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= iou[i] > iou_thr
    return keep


def _boxes(dets: list[Detection]) -> np.ndarray:
    return np.array([d.box_xyxy for d in dets], dtype=np.float64).reshape(-1, 4)


def _scores(dets: list[Detection]) -> np.ndarray:
    return np.array([d.score for d in dets], dtype=np.float64)


def apply_threshold(
    dets: list[Detection], threshold: float, per_class: dict[str, float] | None = None
) -> list[Detection]:
    """Umbral global; `per_class` (nombre de categoría → umbral) lo reemplaza para esa clase."""
    per_class = per_class or {}
    return [d for d in dets if d.score >= per_class.get(d.category_name, threshold)]


def per_class_nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """NMS dentro de cada categoría. También une las cajas de sinónimos de una misma clase."""
    by_cat: dict[int, list[Detection]] = {}
    for d in dets:
        by_cat.setdefault(d.category_id, []).append(d)
    out: list[Detection] = []
    for group in by_cat.values():
        out.extend(group[i] for i in nms(_boxes(group), _scores(group), iou_thr))
    return sorted(out, key=lambda d: -d.score)


def cross_class_nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """NMS ignorando la clase: ante cajas casi iguales queda la de mayor score."""
    if not dets:
        return []
    return [dets[i] for i in nms(_boxes(dets), _scores(dets), iou_thr)]


@dataclass
class PostprocessResult:
    detections: list[Detection]
    # Cantidad que sobrevive a cada paso, en orden: {"raw": 812, "threshold": 95, ...}.
    counts: dict[str, int] = field(default_factory=dict)


def postprocess(dets: list[Detection], det_cfg: dict[str, Any]) -> PostprocessResult:
    """Aplica los filtros de `cfg['detection']` en orden y registra cuántas quedan tras cada uno."""
    counts = {"raw": len(dets)}
    out = apply_threshold(dets, det_cfg["score_threshold"], det_cfg.get("per_class_threshold"))
    counts["threshold"] = len(out)
    out = per_class_nms(out, det_cfg["nms_iou"])
    counts["nms_per_class"] = len(out)
    cross = det_cfg.get("cross_class_nms") or {}
    if cross.get("enabled"):
        out = cross_class_nms(out, cross["iou"])
        counts["nms_cross_class"] = len(out)
    out = sorted(out, key=lambda d: -d.score)[: det_cfg["max_detections"]]
    counts["max_detections"] = len(out)
    return PostprocessResult(detections=out, counts=counts)
