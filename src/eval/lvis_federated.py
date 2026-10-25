"""Evaluación federada estilo LVIS: precisión, recall y F1 con IoU ≥ umbral (cajas o máscaras).

Reglas por (imagen, categoría):
- Categoría positiva (tiene GT): matching greedy por score. Match → TP; GT sin match → FN;
  predicción sin match → FP, salvo que la categoría sea no exhaustiva (→ ignorada).
- Categoría negativa (`neg_category_ids`): toda predicción es FP.
- Categoría no verificada (ni positiva ni negativa): las predicciones se ignoran.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from eval.annotations import GTInstance, ImageGT

IOU_TYPES = ("bbox", "segm")
SIZE_LABELS = ("small", "medium", "large")


@dataclass
class Prediction:
    """Predicción a evaluar. `mask` es RLE no comprimido (como en el SceneJSON) o None."""

    category_id: int
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: dict | None = None
    pred_id: str | None = None  # id del objeto en el SceneJSON (para emparejar atributos)


@dataclass
class MatchEvent:
    """Resultado de una predicción o de un GT: tp | fp | fn | ignored."""

    image_id: str
    category_id: int
    kind: str
    score: float | None
    iou: float | None
    area_ratio: float  # del GT para tp/fn, de la predicción para fp/ignored
    reason: str = ""
    pred_id: str | None = None
    gt_id: int | None = None


def predictions_from_scenes(scenes: dict[str, dict]) -> dict[str, list[Prediction]]:
    """Convierte SceneJSON (por image_id) en predicciones."""
    return {
        image_id: [
            Prediction(
                o["category_id"], o["confidence"], tuple(o["bbox"]), o.get("mask"), o.get("id")
            )
            for o in scene["objects"]
        ]
        for image_id, scene in scenes.items()
    }


def size_label(area_ratio: float, small_max: float, medium_max: float) -> str:
    """small si area_ratio < small_max; medium si < medium_max; si no, large."""
    if area_ratio < small_max:
        return "small"
    return "medium" if area_ratio < medium_max else "large"


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precisión, recall y F1 (0 cuando el denominador es 0)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _xywh(b: tuple[float, float, float, float]) -> list[float]:
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


class _GTCache:
    """RLE comprimido de cada instancia GT (se calcula una sola vez)."""

    def __init__(self) -> None:
        self._rles: dict[int, dict] = {}

    def rle(self, inst: GTInstance, h: int, w: int) -> dict:
        if inst.id not in self._rles:
            rles = mask_utils.frPyObjects(inst.segmentation, h, w)
            self._rles[inst.id] = mask_utils.merge(rles) if isinstance(rles, list) else rles
        return self._rles[inst.id]


def _ious(
    preds: list[Prediction], gts: list[GTInstance], img: ImageGT, iou_type: str, cache: _GTCache
) -> np.ndarray:
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    crowd = [0] * len(gts)
    if iou_type == "bbox":
        dt = [_xywh(p.bbox_xyxy) for p in preds]
        gt = [_xywh(g.bbox_xyxy) for g in gts]
    else:
        if any(p.mask is None for p in preds):
            raise ValueError("iou_type='segm' requiere máscara en todas las predicciones")
        dt = [mask_utils.frPyObjects(p.mask, img.height, img.width) for p in preds]
        gt = [cache.rle(g, img.height, img.width) for g in gts]
    return np.asarray(mask_utils.iou(dt, gt, crowd)).reshape(len(preds), len(gts))


def _pred_area_ratio(p: Prediction, img: ImageGT, iou_type: str) -> float:
    if iou_type == "segm" and p.mask is not None:
        area = sum(p.mask["counts"][1::2])
    else:
        x0, y0, x1, y1 = p.bbox_xyxy
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return area / (img.width * img.height)


def _pred_event(p: Prediction, img: ImageGT, iou_type: str, kind: str, reason: str):
    """Evento de una predicción sin match (fp o ignorada)."""
    ratio = _pred_area_ratio(p, img, iou_type)
    return MatchEvent(
        img.image_id, p.category_id, kind, p.score, None, ratio, reason, pred_id=p.pred_id
    )


def match_image(
    preds: list[Prediction],
    img: ImageGT,
    iou_type: str = "bbox",
    iou_thr: float = 0.5,
    cache: _GTCache | None = None,
) -> list[MatchEvent]:
    """Eventos TP/FP/FN/ignorados de una imagen según las reglas federadas."""
    if iou_type not in IOU_TYPES:
        raise ValueError(f"iou_type inválido {iou_type!r}; opciones: {IOU_TYPES}")
    cache = cache or _GTCache()
    img_area = img.width * img.height
    events: list[MatchEvent] = []
    categories = {p.category_id for p in preds} | img.positive_category_ids
    for cid in sorted(categories):
        cat_preds = sorted(
            (p for p in preds if p.category_id == cid), key=lambda p: p.score, reverse=True
        )
        gts = [g for g in img.instances if g.category_id == cid]

        if not gts:
            if cid in img.neg_category_ids:
                events += [
                    _pred_event(p, img, iou_type, "fp", "negative_category") for p in cat_preds
                ]
            else:
                events += [
                    _pred_event(p, img, iou_type, "ignored", "unverified_category")
                    for p in cat_preds
                ]
            continue

        ious = _ious(cat_preds, gts, img, iou_type, cache)
        matched = np.zeros(len(gts), dtype=bool)
        for i, p in enumerate(cat_preds):
            # GT libre con mayor IoU (si supera el umbral).
            cand = np.where(~matched & (ious[i] >= iou_thr), ious[i], -1.0)
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                matched[j] = True
                ratio = gts[j].area / img_area
                events.append(
                    MatchEvent(
                        img.image_id,
                        cid,
                        "tp",
                        p.score,
                        float(ious[i, j]),
                        ratio,
                        pred_id=p.pred_id,
                        gt_id=gts[j].id,
                    )
                )
            elif cid in img.not_exhaustive_category_ids:
                events.append(_pred_event(p, img, iou_type, "ignored", "not_exhaustive"))
            else:
                events.append(_pred_event(p, img, iou_type, "fp", "unmatched"))
        for j, g in enumerate(gts):
            if not matched[j]:
                events.append(
                    MatchEvent(
                        img.image_id, cid, "fn", None, None, g.area / img_area, "missed", gt_id=g.id
                    )
                )
    return events


@dataclass
class FederatedResult:
    """Métricas globales, desgloses y eventos individuales."""

    iou_type: str
    iou_thr: float
    summary: dict
    by_size: dict[str, dict]
    by_frequency: dict[str, dict]
    per_category: pd.DataFrame
    events: pd.DataFrame


def _counts(df: pd.DataFrame) -> dict:
    kinds = df["kind"].value_counts() if len(df) else pd.Series(dtype=int)
    tp, fp, fn, ign = (int(kinds.get(k, 0)) for k in ("tp", "fp", "fn", "ignored"))
    n_pred = tp + fp + ign
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored": ign,
        "n_pred": n_pred,
        **prf(tp, fp, fn),
        # Fracción de predicciones que entraron en la cuenta de precisión.
        "evaluable_coverage": (tp + fp) / n_pred if n_pred else 0.0,
    }


def category_frequency(gt: dict[str, ImageGT], rare_max: int, common_max: int) -> dict[int, str]:
    """rare / common / frequent según en cuántas imágenes de GT aparece cada categoría."""
    n_images: dict[int, int] = {}
    for img in gt.values():
        for cid in img.positive_category_ids:
            n_images[cid] = n_images.get(cid, 0) + 1
    return {
        cid: "rare" if n <= rare_max else "common" if n <= common_max else "frequent"
        for cid, n in n_images.items()
    }


def evaluate(
    predictions: dict[str, list[Prediction]],
    gt: dict[str, ImageGT],
    image_ids: list[str] | None = None,
    *,
    iou_type: str,
    iou_thr: float,
    size_thresholds: tuple[float, float],
    freq_bins: tuple[int, int],
) -> FederatedResult:
    """Evalúa las imágenes indicadas (por defecto, las que tienen predicciones).

    `size_thresholds` = (small_max, medium_max) en área relativa; `freq_bins` = (rare_max,
    common_max) en cantidad de imágenes. Una imagen sin predicciones cuenta todo su GT como FN.
    """
    image_ids = sorted(predictions) if image_ids is None else list(image_ids)
    unknown = [i for i in image_ids if i not in gt]
    if unknown:
        raise KeyError(f"Imágenes sin GT: {unknown[:5]}")
    cache = _GTCache()
    events: list[MatchEvent] = []
    for image_id in image_ids:
        events += match_image(predictions.get(image_id, []), gt[image_id], iou_type, iou_thr, cache)

    columns = list(MatchEvent.__dataclass_fields__)
    df = pd.DataFrame([asdict(e) for e in events], columns=columns)
    df["size"] = [size_label(a, *size_thresholds) for a in df["area_ratio"]]
    freq = category_frequency(gt, *freq_bins)
    df["frequency"] = [freq.get(c, "unseen") for c in df["category_id"]]

    by_size = {s: _counts(df[df["size"] == s]) for s in SIZE_LABELS}
    by_freq = {f: _counts(df[df["frequency"] == f]) for f in ("rare", "common", "frequent")}
    rows = [{"category_id": int(c), **_counts(g)} for c, g in df.groupby("category_id")]
    per_cat = pd.DataFrame(rows, columns=["category_id", *_counts(df.iloc[:0])])
    summary = {"n_images": len(image_ids), **_counts(df)}
    # F1 macro: promedio sobre categorías con al menos un TP, FP o FN.
    scored = per_cat[(per_cat["tp"] + per_cat["fp"] + per_cat["fn"]) > 0]
    summary["f1_macro"] = float(scored["f1"].mean()) if len(scored) else 0.0
    return FederatedResult(iou_type, iou_thr, summary, by_size, by_freq, per_cat, df)


def evaluate_from_config(
    predictions: dict[str, list[Prediction]],
    gt: dict[str, ImageGT],
    cfg: dict,
    iou_type: str,
    image_ids: list[str] | None = None,
) -> FederatedResult:
    """`evaluate` con los parámetros de la config (`eval.*` y `scene.size_thresholds`)."""
    sizes = cfg["scene"]["size_thresholds"]
    freq = cfg["eval"]["frequency_bins"]
    return evaluate(
        predictions,
        gt,
        image_ids,
        iou_type=iou_type,
        iou_thr=cfg["eval"]["iou_thr"],
        size_thresholds=(sizes["small_max"], sizes["medium_max"]),
        freq_bins=(freq["rare_max"], freq["common_max"]),
    )


def save_metrics(
    result: FederatedResult,
    out_dir: str | Path,
    name: str = "detection",
    category_names: dict[int, str] | None = None,
) -> Path:
    """Escribe `metrics_<name>.json` y `metrics_<name>_per_category.csv`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "iou_type": result.iou_type,
        "iou_thr": result.iou_thr,
        "summary": result.summary,
        "by_size": result.by_size,
        "by_frequency": result.by_frequency,
    }
    path = out_dir / f"metrics_{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    per_cat = result.per_category.copy()
    if category_names:
        per_cat.insert(1, "category_name", per_cat["category_id"].map(category_names))
    per_cat.to_csv(out_dir / f"metrics_{name}_per_category.csv", index=False)
    return path
