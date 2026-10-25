"""Ajuste del post-proceso de detección (E1-29) sobre la caché de detecciones crudas.

El umbral se elige en un split de **tuning** y se reporta en un **holdout** que no participa de
la elección. Ajustar hiperparámetros con GT es legítimo (no asiste a la inferencia de ninguna
imagen); lo que no se puede es reportar el F1 del mismo conjunto con el que se eligió.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from eval.annotations import ImageGT
from eval.lvis_federated import Prediction, evaluate_from_config
from scene.detection.base import Detection
from scene.detection.postprocess import postprocess


def split_tuning_holdout(
    gt: dict[str, ImageGT], image_ids: list[str], n_holdout: int, seed: int
) -> tuple[list[str], list[str]]:
    """Split estratificado por cantidad de instancias GT.

    Se ordenan las imágenes por cantidad de instancias y se cortan en grupos consecutivos de
    `len / n_holdout`; de cada grupo va una imagen al azar (con semilla) al holdout.
    """
    if not 0 < n_holdout < len(image_ids):
        raise ValueError("n_holdout tiene que estar entre 1 y la cantidad de imágenes - 1")
    rng = np.random.default_rng(seed)
    ordered = sorted(image_ids, key=lambda i: (len(gt[i].instances), i))
    groups = np.array_split(np.array(ordered), n_holdout)
    holdout = sorted(str(g[rng.integers(len(g))]) for g in groups)
    tuning = sorted(set(image_ids) - set(holdout))
    return tuning, holdout


def to_predictions(dets: list[Detection]) -> list[Prediction]:
    return [Prediction(d.category_id, d.score, d.box_xyxy) for d in dets]


def evaluate_postprocess(
    raw: dict[str, list[Detection]],
    gt: dict[str, ImageGT],
    image_ids: list[str],
    cfg: dict,
    det_overrides: dict[str, Any],
) -> dict[str, float]:
    """F1 de cajas de aplicar `postprocess` con `cfg['detection']` + overrides a la caché."""
    det_cfg = copy.deepcopy(cfg["detection"])
    det_cfg.update(det_overrides)
    preds = {i: to_predictions(postprocess(raw[i], det_cfg).detections) for i in image_ids}
    res = evaluate_from_config(preds, gt, cfg, iou_type="bbox", image_ids=image_ids)
    s = res.summary
    return {
        "precision": s["precision"],
        "recall": s["recall"],
        "f1": s["f1"],
        "evaluable_coverage": s["evaluable_coverage"],
        "preds_per_image": s["n_pred"] / len(image_ids),
        "f1_small": res.by_size["small"]["f1"],
    }


def sweep(
    raw: dict[str, list[Detection]],
    gt: dict[str, ImageGT],
    splits: dict[str, list[str]],
    cfg: dict,
    thresholds: list[float],
    variants: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Una fila por (variante, umbral, split). `variants` = nombre → overrides de detección."""
    rows = []
    for name, overrides in variants.items():
        for thr in thresholds:
            for split, ids in splits.items():
                m = evaluate_postprocess(
                    raw, gt, ids, cfg, {**overrides, "score_threshold": float(thr)}
                )
                rows.append({"variant": name, "threshold": float(thr), "split": split, **m})
    return pd.DataFrame(rows)


def choose(curve: pd.DataFrame, variant: str, split: str = "tuning") -> pd.Series:
    """Fila de máximo F1 en `split`; ante empates, el umbral más alto (menos predicciones)."""
    sub = curve[(curve["variant"] == variant) & (curve["split"] == split)]
    best = sub["f1"].max()
    return sub[np.isclose(sub["f1"], best)].sort_values("threshold").iloc[-1]
