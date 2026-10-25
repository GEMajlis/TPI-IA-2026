"""Análisis de errores y su propagación por la pipeline (E1-31).

1. **Embudo por instancia GT:** primera etapa en la que se pierde cada objeto anotado. Las etapas
   de detección se recalculan desde la caché de detecciones crudas con el mismo post-proceso que
   la pipeline; en ellas basta con que *alguna* caja de la clase correcta tenga IoU ≥ umbral
   ("cobertura"). Las dos últimas usan el matching uno a uno del evaluador federado.
2. **Tipos de falso positivo** (solo los que el evaluador cuenta como FP).
3. **Galería** de imágenes con mejor y peor F1 y de casos ambiguos (muchas predicciones ignoradas
   por la política federada o confusiones de clase).
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from eval.annotations import ImageGT
from eval.lvis_federated import FederatedResult, size_label
from scene.detection.base import Detection
from scene.detection.postprocess import apply_threshold, box_iou_matrix, per_class_nms, postprocess

# Orden del embudo: la primera condición que falla es la etapa donde se pierde la instancia.
FUNNEL = (
    "no_box",  # ninguna caja cruda (de ninguna clase) la cubre
    "wrong_class",  # solo la cubren cajas de otra clase
    "below_threshold",  # la clase correcta la cubre, pero con score bajo el umbral
    "nms",  # la eliminó el NMS
    "max_detections",  # la cortó el tope de detecciones
    "box_match",  # quedó una caja, pero el matching uno a uno la asignó a otra instancia
    "mask",  # caja TP, pero la máscara no llega al IoU
    "ok",  # TP de máscara
)


def _boxes(dets: list[Detection]) -> np.ndarray:
    return np.array([d.box_xyxy for d in dets], dtype=float).reshape(-1, 4)


def _covered(gt_box: np.ndarray, dets: list[Detection], cid: int | None, thr: float) -> bool:
    cand = [d for d in dets if cid is None or d.category_id == cid]
    return bool(cand) and float(box_iou_matrix(gt_box[None], _boxes(cand)).max()) >= thr


def funnel_rows(
    gt: dict[str, ImageGT],
    raw: dict[str, list[Detection]],
    box_res: FederatedResult,
    segm_res: FederatedResult,
    cfg: dict,
) -> pd.DataFrame:
    """Una fila por instancia GT con la etapa en la que se pierde (o `ok`)."""
    det_cfg = cfg["detection"]
    thr = cfg["eval"]["iou_thr"]
    t = cfg["scene"]["size_thresholds"]
    tp_box = {(r.image_id, r.gt_id) for r in box_res.events.itertuples() if r.kind == "tp"}
    tp_segm = {(r.image_id, r.gt_id) for r in segm_res.events.itertuples() if r.kind == "tp"}
    rows = []
    for iid in box_res.events["image_id"].unique():
        img = gt[iid]
        dets_raw = raw[iid]
        dets_thr = apply_threshold(
            dets_raw, det_cfg["score_threshold"], det_cfg.get("per_class_threshold")
        )
        dets_nms = per_class_nms(dets_thr, det_cfg["nms_iou"])
        dets_final = postprocess(dets_raw, det_cfg).detections
        for inst in img.instances:
            box = np.array(inst.bbox_xyxy, dtype=float)
            cid = inst.category_id
            if (iid, inst.id) in tp_segm:
                stage = "ok"
            elif (iid, inst.id) in tp_box:
                stage = "mask"
            elif not _covered(box, dets_raw, None, thr):
                stage = "no_box"
            elif not _covered(box, dets_raw, cid, thr):
                stage = "wrong_class"
            elif not _covered(box, dets_thr, cid, thr):
                stage = "below_threshold"
            elif not _covered(box, dets_nms, cid, thr):
                stage = "nms"
            elif not _covered(box, dets_final, cid, thr):
                stage = "max_detections"
            else:
                stage = "box_match"
            ratio = inst.area / (img.width * img.height)
            rows.append(
                {
                    "image_id": iid,
                    "gt_id": inst.id,
                    "category_id": cid,
                    "area_ratio": ratio,
                    "size": size_label(ratio, t["small_max"], t["medium_max"]),
                    "lost_at": stage,
                }
            )
    return pd.DataFrame(rows)


def funnel_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Conteos y fracciones por etapa, global y por tamaño, más el embudo acumulado."""

    def counts(df: pd.DataFrame) -> dict[str, Any]:
        c = df["lost_at"].value_counts()
        n = len(df)
        return {
            "n": n,
            **{s: int(c.get(s, 0)) for s in FUNNEL},
            **{f"frac_{s}": float(c.get(s, 0) / n) if n else 0.0 for s in FUNNEL},
        }

    out = {"global": counts(rows)}
    out["by_size"] = {s: counts(rows[rows["size"] == s]) for s in ("small", "medium", "large")}
    # Instancias que siguen "vivas" después de cada etapa (recall acumulado).
    n = len(rows)
    alive, cumulative = n, {}
    for stage in FUNNEL[:-1]:
        alive -= int((rows["lost_at"] == stage).sum())
        cumulative[f"after_{stage}"] = alive / n if n else 0.0
    out["recall_after_stage"] = cumulative
    return out


def fp_types(
    scenes: dict[str, dict], gt: dict[str, ImageGT], res: FederatedResult, cfg: dict
) -> pd.DataFrame:
    """Clasifica cada FP de cajas: duplicado, localización, confusión de clase o fondo."""
    thr = cfg["eval"]["iou_thr"]
    loc_min = cfg["eval"]["error_analysis"]["localization_min_iou"]
    rows = []
    for r in res.events[res.events["kind"] == "fp"].itertuples():
        obj = next(o for o in scenes[r.image_id]["objects"] if o["id"] == r.pred_id)
        img = gt[r.image_id]
        box = np.array(obj["bbox"], dtype=float)[None]
        same = [g for g in img.instances if g.category_id == r.category_id]
        other = [g for g in img.instances if g.category_id != r.category_id]
        iou_same = (
            float(box_iou_matrix(box, np.array([g.bbox_xyxy for g in same])).max()) if same else 0.0
        )
        iou_other, other_cat = 0.0, None
        if other:
            ious = box_iou_matrix(box, np.array([g.bbox_xyxy for g in other]))[0]
            j = int(ious.argmax())
            iou_other, other_cat = float(ious[j]), other[j].category_id
        if r.reason == "unmatched" and iou_same >= thr:
            kind = "duplicate"  # la instancia ya la tomó una predicción de mayor score
        elif r.reason == "unmatched" and iou_same >= loc_min:
            kind = "localization"
        elif iou_other >= thr:
            kind = "class_confusion"
        elif r.reason == "negative_category":
            kind = "negative_category"  # clase verificada como ausente, sin objeto debajo
        else:
            kind = "background"
        rows.append(
            {
                "image_id": r.image_id,
                "pred_id": r.pred_id,
                "category_id": r.category_id,
                "score": r.score,
                "fp_type": kind,
                "iou_same_class": iou_same,
                "iou_other_class": iou_other,
                "confused_with": other_cat if kind == "class_confusion" else None,
            }
        )
    return pd.DataFrame(rows)


def per_image_scores(res: FederatedResult) -> pd.DataFrame:
    """TP/FP/FN/ignorados y F1 por imagen."""
    ev = res.events
    rows = []
    for iid, g in ev.groupby("image_id"):
        k = g["kind"].value_counts()
        tp, fp, fn, ign = (int(k.get(x, 0)) for x in ("tp", "fp", "fn", "ignored"))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        rows.append({"image_id": iid, "tp": tp, "fp": fp, "fn": fn, "ignored": ign, "f1": f1})
    return pd.DataFrame(rows).sort_values("f1")


def pick_gallery(scores: pd.DataFrame, fps: pd.DataFrame, n: int) -> dict[str, list[str]]:
    """Mejores, peores y ambiguas (más confusiones de clase y más predicciones ignoradas)."""
    confusions = Counter(fps.loc[fps["fp_type"] == "class_confusion", "image_id"])
    amb = scores.assign(
        conf=scores["image_id"].map(confusions).fillna(0),
        ign_frac=scores["ignored"]
        / (scores["tp"] + scores["fp"] + scores["ignored"]).clip(lower=1),
    ).sort_values(["conf", "ign_frac"], ascending=False)
    worst = list(scores["image_id"][:n])
    # Los grupos no se superponen (con pocas imágenes, la peor también podría ser la mejor).
    best = [i for i in scores["image_id"][::-1] if i not in worst][:n]
    taken = set(worst) | set(best)
    ambiguous = [i for i in amb["image_id"] if i not in taken][:n]
    return {"correct": best, "failed": worst, "ambiguous": ambiguous}


def draw_case(
    image: np.ndarray,
    img: ImageGT,
    scene: dict,
    box_res: FederatedResult,
    names: dict[int, str],
    path: str | Path,
) -> Path:
    """GT (verde, FN en amarillo) y predicciones (azul TP, rojo FP, gris ignoradas)."""
    ev = box_res.events[box_res.events["image_id"] == img.image_id]
    kind_by_pred = dict(zip(ev["pred_id"], ev["kind"], strict=True))
    fn_ids = set(ev.loc[ev["kind"] == "fn", "gt_id"])
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for inst in img.instances:
        color = (255, 210, 0) if inst.id in fn_ids else (0, 200, 0)
        draw.rectangle(inst.bbox_xyxy, outline=color, width=2)
    colors = {"tp": (40, 110, 255), "fp": (255, 40, 40), "ignored": (150, 150, 150)}
    for obj in scene["objects"]:
        kind = kind_by_pred.get(obj["id"], "ignored")
        x1, y1, *_ = obj["bbox"]
        draw.rectangle(obj["bbox"], outline=colors.get(kind, colors["ignored"]), width=1)
        if kind != "ignored":
            draw.text(
                (x1 + 2, y1 + 1), f"{names.get(obj['category_id'], '?')} {kind}", fill=colors[kind]
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def worst_classes(per_category: pd.DataFrame, names: dict[int, str], min_gt: int) -> pd.DataFrame:
    """Clases con peor F1 entre las que tienen al menos `min_gt` instancias GT."""
    df = copy.deepcopy(per_category)
    df["n_gt"] = df["tp"] + df["fn"]
    df["category_name"] = df["category_id"].map(names)
    df = df[df["n_gt"] >= min_gt].sort_values("f1")
    return df[
        ["category_id", "category_name", "n_gt", "tp", "fp", "fn", "precision", "recall", "f1"]
    ]
