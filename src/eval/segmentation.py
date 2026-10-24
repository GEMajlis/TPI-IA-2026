"""Evaluación de la segmentación con oráculo (E1-12).

Se corre SAM2 con las **cajas GT** como prompt (solo en evaluación, nunca en la inferencia).
Como cada máscara sale de la caja de una instancia conocida, se compara directo con su máscara
GT, sin matching: eso mide el error propio de SAM2 y del post-proceso, separado del detector.
Con una sola pasada de SAM2 se evalúan la máscara cruda y la post-procesada.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eval.annotations import ImageGT
from eval.lvis_federated import FederatedResult, size_label
from scene.detection.base import Detection
from scene.segmentation.postprocess import postprocess_mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def gt_detections(img: ImageGT) -> list[Detection]:
    """Una detección por instancia GT (score 1), con `raw_query` = id de la instancia."""
    return [
        Detection(inst.bbox_xyxy, 1.0, inst.category_id, str(inst.category_id), "gt", str(inst.id))
        for inst in img.instances
        if not inst.iscrowd
    ]


def oracle_rows(
    img: ImageGT, segmenter: Any, image: np.ndarray, pp_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """IoU con GT de cada instancia, para la máscara cruda de SAM2 y la post-procesada.

    `segmenter` debe estar construido con el post-proceso apagado (así devuelve la cruda).
    """
    dets = gt_detections(img)
    by_id = {str(inst.id): inst for inst in img.instances}
    rows = []
    for obj in segmenter.segment(image, dets):
        inst = by_id[obj.detection.raw_query]
        gmask = img.mask(inst)
        post = postprocess_mask(obj.mask, obj.detection.box_xyxy, pp_cfg)
        rows.append(
            {
                "image_id": img.image_id,
                "gt_id": inst.id,
                "category_id": inst.category_id,
                "area_ratio": float(gmask.sum()) / (img.width * img.height),
                "mask_score": obj.mask_score,
                "iou_raw": mask_iou(obj.mask, gmask),
                "iou_post": mask_iou(post, gmask),
            }
        )
    return rows


def summarize_oracle(rows: list[dict[str, Any]], cfg: dict) -> dict[str, Any]:
    """mIoU y fracción con IoU ≥ umbral, global y por tamaño, para cruda y post-procesada."""
    thr = cfg["eval"]["iou_thr"]
    t = cfg["scene"]["size_thresholds"]

    def stats(sub: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(sub)}
        for kind in ("raw", "post"):
            ious = np.array([r[f"iou_{kind}"] for r in sub]) if sub else np.zeros(0)
            out[f"miou_{kind}"] = float(ious.mean()) if len(ious) else None
            out[f"frac_iou_ge_thr_{kind}"] = float((ious >= thr).mean()) if len(ious) else None
        return out

    by_size = {
        s: stats(
            [r for r in rows if size_label(r["area_ratio"], t["small_max"], t["medium_max"]) == s]
        )
        for s in ("small", "medium", "large")
    }
    # Correlación entre el IoU que predice SAM2 y el real: ¿sirve mask_score como confianza?
    if len(rows) > 2:
        corr = float(
            np.corrcoef([r["mask_score"] for r in rows], [r["iou_post"] for r in rows])[0, 1]
        )
    else:
        corr = None
    return {"iou_thr": thr, **stats(rows), "by_size": by_size, "corr_mask_score_iou": corr}


def predicted_pairs_summary(box: FederatedResult, segm: FederatedResult) -> dict[str, Any]:
    """Con cajas predichas: mIoU de máscaras en TP y cuántos TP de cajas se pierden en máscaras."""
    tp_box = box.events[box.events["kind"] == "tp"]
    tp_segm = segm.events[segm.events["kind"] == "tp"]
    box_ids = set(zip(tp_box["image_id"], tp_box["pred_id"], strict=True))
    segm_ids = set(zip(tp_segm["image_id"], tp_segm["pred_id"], strict=True))
    return {
        "f1_box": box.summary["f1"],
        "f1_mask": segm.summary["f1"],
        "tp_box": len(box_ids),
        "tp_mask": len(segm_ids),
        "box_tp_lost_in_mask": len(box_ids - segm_ids),
        "miou_mask_tp": float(tp_segm["iou"].mean()) if len(tp_segm) else None,
    }
