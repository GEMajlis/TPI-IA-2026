"""Tablas del EDA calculadas desde el GT (solo análisis, nunca inferencia)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from eval.annotations import ImageGT
from scene.scene.schema import POSITIONS


def grid_position(cx: float, cy: float, width: int, height: int) -> str:
    """Celda 3×3 (nombre oficial del schema) en la que cae un punto."""
    col = min(int(3 * cx / width), 2)
    row = min(int(3 * cy / height), 2)
    return POSITIONS[3 * row + col]


def instances_table(gt: dict[str, ImageGT]) -> pd.DataFrame:
    """Una fila por instancia: área relativa, caja, centroide de la máscara y celda 3×3."""
    rows = []
    for img in gt.values():
        img_area = img.width * img.height
        for inst in img.instances:
            mask = img.mask(inst)
            ys, xs = np.nonzero(mask)
            cx, cy = (xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan)
            x0, y0, x1, y1 = inst.bbox_xyxy
            rows.append(
                {
                    "image_id": img.image_id,
                    "instance_id": inst.id,
                    "category_id": inst.category_id,
                    "area": inst.area,
                    "area_ratio": inst.area / img_area,
                    "mask_area_ratio": mask.sum() / img_area,
                    "bbox_w": x1 - x0,
                    "bbox_h": y1 - y0,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "position": grid_position(cx, cy, img.width, img.height) if len(xs) else None,
                }
            )
    return pd.DataFrame(rows)


def images_table(gt: dict[str, ImageGT], n_categories: int) -> pd.DataFrame:
    """Una fila por imagen: instancias y estado federado de las categorías."""
    rows = []
    for img in gt.values():
        pos = img.positive_category_ids
        rows.append(
            {
                "image_id": img.image_id,
                "width": img.width,
                "height": img.height,
                "n_instances": len(img.instances),
                "n_positive": len(pos),
                "n_negative": len(img.neg_category_ids),
                "n_not_exhaustive": len(img.not_exhaustive_category_ids),
                # Ni positivas ni negativas: sus predicciones se ignoran en la evaluación.
                "n_unverified": n_categories - len(pos | img.neg_category_ids),
            }
        )
    return pd.DataFrame(rows)


def categories_table(gt: dict[str, ImageGT], names: dict[int, str]) -> pd.DataFrame:
    """Una fila por categoría: instancias e imágenes en las que es positiva o negativa."""
    rows = {cid: {"category_id": cid, "category_name": n} for cid, n in names.items()}
    for row in rows.values():
        row.update(n_instances=0, n_images_positive=0, n_images_negative=0, n_images_not_exh=0)
    for img in gt.values():
        for inst in img.instances:
            rows[inst.category_id]["n_instances"] += 1
        for cid in img.positive_category_ids:
            rows[cid]["n_images_positive"] += 1
        for cid in img.neg_category_ids:
            rows[cid]["n_images_negative"] += 1
        for cid in img.not_exhaustive_category_ids:
            rows[cid]["n_images_not_exh"] += 1
    return pd.DataFrame(rows.values()).sort_values("n_instances", ascending=False)


def overlap_table(gt: dict[str, ImageGT], iou_thr: float) -> pd.DataFrame:
    """Por instancia: cuánto la tapa otra instancia y si hay otra de su clase con IoU de caja alto.

    `max_covered` = máxima fracción de su máscara compartida con otra instancia (oclusión).
    `same_class_box_iou` = máximo IoU de caja con otra instancia de la misma categoría.
    """
    rows = []
    for img in gt.values():
        insts = img.instances
        n = len(insts)
        if n == 0:
            continue
        rles = []
        for inst in insts:
            r = mask_utils.frPyObjects(inst.segmentation, img.height, img.width)
            rles.append(mask_utils.merge(r) if isinstance(r, list) else r)
        # Con iscrowd=1, pycocotools devuelve intersección / área de la predicción (fila).
        covered = np.asarray(mask_utils.iou(rles, rles, [1] * n)).reshape(n, n)
        boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in (i.bbox_xyxy for i in insts)]
        box_iou = np.asarray(mask_utils.iou(boxes, boxes, [0] * n)).reshape(n, n)
        np.fill_diagonal(covered, 0)
        np.fill_diagonal(box_iou, 0)
        cats = np.array([i.category_id for i in insts])
        for k, inst in enumerate(insts):
            same = cats == cats[k]
            rows.append(
                {
                    "image_id": img.image_id,
                    "instance_id": inst.id,
                    "category_id": inst.category_id,
                    "max_covered": float(covered[k].max()) if n > 1 else 0.0,
                    "same_class_box_iou": float(box_iou[k][same].max()) if same.sum() > 1 else 0.0,
                    "same_class_dup": bool(same.sum() > 1 and box_iou[k][same].max() >= iou_thr),
                }
            )
    return pd.DataFrame(rows)


def benchmark_table(questions: list[dict], gt: dict[str, ImageGT], name_to_id: dict[str, int]):
    """Por pregunta: estado federado y cantidad de instancias GT de cada clase mencionada."""
    rows = []
    for q in questions:
        img = gt[q["image_file"].rsplit(".", 1)[0]]
        row = {"question_id": q["question_id"], "image_id": img.image_id, "type": q["type"]}
        for k, name in enumerate(q["class_en"]):
            cid = name_to_id[name]
            if cid in img.positive_category_ids:
                status = "positive"
            elif cid in img.neg_category_ids:
                status = "negative"
            else:
                status = "unverified"
            row[f"class_{k}"] = name
            row[f"status_{k}"] = status
            row[f"n_gt_{k}"] = sum(i.category_id == cid for i in img.instances)
        rows.append(row)
    return pd.DataFrame(rows)
