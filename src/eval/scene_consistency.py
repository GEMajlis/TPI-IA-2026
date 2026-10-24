"""Consistencia de la representación estructurada (E1-19).

1. **Interna** (sin GT): recalcula desde la máscara decodificada el área, el centroide, la
   posición y el tamaño, y revisa relaciones (ids, enum, inversa presente, coherencia con los
   centroides) y evidencias. Se recalcula con código propio, no con el de `scene.scene`, para que
   el chequeo no herede los mismos errores que el builder.
2. **Contra GT** (solo dev): en los pares predicción–GT emparejados por el evaluador federado,
   exactitud de posición, tamaño y color (el color "real" es el dominante dentro de la máscara GT).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from eval.annotations import ImageGT
from eval.lvis_federated import evaluate_from_config, predictions_from_scenes

GRID = (
    ("top_left", "top_center", "top_right"),
    ("middle_left", "center", "middle_right"),
    ("bottom_left", "bottom_center", "bottom_right"),
)
INVERSE = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above"}
SCHEMA_PREDICATES = {"left_of", "right_of", "above", "below", "overlaps"}


def decode_rle(rle: dict) -> np.ndarray:
    """RLE no comprimido column-major → bool HxW (vía pycocotools, independiente de scene)."""
    h, w = rle["size"]
    return mask_utils.decode(mask_utils.frPyObjects(rle, h, w)).astype(bool)


def grid_cell(cx: float, cy: float, width: int, height: int) -> str:
    col = min(int(3 * cx / width), 2) if cx >= 0 else 0
    row = min(int(3 * cy / height), 2) if cy >= 0 else 0
    return GRID[row][col]


def size_class(area_ratio: float, small_max: float, medium_max: float) -> str:
    return "small" if area_ratio < small_max else "medium" if area_ratio < medium_max else "large"


def object_issues(
    obj: dict, width: int, height: int, sizes: tuple[float, float], tol_px: float
) -> tuple[list[str], dict[str, float]]:
    """Inconsistencias de un objeto y medidas auxiliares (por ejemplo, máscara fuera de la caja)."""
    issues: list[str] = []
    mask = decode_rle(obj["mask"])
    if list(mask.shape) != [height, width]:
        return ["mask_size"], {}
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return ["mask_empty"], {}
    area_ratio = len(xs) / (width * height)
    cx, cy = float(xs.mean()), float(ys.mean())
    if not np.isclose(obj["area_ratio"], area_ratio, rtol=1e-6, atol=1e-9):
        issues.append("area_ratio")
    if abs(obj["centroid"][0] - cx) > tol_px or abs(obj["centroid"][1] - cy) > tol_px:
        issues.append("centroid")
    if obj["position"] != grid_cell(cx, cy, width, height):
        issues.append("position")
    if obj["size"] != size_class(area_ratio, *sizes):
        issues.append("size")
    x1, y1, x2, y2 = obj["bbox"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        issues.append("bbox_out_of_image")
    inside = mask[
        max(0, int(np.floor(y1))) : int(np.ceil(y2)), max(0, int(np.floor(x1))) : int(np.ceil(x2))
    ].sum()
    return issues, {"mask_outside_bbox": 1 - inside / len(xs)}


def relation_issues(scene: dict) -> list[str]:
    """Ids inexistentes, predicados fuera del enum, inversas faltantes y dirección incoherente."""
    issues: list[str] = []
    objs = {o["id"]: o for o in scene["objects"]}
    triples = {(r["subject"], r["predicate"], r["object"]) for r in scene["relations"]}
    for s, p, o in triples:
        if s not in objs or o not in objs:
            issues.append("relation_unknown_id")
            continue
        if s == o:
            issues.append("relation_self")
        if p not in SCHEMA_PREDICATES:
            issues.append("relation_predicate")
            continue
        inverse = INVERSE.get(p, p)  # overlaps es simétrica
        if (o, inverse, s) not in triples:
            issues.append("relation_missing_inverse")
        (sx, sy), (ox, oy) = objs[s]["centroid"], objs[o]["centroid"]
        wrong = {"left_of": sx >= ox, "right_of": sx <= ox, "above": sy >= oy, "below": sy <= oy}
        if wrong.get(p, False):
            issues.append("relation_direction")
    for field in ("questions", "questions_llm"):
        for qa in scene.get(field) or []:
            if any(ev not in objs for ev in qa.get("evidence", [])):
                issues.append("evidence_unknown_id")
    return issues


def internal_consistency(scenes: dict[str, dict], cfg: dict) -> dict[str, Any]:
    """Resumen de inconsistencias internas sobre todas las escenas."""
    t = cfg["scene"]["size_thresholds"]
    sizes = (t["small_max"], t["medium_max"])
    tol = 10 ** -cfg["eval"]["centroid_decimals"]
    counts: Counter[str] = Counter()
    outside: list[float] = []
    scenes_with_issues: list[str] = []
    n_objects = 0
    for iid, scene in scenes.items():
        w, h = scene["image"]["width"], scene["image"]["height"]
        before = sum(counts.values())
        for obj in scene["objects"]:
            n_objects += 1
            issues, extra = object_issues(obj, w, h, sizes, tol)
            counts.update(issues)
            if "mask_outside_bbox" in extra:
                outside.append(extra["mask_outside_bbox"])
        counts.update(relation_issues(scene))
        if sum(counts.values()) > before:
            scenes_with_issues.append(iid)
    out = np.array(outside) if outside else np.zeros(1)
    return {
        "n_scenes": len(scenes),
        "n_objects": n_objects,
        "n_relations": sum(len(s["relations"]) for s in scenes.values()),
        "n_issues": int(sum(counts.values())),
        "issues": dict(counts),
        "scenes_with_issues": scenes_with_issues,
        # Fracción de la máscara que cae fuera de la caja entregada (no es error del schema,
        # pero la caja y la máscara deberían describir el mismo objeto).
        "mask_outside_bbox": {
            "mean": float(out.mean()),
            "p95": float(np.percentile(out, 95)),
            "frac_objects_gt_10pct": float((out > 0.10).mean()),
        },
    }


def attribute_accuracy(
    scenes: dict[str, dict],
    gt: dict[str, ImageGT],
    cfg: dict,
    images: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Exactitud de posición, tamaño (y color, si se pasan las imágenes) en pares TP de máscaras."""
    from scene.scene.color import dominant_color_from_config

    ids = [i for i in scenes if i in gt]
    preds = predictions_from_scenes({i: scenes[i] for i in ids})
    result = evaluate_from_config(preds, gt, cfg, iou_type="segm", image_ids=ids)
    t = cfg["scene"]["size_thresholds"]
    sizes = (t["small_max"], t["medium_max"])
    hits: Counter[str] = Counter()
    confusion: dict[str, Counter] = {"size": Counter(), "color": Counter()}
    n = n_color = 0
    tp = result.events[result.events["kind"] == "tp"]
    for row in tp.itertuples():
        img = gt[row.image_id]
        obj = next(o for o in scenes[row.image_id]["objects"] if o["id"] == row.pred_id)
        inst = next(g for g in img.instances if g.id == row.gt_id)
        gmask = img.mask(inst)
        ys, xs = np.nonzero(gmask)
        if len(xs) == 0:
            continue
        n += 1
        gt_pos = grid_cell(float(xs.mean()), float(ys.mean()), img.width, img.height)
        gt_size = size_class(len(xs) / (img.width * img.height), *sizes)
        hits["position"] += obj["position"] == gt_pos
        hits["size"] += obj["size"] == gt_size
        confusion["size"][f"{gt_size}->{obj['size']}"] += 1
        if images is not None and row.image_id in images:
            ref = dominant_color_from_config(images[row.image_id], gmask, cfg)
            pred = obj.get("attributes", {}).get("color")
            if ref is not None and pred is not None:
                n_color += 1
                hits["color"] += pred == ref.name
                confusion["color"][f"{ref.name}->{pred}"] += 1
    return {
        "n_pairs": n,
        "position_acc": hits["position"] / n if n else None,
        "size_acc": hits["size"] / n if n else None,
        "n_color_pairs": n_color,
        "color_acc": hits["color"] / n_color if n_color else None,
        "size_confusion": dict(confusion["size"]),
        "color_confusion": dict(confusion["color"].most_common(20)),
    }


def save_scene_metrics(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / "metrics_scene.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
