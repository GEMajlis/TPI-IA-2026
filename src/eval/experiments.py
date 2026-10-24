"""Comparación controlada de dos runs de dev (E1-30): métricas en tuning y holdout, costo y
registro en `experiments.csv`."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from eval.annotations import ImageGT
from eval.lvis_federated import evaluate_from_config, predictions_from_scenes
from eval.scene_consistency import internal_consistency
from eval.tuning import split_tuning_holdout

STAGES_COST = ("detection", "segmentation", "scene")


def run_metrics(
    scenes: dict[str, dict], gt: dict[str, ImageGT], cfg: dict, splits: dict[str, list[str]]
) -> dict[str, Any]:
    """F1 de cajas y máscaras por split (con desglose por tamaño en máscaras) y consistencia."""
    out: dict[str, Any] = {}
    for split, ids in splits.items():
        ids = [i for i in ids if i in scenes]
        preds = predictions_from_scenes({i: scenes[i] for i in ids})
        for iou_type, name in (("bbox", "box"), ("segm", "mask")):
            res = evaluate_from_config(preds, gt, cfg, iou_type=iou_type, image_ids=ids)
            out[f"{split}_f1_{name}"] = res.summary["f1"]
            if iou_type == "segm":
                for size, v in res.by_size.items():
                    out[f"{split}_f1_mask_{size}"] = v["f1"]
    cons = internal_consistency(scenes, cfg)
    out["consistency_issues"] = cons["n_issues"]
    out["mask_outside_bbox_gt10pct"] = cons["mask_outside_bbox"]["frac_objects_gt_10pct"]
    out["objects_per_image"] = cons["n_objects"] / max(cons["n_scenes"], 1)
    return out


def stage_cost_s(trace: list[dict[str, Any]]) -> dict[str, float]:
    """Segundos por etapa en un run (solo lo que se ejecutó, no lo que vino de caché)."""
    cost = dict.fromkeys(STAGES_COST, 0.0)
    for r in trace:
        if r["stage"] in cost and r["status"] == "ok":
            cost[r["stage"]] += r["duration_ms"] / 1000
    return cost


def dev_splits(gt: dict[str, ImageGT], cfg: dict, n_holdout: int = 40) -> dict[str, list[str]]:
    """El mismo split tuning/holdout de E1-29 (misma semilla)."""
    tuning, holdout = split_tuning_holdout(gt, sorted(gt), n_holdout, cfg["seed"])
    return {"tuning": tuning, "holdout": holdout}


def append_row(path: str | Path, row: dict[str, Any]) -> Path:
    """Agrega una fila a `experiments.csv` (crea el archivo con encabezado si no existe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path
