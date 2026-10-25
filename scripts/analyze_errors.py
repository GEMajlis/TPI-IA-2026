"""Análisis de errores de un run de dev (E1-31): embudo por instancia, tipos de FP, peores clases
y galería de casos.

Uso:
    python scripts/analyze_errors.py --config configs/base.yaml configs/kaggle.yaml --run latest

Necesita la caché de detecciones crudas de la misma config (la deja run_pipeline.py).
Salidas en <run>/error_analysis/: summary.json, funnel.csv, fp_types.csv, per_image.csv,
worst_classes.csv y gallery/<grupo>_<image_id>.png.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.annotations import annotations_path, load_annotations
from eval.error_analysis import (
    draw_case,
    fp_types,
    funnel_rows,
    funnel_summary,
    per_image_scores,
    pick_gallery,
    worst_classes,
)
from eval.lvis_federated import evaluate_from_config, predictions_from_scenes
from scene.config import load_config
from scene.detection.base import load_detections
from scene.io.dataset import list_images, load_image, load_vocabulary
from scene.pipeline import load_scenes, stage_keys


def resolve(cfg: dict, run: str) -> Path:
    if run != "latest":
        return Path(run)
    runs = Path(cfg["paths"]["outputs"]).iterdir()
    return sorted(p for p in runs if (p / "scenes").is_dir())[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--run", default="latest")
    parser.add_argument("--raw-dir", help="caché de detecciones crudas (por defecto, la de cfg)")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    ea = cfg["eval"]["error_analysis"]
    run_dir = resolve(cfg, args.run)
    # La config del run manda (umbral, NMS) para recalcular el embudo igual que la pipeline.
    run_cfg = json.loads((run_dir / cfg["tracing"]["manifest_file"]).read_text("utf-8"))["config"]
    run_cfg["paths"] = cfg["paths"]
    run_cfg["eval"] = cfg["eval"]  # los parámetros de evaluación no son parte de la inferencia
    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir
        else Path(cfg["paths"]["cache"]) / "detections_raw" / stage_keys(run_cfg)["detection"]
    )
    gt = load_annotations(annotations_path(cfg))
    scenes = {i: s for i, s in load_scenes(run_dir).items() if i in gt}
    if not scenes:
        raise SystemExit(f"{run_dir} no tiene escenas de dev")
    ids = sorted(scenes)
    raw = {i: load_detections(raw_dir / f"{i}.json") for i in ids}
    names = {c.id: c.name for c in load_vocabulary(cfg)}

    preds = predictions_from_scenes(scenes)
    box = evaluate_from_config(preds, gt, run_cfg, iou_type="bbox", image_ids=ids)
    segm = evaluate_from_config(preds, gt, run_cfg, iou_type="segm", image_ids=ids)

    funnel = funnel_rows(gt, raw, box, segm, run_cfg)
    fps = fp_types(scenes, gt, box, run_cfg)
    scores = per_image_scores(box)
    worst = worst_classes(box.per_category, names, ea["worst_classes_min_gt"])
    gallery = pick_gallery(scores, fps, ea["gallery_per_group"])

    out = run_dir / "error_analysis"
    out.mkdir(exist_ok=True)
    funnel.assign(category_name=funnel["category_id"].map(names)).to_csv(
        out / "funnel.csv", index=False
    )
    fps.assign(category_name=fps["category_id"].map(names)).to_csv(
        out / "fp_types.csv", index=False
    )
    scores.to_csv(out / "per_image.csv", index=False)
    worst.to_csv(out / "worst_classes.csv", index=False)

    lost = funnel[funnel["lost_at"] != "ok"]
    confusions = Counter(
        f"{names.get(r.category_id)} ← {names.get(int(r.confused_with))}"
        for r in fps[fps["fp_type"] == "class_confusion"].itertuples()
    )
    summary = {
        "run": str(run_dir),
        "n_images": len(ids),
        "funnel": funnel_summary(funnel),
        "fp_types": fps["fp_type"].value_counts().to_dict() if len(fps) else {},
        "fp_types_by_score": (
            fps.assign(bin=(fps["score"] // 0.1 * 0.1).round(1))
            .groupby(["bin", "fp_type"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            if len(fps)
            else {}
        ),
        "top_class_confusions": dict(confusions.most_common(10)),
        "most_lost_classes": lost["category_id"].map(names).value_counts().head(10).to_dict(),
        "worst_classes": worst.head(10).to_dict(orient="records"),
        "gallery": gallery,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")

    recs = {r.image_id: r for r in list_images(cfg, "dev")}
    for group, group_ids in gallery.items():
        for iid in group_ids:
            draw_case(
                load_image(recs[iid]),
                gt[iid],
                scenes[iid],
                box,
                names,
                out / "gallery" / f"{group}_{iid}.png",
            )
    printable = {k: v for k, v in summary.items() if k != "worst_classes"}
    print(json.dumps(printable, indent=2, default=str))
    print(f"salidas en {out}")


if __name__ == "__main__":
    main()
