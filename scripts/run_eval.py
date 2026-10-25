"""Evaluación de un run de `run_pipeline.py`: F1 de cajas y máscaras, consistencia del SceneJSON
y exactitud de posición, tamaño y color contra GT (solo imágenes de dev).

Uso:
    python scripts/run_eval.py --run outputs/runs/<run_id>
    python scripts/run_eval.py --config configs/base.yaml configs/kaggle.yaml --run latest

Escribe en el run: metrics_detection.json, metrics_segmentation.json (+ CSV por categoría) y
metrics_scene.json. La consistencia interna se calcula para todas las escenas (también test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.annotations import annotations_path, load_annotations
from eval.lvis_federated import evaluate_from_config, predictions_from_scenes, save_metrics
from eval.scene_consistency import attribute_accuracy, internal_consistency, save_scene_metrics
from scene.config import load_config
from scene.io.dataset import list_images, load_image, load_vocabulary
from scene.pipeline import load_scenes


def resolve_run(cfg: dict, run: str) -> Path:
    if run != "latest":
        return Path(run)
    runs = sorted(p for p in Path(cfg["paths"]["outputs"]).iterdir() if (p / "scenes").is_dir())
    if not runs:
        raise SystemExit("No hay runs con scenes/: correr antes scripts/run_pipeline.py")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--run", default="latest")
    parser.add_argument("--no-color", action="store_true", help="no medir color (más rápido)")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    run_dir = resolve_run(cfg, args.run)
    scenes = load_scenes(run_dir)
    out: dict = {"run": str(run_dir), "consistency": internal_consistency(scenes, cfg)}

    gt = load_annotations(annotations_path(cfg))
    dev_ids = [i for i in scenes if i in gt]
    if dev_ids:
        names = {c.id: c.name for c in load_vocabulary(cfg)}
        preds = predictions_from_scenes({i: scenes[i] for i in dev_ids})
        for iou_type, name in (("bbox", "detection"), ("segm", "segmentation")):
            res = evaluate_from_config(preds, gt, cfg, iou_type=iou_type, image_ids=dev_ids)
            save_metrics(res, run_dir, name, names)
            out[name] = {
                k: res.summary[k] for k in ("precision", "recall", "f1", "evaluable_coverage")
            }
            out[name]["by_size_f1"] = {s: v["f1"] for s, v in res.by_size.items()}
        images = None
        if not args.no_color:
            recs = {r.image_id: r for r in list_images(cfg, "dev")}
            images = {i: load_image(recs[i]) for i in dev_ids}
        out["attributes"] = attribute_accuracy(scenes, gt, cfg, images)

    save_scene_metrics({k: out[k] for k in ("consistency", "attributes") if k in out}, run_dir)
    printable = {**out, "consistency": {**out["consistency"]}}
    printable["consistency"].pop("scenes_with_issues")
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
