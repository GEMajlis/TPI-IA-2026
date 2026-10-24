"""Segmentación con oráculo (E1-12): SAM2 con cajas GT vs. con cajas predichas.

Uso:
    python scripts/eval_segmentation.py --config configs/base.yaml configs/kaggle.yaml --run latest
    # smoke test en CPU
    python scripts/eval_segmentation.py --limit 2 \\
        --set segmentation.model_id=facebook/sam2.1-hiera-tiny

Con `--run` (un run de run_pipeline.py sobre dev) agrega la comparación con las cajas predichas.
Escribe metrics_segmentation_oracle.json y segmentation_oracle.csv en el run (o en outputs/eval/).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from eval.annotations import annotations_path, load_annotations
from eval.lvis_federated import evaluate_from_config, predictions_from_scenes
from eval.segmentation import oracle_rows, predicted_pairs_summary, summarize_oracle
from scene.config import load_config, resolve_device, set_seed
from scene.io.dataset import list_images, load_image
from scene.pipeline import load_scenes
from scene.segmentation.sam2 import build_segmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--set", nargs="*", default=[], help="overrides clave.sub=valor")
    parser.add_argument("--run", help="run de dev con scenes/ ('latest' = el más reciente)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(*args.config, dotlist=args.set)
    set_seed(cfg["seed"])
    gt = load_annotations(annotations_path(cfg))
    run_dir = None
    if args.run:
        if args.run == "latest":
            runs = Path(cfg["paths"]["outputs"]).iterdir()
            run_dir = sorted(p for p in runs if (p / "scenes").is_dir())[-1]
        else:
            run_dir = Path(args.run)

    # SAM2 sin post-proceso: la cruda se post-procesa aparte para evaluar las dos.
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg["segmentation"]["postprocess"] = {
        "crop_to_box": False,
        "largest_component": False,
        "fill_holes": False,
        "box_margin": 0.0,
        "max_hole_ratio": 0.0,
    }
    segmenter = build_segmenter(raw_cfg, resolve_device(cfg.get("device", "auto")))
    pp_cfg = cfg["segmentation"]["postprocess"]

    recs = [r for r in list_images(cfg, "dev") if r.image_id in gt][: args.limit]
    small_max = cfg["scene"]["size_thresholds"]["small_max"]
    rows = []
    for k, rec in enumerate(recs, 1):
        rows += oracle_rows(gt[rec.image_id], segmenter, load_image(rec), pp_cfg, small_max)
        print(f"{k}/{len(recs)} {rec.image_id}", flush=True)

    out = {
        "model_id": cfg["segmentation"]["model_id"],
        "postprocess": pp_cfg,
        "oracle_gt_boxes": summarize_oracle(rows, cfg),
    }
    if run_dir is not None:
        scenes = load_scenes(run_dir)
        ids = [r.image_id for r in recs if r.image_id in scenes]
        preds = predictions_from_scenes({i: scenes[i] for i in ids})
        box = evaluate_from_config(preds, gt, cfg, iou_type="bbox", image_ids=ids)
        segm = evaluate_from_config(preds, gt, cfg, iou_type="segm", image_ids=ids)
        out["predicted_boxes"] = {"run": str(run_dir), **predicted_pairs_summary(box, segm)}

    dest = run_dir or Path(cfg["paths"]["outputs"]).parent / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "segmentation_oracle.csv", index=False)
    (dest / "metrics_segmentation_oracle.json").write_text(json.dumps(out, indent=2), "utf-8")
    print(json.dumps(out, indent=2))
    print(f"salidas en {dest}")


if __name__ == "__main__":
    main()
