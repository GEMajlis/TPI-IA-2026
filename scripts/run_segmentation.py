"""Segmentación con SAM2 sobre las detecciones cacheadas de un run de `run_detection.py`.

Uso:
    # smoke test en CPU con el modelo chico
    python scripts/run_segmentation.py --from-run latest --limit 2 --debug \\
        --set segmentation.model_id=facebook/sam2.1-hiera-tiny
    # dev completo en Kaggle + F1 de cajas y de máscaras
    python scripts/run_segmentation.py --config configs/base.yaml configs/kaggle.yaml \\
        --from-run latest --eval

Lee `<from-run>/detections/*.json` y escribe en un run nuevo: masks/ (RLE por imagen), debug/,
trace.jsonl, run_manifest.json y, con --eval, metrics_detection.json y metrics_segmentation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene.config import load_config, start_run
from scene.detection.base import load_detections
from scene.io.dataset import list_images, load_image, load_vocabulary
from scene.masks import rle
from scene.segmentation.debug import draw_masks
from scene.segmentation.sam2 import build_segmenter, load_segmented, save_segmented
from scene.tracing.tracer import Tracer


def resolve_run(cfg: dict, from_run: str) -> Path:
    """`latest` = el run más reciente que tenga detecciones."""
    if from_run != "latest":
        return Path(from_run)
    runs = sorted(p for p in Path(cfg["paths"]["outputs"]).iterdir() if (p / "detections").is_dir())
    if not runs:
        raise SystemExit("No hay runs con detecciones: correr antes scripts/run_detection.py")
    return runs[-1]


def evaluate(cfg: dict, run_dir: Path, image_ids: list[str], names: dict[int, str]) -> dict:
    """F1 federado de cajas y de máscaras (GT de dev, fuera de la inferencia)."""
    from eval.annotations import annotations_path, load_annotations
    from eval.lvis_federated import Prediction, evaluate_from_config, save_metrics

    gt = load_annotations(annotations_path(cfg))
    preds = {}
    for i in image_ids:
        path = run_dir / "masks" / f"{i}.json"
        objs = load_segmented(path) if path.is_file() else []
        preds[i] = [
            Prediction(
                o.detection.category_id, o.detection.score, o.detection.box_xyxy, rle.encode(o.mask)
            )
            for o in objs
        ]
    out = {}
    for iou_type, name in (("bbox", "detection"), ("segm", "segmentation")):
        result = evaluate_from_config(preds, gt, cfg, iou_type=iou_type, image_ids=image_ids)
        save_metrics(result, run_dir, name, names)
        out[name] = result.summary
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--set", nargs="*", default=[], help="overrides clave.sub=valor")
    parser.add_argument("--from-run", default="latest", help="run con detections/ o 'latest'")
    parser.add_argument("--split", choices=["dev", "test", "benchmark"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="guarda PNG con las máscaras")
    parser.add_argument("--eval", action="store_true", help="F1 de cajas y máscaras (solo dev)")
    args = parser.parse_args()

    cfg = load_config(*args.config, dotlist=args.set)
    src_run = resolve_run(cfg, args.from_run)
    ctx = start_run(cfg)
    ctx.update_manifest("from_run", str(src_run))
    tracer = Tracer(ctx.trace_path, run_id=ctx.run_id, config_hash=ctx.config_hash)
    segmenter = build_segmenter(cfg, ctx.device)
    ctx.update_manifest("models", {"segmenter": segmenter.params()})

    available = {p.stem for p in (src_run / "detections").glob("*.json")}
    images = [r for r in list_images(cfg, args.split) if r.image_id in available][: args.limit]
    print(f"{len(images)} imágenes con detecciones en {src_run}")

    for rec in images:
        with tracer.image_guard(rec.image_id):
            dets = load_detections(src_run / "detections" / f"{rec.image_id}.json")
            image = load_image(rec)
            with tracer.stage(
                rec.image_id, "segmentation", segmenter.params(), segmenter.model_id, len(dets)
            ) as s:
                objs = segmenter.segment(image, dets)
                path = ctx.run_dir / "masks" / f"{rec.image_id}.json"
                save_segmented(objs, path)
                s.n_out, s.artifact_path = len(objs), str(path)
                s.extra["empty_masks"] = sum(1 for o in objs if not o.mask.any())
                s.extra["px_removed_by_postprocess"] = sum(
                    o.raw_area - int(o.mask.sum()) for o in objs
                )
            if args.debug:
                draw_masks(image, objs, ctx.run_dir / "debug" / f"{rec.image_id}.png")
            print(f"{rec.image_id}: {len(objs)} máscaras")

    print(
        f"run {ctx.run_id}: {len(images) - len(tracer.failed_images)} ok, "
        f"{len(tracer.failed_images)} con error"
    )
    if args.eval:
        if args.split != "dev":
            raise SystemExit("--eval solo tiene GT para el split dev")
        names = {c.id: c.name for c in load_vocabulary(cfg)}
        summary = evaluate(cfg, ctx.run_dir, [r.image_id for r in images], names)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
