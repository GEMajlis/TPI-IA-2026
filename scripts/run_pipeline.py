"""Pipeline completa (detección → SAM2 → SceneJSON) sobre uno o más splits o imágenes sueltas.

Uso:
    # imagen cualquiera (fuera del dataset), en CPU con el SAM2 chico
    python scripts/run_pipeline.py --images foto.jpg --debug \\
        --set segmentation.model_id=facebook/sam2.1-hiera-tiny
    # smoke test
    python scripts/run_pipeline.py --split dev --limit 3 --debug
    # dev completo en Kaggle + F1 de cajas y máscaras
    python scripts/run_pipeline.py --config configs/base.yaml configs/kaggle.yaml --split dev --eval
    # las 80 imágenes del submission (después: scripts/make_submission.py --run <run>)
    python scripts/run_pipeline.py --config configs/base.yaml configs/kaggle.yaml \\
        --split test benchmark

Salidas en outputs/runs/<run_id>/: scenes/ (SceneJSON por imagen), debug/, trace.jsonl,
run_manifest.json y, con --eval, metrics_detection.json y metrics_segmentation.json.
La caché de detecciones crudas y máscaras queda en paths.cache y se reutiliza entre runs.
"""

from __future__ import annotations

import argparse
import json

from scene.config import load_config, start_run
from scene.detection.owlv2 import build_detector
from scene.detection.vocabulary import vocabulary_from_config
from scene.io.dataset import image_record_from_path, list_images, load_vocabulary
from scene.pipeline import STAGES, Pipeline
from scene.segmentation.sam2 import build_segmenter
from scene.tracing.tracer import Tracer, read_trace


def evaluate(cfg: dict, run_dir, scenes: dict, image_ids: list[str], names: dict) -> dict:
    """F1 federado de cajas y máscaras desde los SceneJSON (GT de dev, fuera de la inferencia)."""
    from eval.annotations import annotations_path, load_annotations
    from eval.lvis_federated import evaluate_from_config, predictions_from_scenes, save_metrics

    gt = load_annotations(annotations_path(cfg))
    preds = predictions_from_scenes(scenes)
    out = {}
    for iou_type, name in (("bbox", "detection"), ("segm", "segmentation")):
        result = evaluate_from_config(preds, gt, cfg, iou_type=iou_type, image_ids=image_ids)
        save_metrics(result, run_dir, name, names)
        out[name] = {k: result.summary[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--set", nargs="*", default=[], help="overrides clave.sub=valor")
    parser.add_argument("--split", nargs="+", default=["dev"], choices=["dev", "test", "benchmark"])
    parser.add_argument("--images", nargs="*", help="imágenes sueltas (ignora --split)")
    parser.add_argument("--limit", type=int, default=None, help="tope de imágenes por split")
    parser.add_argument("--from-stage", choices=STAGES, help="ignora la caché desde esta etapa")
    parser.add_argument("--debug", action="store_true", help="PNG con cajas y máscaras")
    parser.add_argument("--eval", action="store_true", help="F1 de cajas y máscaras (solo dev)")
    args = parser.parse_args()

    cfg = load_config(*args.config, dotlist=args.set)
    ctx = start_run(cfg)
    tracer = Tracer(ctx.trace_path, run_id=ctx.run_id, config_hash=ctx.config_hash)
    vocab = vocabulary_from_config(cfg, load_vocabulary(cfg))

    if args.images:
        images = [image_record_from_path(p) for p in args.images]
    else:
        seen, images = set(), []
        for split in args.split:  # benchmark ⊂ dev: no repetir imágenes
            for rec in list_images(cfg, split)[: args.limit]:
                if rec.image_id not in seen:
                    seen.add(rec.image_id)
                    images.append(rec)

    pipe = Pipeline(
        cfg,
        ctx,
        tracer,
        vocab,
        make_detector=lambda: build_detector(cfg, vocab, ctx.device),
        make_segmenter=lambda: build_segmenter(cfg, ctx.device),
        from_stage=args.from_stage,
        debug=args.debug,
    )
    scenes = pipe.run(images)
    ctx.update_manifest("images", [r.image_id for r in images])

    stages: dict[str, dict[str, int]] = {}
    for r in read_trace(ctx.trace_path):
        stages.setdefault(r["stage"], {}).setdefault(r["status"], 0)
        stages[r["stage"]][r["status"]] += 1
    print(f"run {ctx.run_id}: {len(scenes)} escenas, {len(tracer.failed_images)} con error")
    print(json.dumps(stages))
    if tracer.failed_images:
        print("con error:", tracer.failed_images)

    if args.eval:
        if args.images or args.split != ["dev"]:
            raise SystemExit("--eval solo tiene GT para el split dev")
        names = {c.id: c.name for c in vocab.categories}
        summary = evaluate(cfg, ctx.run_dir, scenes, [r.image_id for r in images], names)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
