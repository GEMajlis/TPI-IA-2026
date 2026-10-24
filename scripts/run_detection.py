"""Detección + post-proceso sobre un split o sobre imágenes sueltas, con traza y caché.

Uso:
    # smoke test en CPU con 2 imágenes de dev
    python scripts/run_detection.py --split dev --limit 2 --debug
    # imágenes arbitrarias (fuera del dataset)
    python scripts/run_detection.py --images foto1.jpg foto2.jpg --debug
    # dev completo en Kaggle + F1 de cajas
    python scripts/run_detection.py --config configs/base.yaml configs/kaggle.yaml \\
        --split dev --eval

Salidas en outputs/runs/<run_id>/: detections_raw/ (caché antes del umbral), detections/,
debug/ (PNG), trace.jsonl, run_manifest.json y, con --eval, metrics_detection.json.
"""

from __future__ import annotations

import argparse
import json

from scene.config import load_config, start_run
from scene.detection.base import load_detections, save_detections
from scene.detection.debug import draw_detections
from scene.detection.owlv2 import build_detector
from scene.detection.postprocess import postprocess
from scene.detection.vocabulary import vocabulary_from_config
from scene.io.dataset import image_record_from_path, list_images, load_image, load_vocabulary
from scene.tracing.tracer import Tracer


def evaluate_boxes(cfg: dict, run_dir, image_ids: list[str], vocab) -> dict:
    """F1 federado de cajas (GT de dev, fuera de la inferencia). Una imagen que falló cuenta como
    imagen sin predicciones, para no inflar el recall."""
    from eval.annotations import annotations_path, load_annotations
    from eval.lvis_federated import Prediction, evaluate_from_config, save_metrics

    gt = load_annotations(annotations_path(cfg))
    preds = {}
    for i in image_ids:
        path = run_dir / "detections" / f"{i}.json"
        dets = load_detections(path) if path.is_file() else []
        preds[i] = [Prediction(d.category_id, d.score, d.box_xyxy) for d in dets]
    result = evaluate_from_config(preds, gt, cfg, iou_type="bbox", image_ids=image_ids)
    save_metrics(result, run_dir, "detection", {c.id: c.name for c in vocab.categories})
    return result.summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--set", nargs="*", default=[], help="overrides clave.sub=valor")
    parser.add_argument("--split", choices=["dev", "test", "benchmark"], default="dev")
    parser.add_argument("--images", nargs="*", help="imágenes sueltas (ignora --split)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="guarda PNG con las cajas")
    parser.add_argument("--eval", action="store_true", help="F1 de cajas (solo split dev)")
    args = parser.parse_args()

    cfg = load_config(*args.config, dotlist=args.set)
    ctx = start_run(cfg)
    tracer = Tracer(ctx.trace_path, run_id=ctx.run_id, config_hash=ctx.config_hash)
    vocab = vocabulary_from_config(cfg, load_vocabulary(cfg))
    detector = build_detector(cfg, vocab, ctx.device)
    ctx.update_manifest("models", {"detector": detector.params()})

    if args.images:
        images = [image_record_from_path(p) for p in args.images]
    else:
        images = list_images(cfg, args.split)
    images = images[: args.limit]
    det_cfg = cfg["detection"]
    pp_params = {k: det_cfg[k] for k in ("score_threshold", "nms_iou", "max_detections")}
    pp_params["cross_class_nms"] = det_cfg["cross_class_nms"]

    done: list[str] = []
    for rec in images:
        with tracer.image_guard(rec.image_id):
            image = load_image(rec)
            with tracer.stage(
                rec.image_id, "detection", detector.params(), detector.model_id, n_in=1
            ) as s:
                raw = detector.detect(image)
                raw_path = ctx.run_dir / "detections_raw" / f"{rec.image_id}.json"
                save_detections(raw, raw_path)
                s.n_out, s.artifact_path = len(raw), str(raw_path)
            with tracer.stage(rec.image_id, "detection_postprocess", pp_params, n_in=len(raw)) as s:
                res = postprocess(raw, det_cfg)
                out_path = ctx.run_dir / "detections" / f"{rec.image_id}.json"
                save_detections(res.detections, out_path)
                s.n_out, s.artifact_path = len(res.detections), str(out_path)
                s.extra["counts"] = res.counts
            if args.debug:
                draw_detections(
                    image, res.detections, ctx.run_dir / "debug" / f"{rec.image_id}.png"
                )
            done.append(rec.image_id)
            print(f"{rec.image_id}: {res.counts}")

    print(f"run {ctx.run_id}: {len(done)} ok, {len(tracer.failed_images)} con error")
    if args.eval:
        if args.images or args.split != "dev":
            raise SystemExit("--eval solo tiene GT para el split dev")
        summary = evaluate_boxes(cfg, ctx.run_dir, [r.image_id for r in images], vocab)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
