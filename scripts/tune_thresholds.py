"""Barrido del umbral de detección (y NMS entre clases) sobre la caché de detecciones crudas.

Uso (después de correr la pipeline sobre dev, así la caché tiene las 120 imágenes):
    python scripts/tune_thresholds.py --config configs/base.yaml configs/kaggle.yaml
    python scripts/tune_thresholds.py --raw-dir outputs/runs/<run_id>/detections_raw

Elige el umbral de máximo F1 en tuning (80 imágenes) y lo reporta en holdout (40), junto con
el umbral actual de la config como referencia. No cambia `base.yaml`: el valor elegido se
copia a mano después de revisar la curva.

Salidas en outputs/tuning/<config_hash>/: curve.csv, curve.png, split.json y summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.annotations import annotations_path, load_annotations  # noqa: E402
from eval.tuning import choose, split_tuning_holdout, sweep  # noqa: E402
from scene.config import config_hash, load_config  # noqa: E402
from scene.detection.base import load_detections  # noqa: E402
from scene.pipeline import stage_keys  # noqa: E402


def parse_range(text: str) -> list[float]:
    start, stop, step = (float(x) for x in text.split(":"))
    return [round(v, 4) for v in np.arange(start, stop + step / 2, step)]


def plot(curve, out: Path, chosen: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for (variant, split), g in curve.groupby(["variant", "split"]):
        style = "-" if split == "tuning" else "--"
        ax.plot(g["threshold"], g["f1"], style, label=f"{variant} · {split}")
    for variant, thr in chosen.items():
        ax.axvline(thr, color="gray", lw=0.8, ls=":")
        ax.annotate(f"{variant}: {thr:.2f}", (thr, ax.get_ylim()[0]), fontsize=8, rotation=90)
    ax.set_xlabel("umbral de score")
    ax.set_ylabel("F1 de cajas (federado, IoU ≥ 0,5)")
    ax.set_title("Ajuste de umbral: tuning (—) y holdout (--)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--raw-dir", help="carpeta con <image_id>.json crudos (por defecto, caché)")
    parser.add_argument("--thresholds", default="0.02:0.5:0.02", help="inicio:fin:paso")
    parser.add_argument("--holdout", type=int, default=40)
    parser.add_argument("--out", default="outputs/tuning")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir
        else Path(cfg["paths"]["cache"]) / "detections_raw" / stage_keys(cfg)["detection"]
    )
    gt = load_annotations(annotations_path(cfg))
    missing = [i for i in gt if not (raw_dir / f"{i}.json").is_file()]
    if missing:
        raise SystemExit(
            f"Faltan {len(missing)} imágenes de dev en {raw_dir} (por ejemplo {missing[:3]}): "
            "correr antes scripts/run_pipeline.py --split dev"
        )
    raw = {i: load_detections(raw_dir / f"{i}.json") for i in gt}
    floor = cfg["detection"]["owlv2"]["score_floor"]
    thresholds = [t for t in parse_range(args.thresholds) if t >= floor]

    tuning, holdout = split_tuning_holdout(gt, sorted(gt), args.holdout, cfg["seed"])
    splits = {"tuning": tuning, "holdout": holdout}
    cross_iou = cfg["detection"]["cross_class_nms"]["iou"]
    variants = {
        "base": {"cross_class_nms": {"enabled": False, "iou": cross_iou}},
        f"cross_nms_{cross_iou}": {"cross_class_nms": {"enabled": True, "iou": cross_iou}},
    }
    curve = sweep(raw, gt, splits, cfg, thresholds, variants)

    out = Path(args.out) / config_hash(cfg)[:8]
    out.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out / "curve.csv", index=False)
    (out / "split.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")

    current = cfg["detection"]["score_threshold"]
    summary: dict = {"raw_dir": str(raw_dir), "current_threshold": current, "variants": {}}
    for variant in variants:
        best = choose(curve, variant)
        hold = curve[
            (curve["variant"] == variant)
            & (curve["split"] == "holdout")
            & np.isclose(curve["threshold"], best["threshold"])
        ].iloc[0]
        summary["variants"][variant] = {
            "chosen_threshold": float(best["threshold"]),
            "tuning": best[["precision", "recall", "f1", "preds_per_image"]].to_dict(),
            "holdout": hold[["precision", "recall", "f1", "preds_per_image"]].to_dict(),
        }
    ref = curve[
        (curve["variant"] == "base")
        & (curve["split"] == "holdout")
        & np.isclose(curve["threshold"], current)
    ]
    if len(ref):
        summary["holdout_at_current_threshold"] = ref.iloc[0][
            ["precision", "recall", "f1"]
        ].to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(
        curve, out / "curve.png", {v: s["chosen_threshold"] for v, s in summary["variants"].items()}
    )
    print(json.dumps(summary, indent=2))
    print(f"salidas en {out}")


if __name__ == "__main__":
    main()
