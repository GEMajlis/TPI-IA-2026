"""EDA del dataset de desarrollo: tablas (CSV), figuras (PNG) y resumen (JSON).

Uso: python scripts/run_eda.py --config configs/base.yaml [--out outputs/eda]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eda import plots
from eda.stats import (
    benchmark_table,
    categories_table,
    images_table,
    instances_table,
    overlap_table,
)
from eval.annotations import annotations_path, load_annotations
from scene.config import load_config
from scene.io.dataset import load_questions, load_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--out", default="outputs/eda")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    gt = load_annotations(annotations_path(cfg))
    vocab = load_vocabulary(cfg)
    names = {c.id: c.name for c in vocab}
    sizes = cfg["scene"]["size_thresholds"]
    freq = cfg["eval"]["frequency_bins"]

    instances = instances_table(gt)
    images = images_table(gt, len(vocab))
    categories = categories_table(gt, names)
    overlaps = overlap_table(gt, cfg["eval"]["eda_duplicate_iou"])
    bench = benchmark_table(load_questions(cfg), gt, {v: k for k, v in names.items()})
    tables = {
        "instances": instances,
        "images": images,
        "categories": categories,
        "overlaps": overlaps,
        "benchmark": bench,
    }
    for name, df in tables.items():
        df.to_csv(out / f"eda_{name}.csv", index=False)

    figures = {
        "instances_per_image": plots.instances_per_image(
            images, cfg["detection"]["max_detections"]
        ),
        "area_ratio": plots.area_ratio(instances, sizes["small_max"], sizes["medium_max"]),
        "category_frequency": plots.category_frequency(
            categories, freq["rare_max"], freq["common_max"]
        ),
        "federated_status": plots.federated_status(images),
        "occlusion": plots.occlusion(overlaps),
        "position_grid": plots.position_grid(instances),
        "benchmark_types": plots.benchmark_types(bench),
    }
    for name, fig in figures.items():
        fig.savefig(out / "figures" / f"{name}.png", dpi=130, bbox_inches="tight")

    a = instances["area_ratio"]
    n_img = categories["n_images_positive"]
    summary = {
        "n_images": len(images),
        "n_instances": len(instances),
        "instances_per_image": images["n_instances"].describe().round(2).to_dict(),
        "area_ratio_quantiles": a.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(5).to_dict(),
        "share_area_lt_0.1pct": round(float((a < 0.001).mean()), 3),
        "share_area_lt_1pct": round(float((a < 0.01).mean()), 3),
        "size_split": {
            "small": round(float((a < sizes["small_max"]).mean()), 3),
            "medium": round(
                float(((a >= sizes["small_max"]) & (a < sizes["medium_max"])).mean()), 3
            ),
            "large": round(float((a >= sizes["medium_max"]).mean()), 3),
        },
        "category_images_range": [int(n_img.min()), int(n_img.max())],
        "frequency_split": {
            "rare": int((n_img <= freq["rare_max"]).sum()),
            "common": int(((n_img > freq["rare_max"]) & (n_img <= freq["common_max"])).sum()),
            "frequent": int((n_img > freq["common_max"]).sum()),
        },
        "federated_per_image_mean": images[
            ["n_positive", "n_negative", "n_not_exhaustive", "n_unverified"]
        ]
        .mean()
        .round(2)
        .to_dict(),
        "share_instances_overlapping": round(float((overlaps["max_covered"] > 0).mean()), 3),
        "share_instances_covered_gt_50pct": round(float((overlaps["max_covered"] > 0.5).mean()), 3),
        "share_same_class_duplicates": round(float(overlaps["same_class_dup"].mean()), 3),
        "position_share": instances["position"].value_counts(normalize=True).round(3).to_dict(),
        "benchmark_types": bench["type"].value_counts().to_dict(),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
