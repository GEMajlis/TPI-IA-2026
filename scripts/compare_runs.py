"""Compara un run experimental contra la línea base (dev) y lo registra en experiments.csv.

Uso:
    python scripts/run_pipeline.py --config configs/base.yaml configs/kaggle.yaml \\
        configs/experiments/e1_30_pp_solo_chicas.yaml --split dev
    python scripts/compare_runs.py --config configs/base.yaml configs/kaggle.yaml \\
        --base outputs/runs/<run_base> --exp latest --name e1_30_pp_solo_chicas \\
        --hypothesis "post-proceso solo en máscaras chicas sube el F1 de máscaras" \\
        --change "segmentation.postprocess.apply_below_area_ratio: null -> 0.00333"

Las métricas se calculan en el mismo split tuning/holdout de E1-29. La decisión se toma con el
holdout. Escribe una fila en <outputs>/../experiments.csv y experiment_<name>.json en el run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from eval.annotations import annotations_path, load_annotations
from eval.experiments import append_row, dev_splits, run_metrics, stage_cost_s
from scene.config import load_config
from scene.pipeline import load_scenes
from scene.tracing.tracer import read_trace


def resolve(cfg: dict, run: str) -> Path:
    if run != "latest":
        return Path(run)
    runs = Path(cfg["paths"]["outputs"]).iterdir()
    return sorted(p for p in runs if (p / "scenes").is_dir())[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--base", required=True, help="run de la línea base")
    parser.add_argument("--exp", required=True, help="run del experimento ('latest')")
    parser.add_argument("--name", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True, help="la única variable que cambia")
    parser.add_argument("--conclusion", default="", help="se puede completar después")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    base_dir, exp_dir = resolve(cfg, args.base), resolve(cfg, args.exp)
    if base_dir == exp_dir:
        raise SystemExit("--base y --exp son el mismo run")
    gt = load_annotations(annotations_path(cfg))
    splits = dev_splits(gt, cfg)
    base = run_metrics(load_scenes(base_dir), gt, cfg, splits)
    exp = run_metrics(load_scenes(exp_dir), gt, cfg, splits)
    cost_base = stage_cost_s(read_trace(base_dir / cfg["tracing"]["trace_file"]))
    cost_exp = stage_cost_s(read_trace(exp_dir / cfg["tracing"]["trace_file"]))

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "name": args.name,
        "hypothesis": args.hypothesis,
        "change": args.change,
        "base_run": base_dir.name,
        "exp_run": exp_dir.name,
    }
    for key in base:
        row[f"base_{key}"] = round(base[key], 4)
        row[f"exp_{key}"] = round(exp[key], 4)
    row["delta_holdout_f1_box"] = round(exp["holdout_f1_box"] - base["holdout_f1_box"], 4)
    row["delta_holdout_f1_mask"] = round(exp["holdout_f1_mask"] - base["holdout_f1_mask"], 4)
    row["delta_tuning_f1_mask"] = round(exp["tuning_f1_mask"] - base["tuning_f1_mask"], 4)
    # Costo: segundos de segmentación de cada run (la detección suele venir de caché).
    row["base_seg_s"] = round(cost_base["segmentation"], 1)
    row["exp_seg_s"] = round(cost_exp["segmentation"], 1)
    row["conclusion"] = args.conclusion

    csv_path = append_row(Path(cfg["paths"]["outputs"]).parent / "experiments.csv", row)
    (exp_dir / f"experiment_{args.name}.json").write_text(json.dumps(row, indent=2), "utf-8")
    print(f"{'métrica':<28}{'base':>9}{'exp':>9}{'Δ':>9}")
    for key in base:
        print(f"{key:<28}{base[key]:>9.4f}{exp[key]:>9.4f}{exp[key] - base[key]:>+9.4f}")
    print(f"registrado en {csv_path}")


if __name__ == "__main__":
    main()
