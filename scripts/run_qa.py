"""QA determinista (E1-23): agrega `questions` a los SceneJSON de las imágenes del benchmark.

Uso:
    python scripts/run_qa.py --run outputs/runs/<run_id>
    python scripts/run_qa.py --config configs/base.yaml configs/kaggle.yaml --run latest

Reescribe `scenes/<image_id>.json` del run (reemplaza `questions` si ya estaba), traza cada
imagen en `trace.jsonl` (etapa `qa_deterministic`) y guarda `qa_deterministic.csv`. Después,
`make_submission.py` toma las escenas con sus respuestas.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scene.config import config_hash, data_path, load_config
from scene.io.dataset import load_questions, load_vocabulary
from scene.pipeline import load_scenes
from scene.qa.category_matcher import load_lexicon, matcher_from_config, spanish_names
from scene.qa.executor import executor_from_config
from scene.qa.structured_adapter import answer_questions, group_by_image
from scene.scene.schema import assert_valid_scene
from scene.tracing.tracer import Tracer


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
    args = parser.parse_args()

    cfg = load_config(*args.config)
    run = resolve_run(cfg, args.run)
    categories = load_vocabulary(cfg)
    matcher = matcher_from_config(cfg, categories)
    lexicon = load_lexicon(cfg["qa"]["matcher"]["lexicon_es"])
    executor = executor_from_config(
        cfg,
        matcher.match_id,
        {c.id: c.name for c in categories},
        display_names=spanish_names(categories, lexicon),
    )
    phrases = cfg["qa"]["adapter"]["relation_phrases"]
    scenes = load_scenes(run)
    tracer = Tracer(run / cfg["tracing"]["trace_file"], run.name, config_hash(cfg))
    schema = data_path(cfg, "scene_schema")

    rows, missing = [], []
    for iid, questions in group_by_image(load_questions(cfg)).items():
        if iid not in scenes:
            missing.append(iid)
            continue
        with (
            tracer.image_guard(iid),
            tracer.stage(iid, "qa_deterministic", n_in=len(questions)) as s,
        ):
            scene = scenes[iid]
            scene["questions"] = answer_questions(questions, scene, executor, phrases)
            assert_valid_scene(scene, schema)
            path = run / "scenes" / f"{iid}.json"
            path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
            s.n_out = len(scene["questions"])
            s.artifact_path = str(path)
            s.extra["status"] = dict(Counter(q["status"] for q in scene["questions"]))
            for q in scene["questions"]:
                rows.append(
                    {
                        "image_id": iid,
                        "question_id": q["question_id"],
                        "type": q["type"],
                        "status": q["status"],
                        "normalized_answer": q["normalized_answer"],
                        "confidence": q["confidence"],
                        "n_evidence": len(q["evidence"]),
                        "detail": q.get("status_detail"),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(run / "qa_deterministic.csv", index=False)
    print(f"{run}: {len(df)} respuestas en {df.image_id.nunique() if len(df) else 0} imágenes")
    if len(df):
        print(df.groupby(["type", "status"]).size().to_string())
    if missing or tracer.failed_images:
        raise SystemExit(f"Sin escena: {missing}; con error: {tracer.failed_images}")


if __name__ == "__main__":
    main()
