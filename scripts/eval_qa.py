"""Evaluación de QA (E1-24) sobre un run con `questions` y, si están, `questions_llm`.

Uso:
    python scripts/eval_qa.py --config configs/base.yaml configs/kaggle.yaml --run <run>

El run tiene que tener las 20 imágenes del benchmark con `run_qa.py` (y `run_qa_llm.py`) ya
corridos. Arma el SceneJSON oráculo de cada imagen desde el GT y responde con el mismo ejecutor.
Escribe en el run: metrics_qa.json y qa_eval.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.annotations import annotations_path, load_annotations
from eval.oracle import oracle_scene
from eval.qa_eval import evaluate_mode, summarize
from scene.config import load_config
from scene.io.dataset import list_images, load_image, load_questions, load_vocabulary
from scene.pipeline import load_scenes
from scene.qa.category_matcher import load_lexicon, matcher_from_config, spanish_names
from scene.qa.executor import executor_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    cfg = load_config(*args.config)
    run = Path(args.run)
    gt = load_annotations(annotations_path(cfg))
    categories = load_vocabulary(cfg)
    names = {c.id: c.name for c in categories}
    matcher = matcher_from_config(cfg, categories)
    executor = executor_from_config(
        cfg,
        matcher.match_id,
        names,
        display_names=spanish_names(categories, load_lexicon(cfg["qa"]["matcher"]["lexicon_es"])),
    )
    phrases = cfg["qa"]["adapter"]["relation_phrases"]
    benchmark = load_questions(cfg)
    scenes = load_scenes(run)
    recs = {r.image_id: r for r in list_images(cfg, "benchmark")}
    oracles = {iid: oracle_scene(gt[iid], load_image(rec), cfg, names) for iid, rec in recs.items()}

    frames = []
    for mode, field in (("deterministic", "questions"), ("llm", "questions_llm")):
        if any(s.get(field) for s in scenes.values()):
            frames.append(
                evaluate_mode(mode, benchmark, scenes, oracles, executor, phrases, matcher.match_id)
            )
    if not frames:
        raise SystemExit(f"{run} no tiene questions ni questions_llm: correr antes run_qa.py")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(run / "qa_eval.csv", index=False)
    summary = summarize(df)
    (run / "metrics_qa.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    print(json.dumps(summary, indent=2, default=str))
    wrong = df[(df["correct"] == False) & df["type"].ne("description")]  # noqa: E712
    if len(wrong):
        cols = ["mode", "question_id", "type", "answer", "oracle_answer", "status", "error_source"]
        print(wrong[cols].to_string(index=False))


if __name__ == "__main__":
    main()
