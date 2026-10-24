"""QA con LLM (E1-28): agrega `questions_llm` a los SceneJSON de las imágenes del benchmark.

Uso (en Kaggle, con GPU):
    pip install -e ".[llm]"
    python scripts/run_qa_llm.py --config configs/base.yaml configs/kaggle.yaml --run <run>

El LLM recibe solo el texto de cada pregunta y produce el JSON intermedio (DSL); lo ejecuta el
mismo motor determinista de `run_qa.py`. Reescribe `scenes/<image_id>.json` del run (reemplaza
`questions_llm`), traza cada imagen (etapa `qa_llm`) y guarda `qa_llm.csv`. Después se corre
`make_submission.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scene.config import config_hash, data_path, load_config, resolve_device, set_seed
from scene.io.dataset import load_questions, load_vocabulary
from scene.llm.describer import build_describer
from scene.llm.interpreter import build_interpreter
from scene.llm.model import build_llm
from scene.pipeline import load_scenes
from scene.qa.category_matcher import load_lexicon, matcher_from_config, spanish_names
from scene.qa.executor import executor_from_config
from scene.qa.llm_mode import answer_questions_llm
from scene.qa.structured_adapter import group_by_image
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
    parser.add_argument("--set", nargs="*", default=[], help="overrides clave.sub=valor")
    parser.add_argument("--run", default="latest")
    args = parser.parse_args()

    cfg = load_config(*args.config, dotlist=args.set)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))
    run = resolve_run(cfg, args.run)
    categories = load_vocabulary(cfg)
    names = {c.id: c.name for c in categories}
    matcher = matcher_from_config(cfg, categories, device)
    lexicon = load_lexicon(cfg["qa"]["matcher"]["lexicon_es"])
    executor = executor_from_config(
        cfg, matcher.match_id, names, display_names=spanish_names(categories, lexicon)
    )
    llm = build_llm(cfg, device=device)
    interpreter = build_interpreter(
        cfg, llm, category_matcher=matcher.match_id, category_names=names
    )
    describer = build_describer(cfg, llm, category_names=names)

    scenes = load_scenes(run)
    tracer = Tracer(run / cfg["tracing"]["trace_file"], run.name, config_hash(cfg))
    schema = data_path(cfg, "scene_schema")
    candidate = cfg["llm"]["candidates"][cfg["llm"]["selected"]]

    rows, missing = [], []
    for iid, questions in group_by_image(load_questions(cfg)).items():
        if iid not in scenes:
            missing.append(iid)
            continue
        with (
            tracer.image_guard(iid),
            tracer.stage(
                iid,
                "qa_llm",
                params=cfg["llm"],
                model_id=candidate["model_id"],
                n_in=len(questions),
            ) as s,
        ):
            scene = scenes[iid]
            scene["questions_llm"] = answer_questions_llm(
                questions, scene, interpreter, executor, describer, names
            )
            assert_valid_scene(scene, schema)
            path = run / "scenes" / f"{iid}.json"
            path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
            s.n_out = len(scene["questions_llm"])
            s.artifact_path = str(path)
            s.extra["status"] = dict(Counter(q["status"] for q in scene["questions_llm"]))
            s.extra["interpreter"] = dict(
                Counter(q["interpreter"]["status"] for q in scene["questions_llm"])
            )
            for q in scene["questions_llm"]:
                rows.append(
                    {
                        "image_id": iid,
                        "question_id": q["question_id"],
                        "type": q["type"],
                        "interpreter_status": q["interpreter"]["status"],
                        "attempts": q["interpreter"]["attempts"],
                        "op": q["interpreted_query"]["op"],
                        "status": q["status"],
                        "normalized_answer": q["normalized_answer"],
                        "confidence": q["confidence"],
                        "describer_status": q["interpreter"].get("describer_status"),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(run / "qa_llm.csv", index=False)
    print(f"{run}: {len(df)} respuestas LLM en {df.image_id.nunique() if len(df) else 0} imágenes")
    if len(df):
        print(df.groupby(["type", "interpreter_status", "status"]).size().to_string())
    if missing or tracer.failed_images:
        raise SystemExit(f"Sin escena: {missing}; con error: {tracer.failed_images}")


if __name__ == "__main__":
    main()
