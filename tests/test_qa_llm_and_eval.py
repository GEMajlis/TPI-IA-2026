"""E1-28 (modo LLM) y E1-24 (evaluación de QA) con un LLM falso y escenas sintéticas."""

import copy
import json
import re

import numpy as np
import pandas as pd
import pytest

from eval.annotations import GTInstance, ImageGT
from eval.oracle import oracle_scene
from eval.qa_eval import evaluate_mode, evidence_supports, summarize
from scene.detection.base import Detection
from scene.llm.describer import SceneDescriber
from scene.llm.interpreter import Interpreter
from scene.qa.executor import executor_from_config
from scene.qa.llm_mode import answer_questions_llm
from scene.qa.structured_adapter import answer_questions
from scene.scene.builder import build_scene
from scene.scene.schema import validate_scene
from scene.segmentation.sam2 import SegmentedObject
from tests.conftest import ROOT

H, W = 100, 200
NAMES = {1: "cup", 2: "plate", 3: "clock"}
IDS = {"cup": 1, "taza": 1, "tazas": 1, "plate": 2, "plato": 2, "clock": 3, "reloj": 3}
BOXES = [(1, (10, 40, 30, 60)), (1, (50, 40, 70, 60)), (2, (150, 40, 190, 80))]  # 2 tazas, 1 plato
SCHEMA = ROOT / "data" / "competition" / "scene_schema.json"


@pytest.fixture
def cfg(base_cfg):
    base_cfg["paths"]["data_root"] = str(ROOT / "data" / "competition")
    base_cfg["paths"]["scene_schema"] = "scene_schema.json"
    return base_cfg


def resolve(text: str):
    return IDS.get(text.strip().lower())


def scene_from(boxes, cfg) -> dict:
    segs = []
    for k, (cid, (x1, y1, x2, y2)) in enumerate(boxes):
        m = np.zeros((H, W), bool)
        m[y1:y2, x1:x2] = True
        d = Detection((x1, y1, x2, y2), 0.9 - k / 10, cid, NAMES[cid], "fake", NAMES[cid])
        segs.append(SegmentedObject(d, m, 0.9, int(m.sum())))
    return build_scene("img", np.zeros((H, W, 3), np.uint8), segs, cfg)[0]


def bench(qid: str, qtype: str, question: str, classes: list[str]) -> dict:
    return {"question_id": qid, "image_file": "img.jpg", "type": qtype, "question": question,
            "class_en": classes}  # fmt: skip


BENCH = [
    bench("Q-COUNT-CUP", "count", '¿Cuántas instancias de "taza" hay?', ["cup"]),
    bench("Q-COUNT-PLATE", "count", '¿Cuántas instancias de "plato" hay?', ["plate"]),
    bench("Q-EXISTS", "existence", '¿Hay al menos una instancia de "reloj"?', ["clock"]),
    bench("Q-DESC", "description", "Describí brevemente la escena.", []),
]


def dsl(op, cat=None):
    t = {"category": cat, "filters": {"color": None, "position": None, "size": None}}
    out = {"count": "int", "exists": "yes_no", "describe": "text"}[op]
    return json.dumps({"op": op, "target": t if cat else None, "target2": None, "relation": None,
                       "attribute": None, "output_type": out})  # fmt: skip


class FakeLLM:
    """Responde según la pregunta del prompt; para el descriptor, un texto fijo."""

    def __init__(self, by_question: dict[str, list[str]], descriptions: list[str]) -> None:
        self.by_question = {k: iter(v) for k, v in by_question.items()}
        self.descriptions = iter(descriptions)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens=None) -> str:
        self.prompts.append(prompt)
        if "Resumen estructurado" in prompt:
            return next(self.descriptions)
        question = re.findall(r"Pregunta: (.+)\nJSON:", prompt)[-1]
        return next(self.by_question[question])


def make(cfg, llm):
    ex = executor_from_config(cfg, resolve, NAMES)
    interp = Interpreter(llm, category_matcher=resolve, category_names=NAMES, max_attempts=2)
    desc = SceneDescriber(llm, category_names=NAMES, max_new_tokens=50, max_objects=50,
                          max_relations=10, retry_on_unfaithful=1)  # fmt: skip
    return ex, interp, desc


def test_modo_llm_usa_el_mismo_ejecutor_y_maneja_fallas(cfg) -> None:
    scene = scene_from(BOXES, cfg)
    llm = FakeLLM(
        {
            BENCH[0]["question"]: [dsl("count", "tazas")],
            BENCH[1]["question"]: ["esto no es json", "tampoco"],  # 2 intentos fallidos
            # fuera del vocabulario: el intérprete reintenta una vez y después da unsupported
            BENCH[2]["question"]: [dsl("exists", "dragón")] * 2,
            BENCH[3]["question"]: [dsl("describe")],
        },
        descriptions=["Hay 5 elefantes.", "Hay 7 jirafas."],  # nunca fiel → plantilla
    )
    ex, interp, desc = make(cfg, llm)
    items = answer_questions_llm(BENCH, scene, interp, ex, desc, NAMES)
    by_id = {i["question_id"]: i for i in items}

    assert by_id["Q-COUNT-CUP"]["normalized_answer"] == 2
    assert by_id["Q-COUNT-CUP"]["interpreted_query"]["target"]["category"] == "cup"
    for qid in ("Q-COUNT-PLATE", "Q-EXISTS"):
        assert by_id[qid]["status"] == "unsupported"
        assert by_id[qid]["interpreted_query"]["op"] == "unsupported"  # DSL válido igual
        assert by_id[qid]["interpreter"]["status"] == "unsupported"
    d = by_id["Q-DESC"]
    assert d["interpreter"]["describer_status"] == "unfaithful"
    assert "plantilla" in d["status_detail"] and "elefante" not in d["answer"]
    # el LLM solo vio el texto de la pregunta: ni type ni class_en
    interp_prompts = [p for p in llm.prompts if "Resumen estructurado" not in p]
    assert not any("class_en" in p or '"existence"' in p for p in interp_prompts)

    scene["questions_llm"] = items
    assert validate_scene(scene, SCHEMA) == []


def test_descripcion_fiel_del_llm_con_evidencia(cfg) -> None:
    scene = scene_from(BOXES, cfg)
    llm = FakeLLM({BENCH[3]["question"]: [dsl("describe")]}, ["Hay 2 cup y 1 plate."])
    ex, interp, desc = make(cfg, llm)
    (item,) = answer_questions_llm([BENCH[3]], scene, interp, ex, desc, NAMES)
    assert item["answer"] == "Hay 2 cup y 1 plate." and item["status"] == "ok"
    assert sorted(item["evidence"]) == ["obj_0", "obj_1", "obj_2"]


def test_evaluacion_atribuye_percepcion_e_interpretacion(cfg) -> None:
    # Predicción: se perdió una taza (error de percepción).
    pred = scene_from([BOXES[0], BOXES[2]], cfg)
    gt = ImageGT(
        "img", "img.jpg", W, H,
        instances=[
            GTInstance(k, cid, tuple(map(float, b)),
                       [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]]], 400.0)
            for k, (cid, b) in enumerate(BOXES, 1)
        ],
    )  # fmt: skip
    oracle = oracle_scene(gt, np.zeros((H, W, 3), np.uint8), cfg, NAMES)
    assert [o["category_name"] for o in oracle["objects"]] == ["cup", "cup", "plate"]

    ex = executor_from_config(cfg, resolve, NAMES)
    phrases = cfg["qa"]["adapter"]["relation_phrases"]
    pred["questions"] = answer_questions(BENCH, pred, ex, phrases)
    llm = FakeLLM(
        {
            BENCH[0]["question"]: [dsl("count", "tazas")],
            BENCH[1]["question"]: [dsl("count", "reloj")],  # interpretó mal la categoría
            BENCH[2]["question"]: [dsl("exists", "reloj")],
            BENCH[3]["question"]: [dsl("describe")],
        },
        descriptions=["Hay 1 cup y 1 plate."],
    )
    _, interp, desc = make(cfg, llm)
    pred["questions_llm"] = answer_questions_llm(BENCH, pred, interp, ex, desc, NAMES)

    det = evaluate_mode(
        "deterministic", BENCH, {"img": pred}, {"img": oracle}, ex, phrases, resolve
    )
    llm_df = evaluate_mode("llm", BENCH, {"img": pred}, {"img": oracle}, ex, phrases, resolve)
    d = det.set_index("question_id")
    assert d.loc["Q-COUNT-CUP", "error_source"] == "perception"  # 1 vs 2 del oráculo
    assert bool(d.loc["Q-COUNT-PLATE", "correct"]) and bool(d.loc["Q-EXISTS", "correct"])
    l_ = llm_df.set_index("question_id")
    assert l_.loc["Q-COUNT-PLATE", "error_source"] == "interpretation"
    assert not l_.loc["Q-COUNT-PLATE", "dsl_match"]
    assert d["evidence_supports"].all() and d["reexecutes"].all()

    s = summarize(pd.concat([det, llm_df], ignore_index=True))
    assert s["llm"]["dsl_match_rate"] < 1 and s["llm"]["json_valid_rate"] == 1.0
    assert s["llm"]["descriptions"] == {
        "n": 1,
        "faithful_rate": 1.0,
        "template_fallback": 0,
        "not_interpreted_as_describe": 0,
    }
    assert s["deterministic"]["accuracy_vs_oracle"] == pytest.approx(2 / 3)
    assert s["deterministic"]["error_source"] == {"perception": 1}


def test_evidencia_que_no_sostiene_la_respuesta(cfg) -> None:
    scene = scene_from(BOXES, cfg)
    ex = executor_from_config(cfg, resolve, NAMES)
    (item,) = answer_questions([BENCH[0]], scene, ex, cfg["qa"]["adapter"]["relation_phrases"])
    assert evidence_supports(item, scene) == (True, "")
    bad = copy.deepcopy(item)
    bad["normalized_answer"] = 3
    assert not evidence_supports(bad, scene)[0]
    bad["evidence"] = ["obj_99"]
    assert evidence_supports(bad, scene) == (False, "evidencia con ids inexistentes")
