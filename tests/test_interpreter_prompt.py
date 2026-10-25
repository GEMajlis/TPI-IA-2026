"""El prompt del intérprete: ejemplos válidos y sin contenido del benchmark (D-18)."""

import json
import re
from importlib import resources

import pytest

from scene.config import load_config
from scene.qa.category_matcher import normalize_text
from scene.qa.dsl import Query
from tests.conftest import BASE_CONFIG, ROOT

QUESTIONS = ROOT / "data" / "competition" / "questions_dev.jsonl"


def prompt_text() -> str:
    version = load_config(BASE_CONFIG)["llm"]["interpreter"]["prompt_version"]
    path = resources.files("scene.llm").joinpath("prompts", version)
    return path.read_text(encoding="utf-8")


def examples() -> list[tuple[str, dict]]:
    text = prompt_text().split("Ejemplos:", 1)[1]
    pairs = re.findall(r"Pregunta: (.+)\nJSON: (\{.+\})", text)
    return [(q, json.loads(js)) for q, js in pairs]


def benchmark() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_config_usa_el_prompt_v2() -> None:
    assert load_config(BASE_CONFIG)["llm"]["interpreter"]["prompt_version"] == "interpreter_v2.txt"


@pytest.mark.parametrize(("question", "payload"), examples())
def test_cada_ejemplo_es_un_dsl_valido(question: str, payload: dict) -> None:
    if "unsupported_reason" in payload:
        return
    Query.model_validate(payload)


def test_hay_ejemplos_de_cada_op() -> None:
    ops = {p.get("op", "unsupported") for _, p in examples()}
    assert {"count", "exists", "position", "size", "relation", "attribute", "describe"} <= ops
    assert "unsupported" in ops


def test_sin_categorias_ni_preguntas_del_benchmark() -> None:
    bench = benchmark()
    names = {normalize_text(c) for q in bench for c in q["class_en"] + q["class_es"]}
    used = set()
    for _, p in examples():
        for key in ("target", "target2"):
            if p.get(key):
                used.add(normalize_text(p[key]["category"]))
    # Se comparan también los singulares de los plurales simples ("tazas" → "taza").
    used |= {u[:-1] for u in used if u.endswith("s")}
    assert not used & names, f"categorías del benchmark en el prompt: {used & names}"
    asked = {normalize_text(q) for q, _ in examples()}
    assert not asked & {normalize_text(q["question"]) for q in bench}
