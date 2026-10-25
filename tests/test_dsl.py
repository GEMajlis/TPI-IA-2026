import json

import pytest
from pydantic import ValidationError

from scene.qa.dsl import (
    ANSWER_FORMAT_TO_OUTPUT,
    QUESTION_TYPE_TO_OP,
    Query,
    QueryResult,
)
from scene.scene.schema import QUESTION_TYPES, validate_scene
from tests.conftest import ROOT

COMP = ROOT / "data" / "competition"
FIXTURE = ROOT / "tests" / "fixtures" / "benchmark_dsl.json"
# Frase del texto de la pregunta → predicado (para controlar el fixture escrito a mano).
RELATION_PHRASES = {
    "por encima de": "above",
    "por debajo de": "below",
    "a la derecha de": "right_of",
    "a la izquierda de": "left_of",
}


def _benchmark() -> list[dict]:
    with (COMP / "questions_dev.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fixture() -> dict[str, dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def test_mapeos_cubren_tipos_y_formatos_del_benchmark() -> None:
    assert set(QUESTION_TYPE_TO_OP) == set(QUESTION_TYPES)
    assert {q["answer_format"] for q in _benchmark()} <= set(ANSWER_FORMAT_TO_OUTPUT)


def test_fixture_cubre_todo_el_benchmark() -> None:
    assert set(_fixture()) == {q["question_id"] for q in _benchmark()}


@pytest.mark.parametrize("q", _benchmark(), ids=lambda q: q["question_id"])
def test_dsl_del_benchmark_valida_y_es_coherente(q: dict) -> None:
    query = Query.model_validate(_fixture()[q["question_id"]])
    assert query.op == QUESTION_TYPE_TO_OP[q["type"]]
    assert query.output_type == ANSWER_FORMAT_TO_OUTPUT[q["answer_format"]]
    # Orden de class_en = [sujeto, objeto].
    assert query.categories() == q["class_en"]
    if query.op == "relation":
        phrases = [p for p in RELATION_PHRASES if p in q["question"]]
        assert len(phrases) == 1
        assert query.relation == RELATION_PHRASES[phrases[0]]
    # Round-trip JSON (es lo que va en `interpreted_query`).
    assert Query.model_validate(query.model_dump(mode="json")) == query


def test_output_type_se_completa_desde_la_op() -> None:
    assert Query(op="count", target={"category": "cup"}).output_type == "int"
    assert Query(op="describe").output_type == "text"


@pytest.mark.parametrize(
    "bad",
    [
        {"op": "count"},  # sin target
        {"op": "relation", "target": {"category": "cup"}, "relation": "above"},  # sin target2
        {"op": "count", "target": {"category": "cup"}, "relation": "above"},
        {"op": "attribute", "target": {"category": "cup"}},  # sin attribute
        {"op": "count", "target": {"category": "cup"}, "output_type": "yes_no"},
        {"op": "describe", "target": {"category": "cup"}},
        {"op": "count", "target": {"category": ""}},
        {"op": "count", "target": {"category": "cup", "filters": {"position": "arriba"}}},
        {"op": "count", "target": {"category": "cup", "filters": {"size": "huge"}}},
        {"op": "count", "target": {"category": "cup"}, "extra": 1},  # campo desconocido
        {"op": "colour", "target": {"category": "cup"}},
        {
            "op": "relation",
            "target": {"category": "a"},
            "relation": "inside",
            "target2": {"category": "b"},
        },
    ],
)
def test_consultas_invalidas(bad: dict) -> None:
    with pytest.raises(ValidationError):
        Query.model_validate(bad)


def test_filtros_y_attribute_validos() -> None:
    q = Query(
        op="attribute",
        attribute="color",
        target={"category": "cup", "filters": {"position": "top_left", "size": "small"}},
    )
    assert q.target.filters.active() == {"position": "top_left", "size": "small"}
    assert Query(op="unsupported", reason="pregunta sobre la hora").reason


def test_json_schema_exportable() -> None:
    schema = Query.model_json_schema()
    assert "top_left" in json.dumps(schema)


@pytest.mark.parametrize("bad", [-1, "", True])
def test_resultado_normalized_answer_invalido(bad: object) -> None:
    with pytest.raises(ValidationError):
        QueryResult(answer="x", normalized_answer=bad, confidence=0.5)


def test_resultado_evidencia_repetida_y_confianza_fuera_de_rango() -> None:
    with pytest.raises(ValidationError):
        QueryResult(answer="x", normalized_answer=1, evidence=["a", "a"], confidence=0.5)
    with pytest.raises(ValidationError):
        QueryResult(answer="x", normalized_answer=1, confidence=1.5)
    with pytest.raises(ValidationError):
        QueryResult(answer="x", normalized_answer=1, confidence=0.5, status="maybe")


def test_resultado_valida_contra_el_schema() -> None:
    obj = {
        "id": "obj_1",
        "category_id": 1,
        "confidence": 0.9,
        "bbox": [0, 0, 1, 1],
        "mask": {"size": [4, 3], "counts": [0, 1, 11]},
        "centroid": [0, 0],
        "area_ratio": 1 / 12,
        "position": "top_left",
        "size": "small",
    }
    query = Query(op="count", target={"category": "sofa"})
    res = QueryResult(answer="Hay 1 sofá.", normalized_answer=1, evidence=["obj_1"], confidence=0.9)
    amb = QueryResult(
        answer="Hay varias instancias.",
        normalized_answer="center",
        confidence=0.3,
        status="ambiguous",
        detail="2 instancias para una pregunta singular",
    )
    scene = {
        "image": {"id": "tpi_0001", "width": 3, "height": 4},
        "objects": [obj],
        "relations": [],
        "questions": [
            res.to_question_answer("DEV-1-Q1", "count", "¿Cuántos?"),
            amb.to_question_answer("DEV-1-Q2", "position", "¿Dónde?"),
        ],
        "questions_llm": [res.to_question_answer("DEV-1-Q1", "count", "¿Cuántos?", query)],
    }
    assert validate_scene(scene, COMP / "scene_schema.json") == []
    assert scene["questions_llm"][0]["interpreted_query"]["op"] == "count"
    assert scene["questions"][1]["status"] == "ambiguous"
