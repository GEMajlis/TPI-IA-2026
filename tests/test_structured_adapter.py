import csv
import json

import pytest

from scene.config import load_config
from scene.io.dataset import Category
from scene.qa.category_matcher import CategoryMatcher, load_lexicon, spanish_names
from scene.qa.dsl import Query
from scene.qa.executor import executor_from_config
from scene.qa.structured_adapter import (
    answer_questions,
    benchmark_image_id,
    group_by_image,
    query_from_benchmark,
    relation_from_text,
)
from scene.scene.schema import validate_scene
from tests.conftest import BASE_CONFIG, ROOT

COMP = ROOT / "data" / "competition"
FIXTURE = ROOT / "tests" / "fixtures" / "benchmark_dsl.json"
CFG = load_config(BASE_CONFIG)
PHRASES = CFG["qa"]["adapter"]["relation_phrases"]
STATUSES = {"ok", "unsupported", "insufficient_information", "ambiguous"}


def _benchmark() -> list[dict]:
    with (COMP / "questions_dev.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _categories() -> list[Category]:
    with (COMP / "categories.csv").open(encoding="utf-8", newline="") as f:
        return [Category(int(r["category_id"]), r["category_name"]) for r in csv.DictReader(f)]


@pytest.mark.parametrize("q", _benchmark(), ids=lambda q: q["question_id"])
def test_adapter_coincide_con_el_dsl_escrito_a_mano(q: dict) -> None:
    expected = Query.model_validate(json.loads(FIXTURE.read_text())[q["question_id"]])
    assert query_from_benchmark(q, PHRASES) == expected


@pytest.mark.parametrize(
    ("text", "rel"),
    [
        ('¿La instancia de "reloj" está por encima de la instancia de "anillo"?', "above"),
        ("¿El vaso está ENCIMA DE la mesa?", "above"),
        ("¿La caja está por debajo de la pancarta?", "below"),
        ("¿Está a la izquierda de la cama?", "left_of"),
        ("¿Está a la derecha de la cama?", "right_of"),
        ("¿El gato está sobre la cama?", "on_top_of"),
        ("¿El gato está cerca de la cama?", "near"),
        ("¿De qué color es el gato?", None),
        ("¿El sobretodo está en la cama?", None),  # "sobre" solo como palabra entera
    ],
)
def test_relation_from_text(text: str, rel: str | None) -> None:
    assert relation_from_text(text, PHRASES) == rel


def test_consignas_no_traducibles_quedan_unsupported() -> None:
    bad_rel = {"type": "relation", "question": "¿Cuál es más grande?", "class_en": ["a", "b"]}
    assert query_from_benchmark(bad_rel, PHRASES).op == "unsupported"
    bad_type = {"type": "color", "question": "?", "class_en": ["cup"]}
    assert query_from_benchmark(bad_type, PHRASES).op == "unsupported"
    bad_count = {"type": "count", "question": "?", "class_en": []}
    assert query_from_benchmark(bad_count, PHRASES).op == "unsupported"


def test_group_by_image() -> None:
    groups = group_by_image(_benchmark())
    assert len(groups) == 20 and all(len(qs) == 3 for qs in groups.values())
    assert benchmark_image_id({"image_file": "tpi_0001.jpg"}) == "tpi_0001"


def _obj(oid: str, cid: int, conf: float) -> dict:
    return {
        "id": oid,
        "category_id": cid,
        "confidence": conf,
        "bbox": [0, 0, 1, 1],
        "mask": {"size": [4, 3], "counts": [0, 1, 11]},
        "centroid": [0, 0],
        "area_ratio": 1 / 12,
        "position": "center",
        "size": "small",
        "attributes": {"color": "red", "color_confidence": 0.5},
    }


@pytest.mark.parametrize("variant", ["vacia", "con_objetos"])
def test_las_60_consignas_tienen_respuesta_o_estado(variant: str) -> None:
    cats = _categories()
    by_name = {c.name: c.id for c in cats}
    lexicon = load_lexicon(ROOT / "configs" / "lexicon_es.yaml")
    matcher = CategoryMatcher(cats, lexicon=lexicon)
    ex = executor_from_config(
        CFG, matcher.match_id, {c.id: c.name for c in cats}, spanish_names(cats, lexicon)
    )
    for iid, questions in group_by_image(_benchmark()).items():
        objects = []
        if variant == "con_objetos":
            # Una instancia de cada clase mencionada (y dos de la primera, para `ambiguous`).
            names = [n for q in questions for n in q["class_en"]]
            objects = [_obj(f"o{i}", by_name[n], 0.5 + 0.01 * i) for i, n in enumerate(names)]
            if names:
                objects.append(_obj("dup", by_name[names[0]], 0.3))
        scene = {
            "image": {"id": iid, "width": 3, "height": 4},
            "objects": objects,
            "relations": [],
        }
        scene["questions"] = answer_questions(questions, scene, ex, PHRASES)
        assert validate_scene(scene, COMP / "scene_schema.json") == []
        for item, q in zip(scene["questions"], questions, strict=True):
            assert (item["question_id"], item["type"], item["question"]) == (
                q["question_id"],
                q["type"],
                q["question"],
            )
            assert item["status"] in STATUSES and item["answer"]
            assert item["status"] != "unsupported"  # todo el benchmark está en vocabulario
            if variant == "vacia" and q["type"] in ("position", "size", "relation"):
                assert item["status"] == "insufficient_information"
