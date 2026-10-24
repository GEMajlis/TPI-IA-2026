import copy
import csv
import json

import pytest

from scene.scene.schema import (
    POSITIONS,
    QUESTION_TYPES,
    RELATION_PREDICATES,
    SIZES,
    SceneValidationError,
    assert_valid_scene,
    schema_errors,
    structural_errors,
    validate_scene,
)
from tests.conftest import ROOT

COMP = ROOT / "data" / "competition"
SCHEMA = COMP / "scene_schema.json"


def _sample_rows() -> list[dict]:
    with (COMP / "sample_submission.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def scene() -> dict:
    """SceneJSON mínimo válido con dos objetos, una relación y una respuesta."""
    obj = {
        "category_id": 1,
        "confidence": 0.9,
        "bbox": [0, 0, 1, 1],
        "mask": {"size": [4, 3], "counts": [0, 1, 11]},
        "centroid": [0, 0],
        "area_ratio": 1 / 12,
        "position": "top_left",
        "size": "small",
    }
    qa = {
        "question_id": "Q1",
        "type": "count",
        "question": "¿Cuántos?",
        "answer": "Hay 2.",
        "normalized_answer": 2,
        "evidence": ["o1", "o2"],
        "confidence": 0.8,
    }
    return {
        "image": {"id": "img", "width": 3, "height": 4},
        "objects": [{**obj, "id": "o1"}, {**obj, "id": "o2"}],
        "relations": [{"subject": "o1", "predicate": "overlaps", "object": "o2"}],
        "questions": [qa],
        "questions_llm": [{**qa, "interpreted_query": {"op": "count"}}],
    }


def test_enums_coinciden_con_el_schema() -> None:
    schema = json.loads(SCHEMA.read_text())
    obj = schema["properties"]["objects"]["items"]["properties"]
    rel = schema["properties"]["relations"]["items"]["properties"]
    qa = schema["$defs"]["questionAnswer"]["properties"]
    assert tuple(obj["position"]["enum"]) == POSITIONS
    assert tuple(obj["size"]["enum"]) == SIZES
    assert tuple(rel["predicate"]["enum"]) == RELATION_PREDICATES
    assert tuple(qa["type"]["enum"]) == QUESTION_TYPES


def test_las_80_filas_de_sample_submission_validan() -> None:
    rows = _sample_rows()
    assert len(rows) == 80
    for row in rows:
        scene = json.loads(row["SceneJSON"])
        assert scene["image"]["id"] == row["image_id"]
        assert validate_scene(scene, SCHEMA) == []


def test_scene_valida(scene) -> None:
    assert validate_scene(scene, SCHEMA) == []
    assert_valid_scene(scene, SCHEMA)


def test_imagen_sin_objetos_es_valida(scene) -> None:
    scene.update(objects=[], relations=[], questions=[], questions_llm=[])
    assert validate_scene(scene, SCHEMA) == []


@pytest.mark.parametrize(
    "path",
    [("objects", 0, "mask"), ("objects", 0, "position"), ("image", "width"), ("relations",)],
)
def test_falta_campo_obligatorio(scene, path) -> None:
    node = scene
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    assert schema_errors(scene, SCHEMA)
    with pytest.raises(SceneValidationError):
        assert_valid_scene(scene, SCHEMA)


@pytest.mark.parametrize(
    ("mutate", "where"),
    [
        (lambda s: s["objects"][0].update(position="arriba"), "objects/0/position"),
        (lambda s: s["objects"][0].update(size="huge"), "objects/0/size"),
        (lambda s: s["relations"][0].update(predicate="near"), "relations/0/predicate"),
        (lambda s: s["objects"][0].update(area_ratio=0), "objects/0/area_ratio"),
        (lambda s: s["objects"][0].update(confidence=1.5), "objects/0/confidence"),
        (lambda s: s["questions"][0].update(type="color"), "questions/0/type"),
        (lambda s: s["questions_llm"][0].pop("interpreted_query"), "questions_llm/0"),
    ],
)
def test_valores_invalidos(scene, mutate, where) -> None:
    mutate(scene)
    errors = schema_errors(scene, SCHEMA)
    assert any(e.startswith(where) for e in errors), errors


def test_campos_extra_permitidos(scene) -> None:
    scene["objects"][0]["color"] = {"value": "red", "source": "derived"}
    scene["extra_relations"] = [{"subject": "o1", "predicate": "near", "object": "o2"}]
    assert validate_scene(scene, SCHEMA) == []


def test_chequeos_estructurales(scene) -> None:
    bad = copy.deepcopy(scene)
    bad["objects"][1]["id"] = "o1"
    bad["objects"][0]["mask"]["size"] = [3, 4]
    bad["relations"].append({"subject": "o1", "predicate": "above", "object": "zz"})
    bad["questions"][0]["evidence"] = ["nope"]
    errors = structural_errors(bad)
    assert any("ids repetidos" in e for e in errors)
    assert any("mask/size" in e for e in errors)
    assert any("relations/1/object" in e for e in errors)
    assert any("evidence" in e for e in errors)
    # El chequeo estructural se puede desactivar.
    assert validate_scene(bad, SCHEMA, structural=False) == []
