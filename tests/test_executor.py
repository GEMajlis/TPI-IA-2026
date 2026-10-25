import pytest

from scene.config import load_config
from scene.io.dataset import Category
from scene.qa.category_matcher import CategoryMatcher
from scene.qa.dsl import Query
from scene.qa.executor import Executor, executor_from_config
from scene.scene.schema import validate_scene
from tests.conftest import BASE_CONFIG, ROOT

SCHEMA = ROOT / "data" / "competition" / "scene_schema.json"
CATS = [Category(1, "cup"), Category(2, "clock"), Category(3, "ring"), Category(4, "sofa")]
NAMES = {c.id: c.name for c in CATS}
PALETTE = ("red", "blue", "white")


def _obj(oid: str, cid: int, conf: float, position="center", size="small", color="red") -> dict:
    """Objeto mínimo válido (un píxel en una imagen de 4×3)."""
    return {
        "id": oid,
        "category_id": cid,
        "category_name": NAMES[cid],
        "confidence": conf,
        "bbox": [0, 0, 1, 1],
        "mask": {"size": [4, 3], "counts": [0, 1, 11]},
        "centroid": [0, 0],
        "area_ratio": 1 / 12,
        "position": position,
        "size": size,
        "attributes": {"color": color, "color_confidence": 0.5 if color else None},
    }


def _scene(objects: list[dict], relations=(), extra=()) -> dict:
    return {
        "image": {"id": "tpi_0001", "width": 3, "height": 4},
        "objects": objects,
        "relations": [
            dict(zip(("subject", "predicate", "object"), r, strict=True)) for r in relations
        ],
        "extra_relations": [
            dict(zip(("subject", "predicate", "object"), r, strict=True)) for r in extra
        ],
    }


@pytest.fixture
def ex() -> Executor:
    matcher = CategoryMatcher(CATS, lexicon={"cup": ["taza"], "clock": ["reloj"]})
    return Executor(
        resolve_category=matcher.match_id,
        category_names=NAMES,
        palette=PALETTE,
        absent_confidence=0.9,
        color_aliases={"rojo": "red"},
        display_names={1: "taza", 2: "reloj"},
    )


@pytest.fixture
def scene() -> dict:
    return _scene(
        [
            _obj("o1", 1, 0.9, "top_left", "small", "red"),
            _obj("o2", 1, 0.6, "bottom_right", "large", "blue"),
            _obj("o3", 2, 0.8, "top_center", "medium", None),
            _obj("o4", 3, 0.5, "bottom_center", "small", "white"),
        ],
        relations=[("o3", "above", "o4"), ("o4", "below", "o3")],
        extra=[("o3", "near", "o4"), ("o4", "near", "o3")],
    )


def q(**kw) -> Query:
    return Query.model_validate(kw)


def T(cat: str, **filters) -> dict:
    return {"category": cat, "filters": filters}


def _check(res, scene: dict) -> None:
    """Invariantes de todo resultado: evidencia existente y sin repetir, texto no vacío."""
    ids = {o["id"] for o in scene["objects"]}
    assert set(res.evidence) <= ids
    assert res.answer and 0.0 <= res.confidence <= 1.0


def test_count(ex, scene) -> None:
    res = ex.execute(q(op="count", target=T("tazas")), scene)
    _check(res, scene)
    assert (res.normalized_answer, res.status) == (2, "ok")
    assert res.evidence == ["o1", "o2"]
    assert res.confidence == pytest.approx(0.75)  # promedio de las contadas
    assert "2 instancias" in res.answer


def test_count_con_filtros(ex, scene) -> None:
    assert ex.execute(q(op="count", target=T("cup", color="rojo")), scene).normalized_answer == 1
    assert ex.execute(q(op="count", target=T("cup", size="large")), scene).evidence == ["o2"]
    res = ex.execute(q(op="count", target=T("cup", position="center")), scene)
    # Ninguna cumple el filtro: confianza = 1 − máx. de las descartadas.
    assert (res.normalized_answer, res.evidence) == (0, [])
    assert res.confidence == pytest.approx(0.1)


def test_exists(ex, scene) -> None:
    yes = ex.execute(q(op="exists", target=T("reloj")), scene)
    assert (yes.normalized_answer, yes.evidence, yes.confidence) == ("yes", ["o3"], 0.8)
    no = ex.execute(q(op="exists", target=T("sofa")), scene)
    _check(no, scene)
    # Ninguna instancia de la categoría: 1 − umbral del detector.
    assert (no.normalized_answer, no.evidence, no.confidence) == ("no", [], 0.9)
    assert no.status == "ok" and no.answer == "No."


def test_position_y_size(ex, scene) -> None:
    pos = ex.execute(q(op="position", target=T("clock")), scene)
    assert (pos.normalized_answer, pos.status, pos.evidence) == ("top_center", "ok", ["o3"])
    size = ex.execute(q(op="size", target=T("ring")), scene)
    assert (size.normalized_answer, size.confidence) == ("small", 0.5)


def test_ambiguous_responde_con_la_de_mayor_confianza(ex, scene) -> None:
    res = ex.execute(q(op="position", target=T("cup")), scene)
    _check(res, scene)
    assert res.status == "ambiguous"
    assert res.normalized_answer == "top_left"  # o1, la de mayor confianza
    assert res.evidence == ["o1", "o2"]  # todas, la elegida primero
    assert res.confidence == pytest.approx(0.9 * 0.9 / 1.5)
    assert "2 instancias" in res.detail and "varias instancias" in res.answer


def test_insufficient_information(ex, scene) -> None:
    res = ex.execute(q(op="size", target=T("sofa")), scene)
    _check(res, scene)
    assert (res.status, res.normalized_answer, res.evidence) == (
        "insufficient_information",
        "insufficient_information",
        [],
    )
    assert res.confidence == 0.0
    rel = q(op="relation", target=T("clock"), relation="above", target2=T("sofa"))
    assert ex.execute(rel, scene).status == "insufficient_information"
    # Color desconocido en el objeto (null en el SceneJSON).
    attr = ex.execute(q(op="attribute", attribute="color", target=T("clock")), scene)
    assert attr.status == "insufficient_information"


@pytest.mark.parametrize(
    "query",
    [
        {"op": "count", "target": {"category": "dinosaurio"}},
        {"op": "exists", "target": {"category": "cup", "filters": {"color": "fucsia"}}},
        {"op": "unsupported", "reason": "pregunta sobre la hora"},
    ],
)
def test_unsupported(ex, scene, query: dict) -> None:
    res = ex.execute(Query.model_validate(query), scene)
    _check(res, scene)
    assert (res.status, res.normalized_answer, res.confidence) == (
        "unsupported",
        "unsupported",
        0.0,
    )
    assert res.detail and res.answer == "La pregunta no está soportada."


def test_relation(ex, scene) -> None:
    yes = ex.execute(
        q(op="relation", target=T("clock"), relation="above", target2=T("ring")), scene
    )
    _check(yes, scene)
    assert (yes.normalized_answer, yes.evidence) == ("yes", ["o3", "o4"])
    assert yes.confidence == pytest.approx(0.8 * 0.5)  # producto
    no = ex.execute(q(op="relation", target=T("ring"), relation="above", target2=T("clock")), scene)
    assert (no.normalized_answer, no.status) == ("no", "ok")
    # near / on_top_of se buscan en extra_relations (D-04).
    near = ex.execute(
        q(op="relation", target=T("clock"), relation="near", target2=T("ring")), scene
    )
    assert near.normalized_answer == "yes"


def test_relation_ambigua_y_misma_categoria(ex) -> None:
    scene = _scene(
        [_obj("a", 1, 0.9), _obj("b", 1, 0.7), _obj("c", 2, 0.8)],
        relations=[("a", "left_of", "b"), ("b", "right_of", "a")],
    )
    res = ex.execute(q(op="relation", target=T("cup"), relation="left_of", target2=T("cup")), scene)
    # El objeto de la relación no puede ser el mismo que el sujeto.
    assert (res.normalized_answer, res.evidence[:2]) == ("yes", ["a", "b"])
    amb = ex.execute(q(op="relation", target=T("cup"), relation="above", target2=T("clock")), scene)
    assert amb.status == "ambiguous" and set(amb.evidence) == {"a", "b", "c"}
    alone = _scene([_obj("a", 1, 0.9)])
    rel = q(op="relation", target=T("cup"), relation="left_of", target2=T("cup"))
    assert ex.execute(rel, alone).status == "insufficient_information"


def test_attribute_color(ex, scene) -> None:
    res = ex.execute(q(op="attribute", attribute="color", target=T("ring")), scene)
    assert (res.normalized_answer, res.evidence) == ("white", ["o4"])
    assert res.confidence == pytest.approx(0.5 * 0.5)


def test_describe(ex, scene) -> None:
    res = ex.execute(q(op="describe"), scene)
    _check(res, scene)
    assert res.status == "ok" and res.answer == res.normalized_answer
    assert "4 objetos de 3 categorías" in res.answer
    assert "taza (2)" in res.answer and "arriba a la izquierda" in res.answer
    empty = ex.execute(q(op="describe"), _scene([]))
    assert empty.normalized_answer == "No se detectaron objetos en la imagen."
    assert empty.evidence == []


def test_resultados_validan_en_el_scenejson(ex, scene) -> None:
    queries = [
        q(op="count", target=T("cup")),
        q(op="position", target=T("cup")),
        q(op="size", target=T("sofa")),
        q(op="count", target=T("dinosaurio")),
        q(op="describe"),
    ]
    scene["questions"] = [
        ex.execute(qq, scene).to_question_answer(f"Q{i}", "count", "¿?")
        for i, qq in enumerate(queries)
    ]
    assert validate_scene(scene, SCHEMA) == []


def test_executor_from_config() -> None:
    cfg = load_config(BASE_CONFIG)
    ex = executor_from_config(cfg, lambda t: 1 if t == "cup" else None, NAMES)
    assert ex.absent_confidence == pytest.approx(1 - cfg["detection"]["score_threshold"])
    assert "red" in ex.palette and ex.color_aliases["rojo"] == "red"
