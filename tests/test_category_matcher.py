import csv
import json

import numpy as np
import pytest

from scene.config import load_config
from scene.io.dataset import Category
from scene.qa.category_matcher import (
    CategoryMatcher,
    load_lexicon,
    matcher_from_config,
    normalize_text,
    singular_candidates,
)
from tests.conftest import BASE_CONFIG, ROOT

COMP = ROOT / "data" / "competition"
LEXICON = ROOT / "configs" / "lexicon_es.yaml"

CATS = [
    Category(1, "sofa"),
    Category(2, "cup"),
    Category(3, "knife"),
    Category(4, "tennis racket"),
    Category(5, "cell phone"),
    Category(6, "berry"),
    Category(7, "person"),
    Category(8, "clock"),
    Category(9, "watch"),
]
LEX = {
    "cup": ["taza"],
    "tennis racket": ["raqueta de tenis"],
    "cell phone": ["teléfono celular", "celular"],
    "clock": ["reloj"],
    "watch": ["reloj de pulsera"],
    "sofa": ["sofá"],
}
SYNS = {1: ["couch", "lounge"], 2: ["teacup"], 9: ["wristwatch"]}


def _competition_categories() -> list[Category]:
    with (COMP / "categories.csv").open(encoding="utf-8", newline="") as f:
        return [Category(int(r["category_id"]), r["category_name"]) for r in csv.DictReader(f)]


@pytest.fixture
def matcher() -> CategoryMatcher:
    return CategoryMatcher(CATS, lexicon=LEX, synonyms=SYNS)


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("  Teléfono  CELULAR ", "telefono celular"),
        ('"raqueta de tenis"', "raqueta de tenis"),
        ("la taza", "taza"),
        ("The cup", "cup"),
        ("niño", "nino"),
        ("t-shirt", "t shirt"),
        ("a", "a"),  # un artículo solo no se borra
    ],
)
def test_normalize_text(raw: str, norm: str) -> None:
    assert normalize_text(raw) == norm


def test_singular_candidates_empieza_por_el_texto_tal_cual() -> None:
    cands = singular_candidates("raquetas de tenis")
    assert cands[0] == "raquetas de tenis"
    assert "raqueta de tenis" in cands
    assert singular_candidates("") == []


@pytest.mark.parametrize(
    ("text", "name", "method"),
    [
        ("cup", "cup", "name"),
        ("cups", "cup", "name"),
        ("knives", "knife", "name"),
        ("berries", "berry", "name"),
        ("people", "person", "name"),
        ("The Sofas", "sofa", "name"),
        ("taza", "cup", "lexicon_es"),
        ("tazas", "cup", "lexicon_es"),
        ("raquetas de tenis", "tennis racket", "lexicon_es"),
        ("teléfonos celulares", "cell phone", "lexicon_es"),
        ("celulares", "cell phone", "lexicon_es"),
        ("sofá", "sofa", "name"),  # sin acento coincide con el nombre en inglés
        ("relojes", "clock", "lexicon_es"),
        ("reloj de pulsera", "watch", "lexicon_es"),
        ("couch", "sofa", "lvis_synonym"),
        ("couches", "sofa", "lvis_synonym"),
        ("wristwatches", "watch", "lvis_synonym"),
    ],
)
def test_match_plurales_sinonimos_y_espanol(
    matcher: CategoryMatcher, text: str, name: str, method: str
) -> None:
    found = matcher.match(text)
    assert found is not None
    assert (found.category_name, found.method, found.score) == (name, method, 1.0)
    assert matcher.match_id(text) == found.category_id


@pytest.mark.parametrize("text", ["dinosaurio", "piano", "elephant", "", "  ", "¿?"])
def test_fuera_de_vocabulario_devuelve_none(matcher: CategoryMatcher, text: str) -> None:
    assert matcher.match(text) is None
    assert matcher.match_id(text) is None


def test_termino_ambiguo_se_descarta() -> None:
    lex = {"cup": ["vaso"], "knife": ["vaso"], "sofa": ["sofá"]}
    m = CategoryMatcher(CATS, lexicon=lex)
    assert m.match("vaso") is None
    assert m.dropped == {"lexicon_es:vaso": [2, 3]}


def test_nombre_tiene_prioridad_sobre_sinonimo() -> None:
    # "cup" es sinónimo de sofa pero nombre de la categoría 2: gana el nombre.
    m = CategoryMatcher(CATS, synonyms={1: ["cup"]})
    assert m.match("cup").category_name == "cup"


def test_lexico_con_categoria_desconocida_falla() -> None:
    with pytest.raises(ValueError, match="no están en el vocabulario"):
        CategoryMatcher(CATS, lexicon={"unicorn": ["unicornio"]})


class FakeEmbedder:
    """Embeddings de juguete: textos conocidos → vectores fijos; el resto → ortogonal."""

    def __init__(self) -> None:
        self.calls = 0
        self.vecs = {"mug": [1.0, 0.1, 0.0], "cup": [1.0, 0.0, 0.0], "taza": [0.9, 0.0, 0.1]}

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rows = np.array([self.vecs.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype=np.float32)
        rows[np.all(rows == [0.0, 0.0, 1.0], axis=1)] = [0.0, 1.0, 0.0]
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_embeddings_como_ultimo_recurso() -> None:
    emb = FakeEmbedder()
    m = CategoryMatcher(CATS, lexicon=LEX, embedder=emb, embedding_threshold=0.9)
    assert m.match("cup").method == "name"
    assert emb.calls == 0  # con match exacto no se usan embeddings
    found = m.match("mug")
    assert (found.category_name, found.method) == ("cup", "embedding")
    assert 0.9 <= found.score <= 1.0
    # Debajo del umbral → None.
    strict = CategoryMatcher(CATS, lexicon=LEX, embedder=FakeEmbedder(), embedding_threshold=0.999)
    assert strict.match("mug") is None


def test_lexico_real_cubre_el_vocabulario_sin_colisiones() -> None:
    cats = _competition_categories()
    lexicon = load_lexicon(LEXICON)
    assert set(lexicon) == {c.name for c in cats}
    m = CategoryMatcher(cats, lexicon=lexicon)
    assert m.dropped == {}


def test_lexico_real_resuelve_las_clases_del_benchmark() -> None:
    m = CategoryMatcher(_competition_categories(), lexicon=load_lexicon(LEXICON))
    with (COMP / "questions_dev.jsonl").open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    pairs = {
        (es, en) for q in questions for es, en in zip(q["class_es"], q["class_en"], strict=True)
    }
    for es, en in sorted(pairs):
        found = m.match(es)
        assert found is not None and found.category_name == en, (es, en, found)
        assert m.match(en).category_name == en


def test_matcher_from_config_sin_red() -> None:
    cfg = load_config(
        BASE_CONFIG,
        dotlist=[
            f"qa.matcher.lexicon_es={LEXICON}",
            "qa.matcher.lvis_synonyms=false",
            "qa.matcher.embeddings.enabled=false",
        ],
    )
    m = matcher_from_config(cfg, _competition_categories())
    assert m.embedder is None
    assert m.match("jarrones").category_name == "vase"
    assert m.match("dinosaurio") is None
