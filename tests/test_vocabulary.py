import json
from pathlib import Path

import pytest

from scene.detection.vocabulary import (
    build_vocabulary,
    clean_name,
    load_lvis_synonyms,
    match_lvis_synonyms,
    parse_lvis_categories,
    vocabulary_from_config,
)
from scene.io.dataset import Category, load_vocabulary
from tests.conftest import ROOT, requires_data

LVIS_CACHE = ROOT / "data" / "cache" / "lvis_synonyms.json"


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("trash_can", "trash can"),
        ("glass_(drink_container)", "glass"),
        ("mouse_(computer_equipment)", "mouse"),
        ("  Traffic  Light ", "traffic light"),
        ("sofa", "sofa"),
    ],
)
def test_clean_name(raw: str, clean: str) -> None:
    assert clean_name(raw) == clean


CATS = [Category(1, "sofa"), Category(2, "cup"), Category(3, "mug"), Category(4, "drinking glass")]
SYNS = {1: ["sofa", "couch", "lounge"], 2: ["cup", "mug"], 3: ["mug"], 4: ["glass", "cup"]}


def test_modo_clean_una_query_por_categoria() -> None:
    vocab = build_vocabulary(CATS, mode="clean", synonyms=SYNS, template="a photo of a {}")
    assert [q.text for q in vocab.queries] == ["sofa", "cup", "mug", "drinking glass"]
    assert vocab.prompts()[0] == "a photo of a sofa"
    assert vocab.query_to_category() == [1, 2, 3, 4]


def test_modo_synonyms_descarta_colisiones() -> None:
    vocab = build_vocabulary(CATS, mode="synonyms", synonyms=SYNS)
    by_cat = {cid: [q.text for q in qs] for cid, qs in vocab.by_category().items()}
    # "mug" es nombre de la categoría 3 y sinónimo de la 2; "cup" es nombre de 2 y sinónimo de 4.
    assert by_cat == {
        1: ["sofa", "couch", "lounge"],
        2: ["cup"],
        3: ["mug"],
        4: ["drinking glass", "glass"],
    }
    assert set(vocab.dropped) == {"cup", "mug"}
    # Ningún texto de query mapea a dos categorías.
    texts = [q.text for q in vocab.queries]
    assert len(texts) == len(set(texts))


def test_max_synonyms() -> None:
    vocab = build_vocabulary(CATS[:1], mode="synonyms", synonyms=SYNS, max_synonyms=1)
    assert [q.text for q in vocab.queries] == ["sofa", "couch"]


def test_modo_invalido() -> None:
    with pytest.raises(ValueError):
        build_vocabulary(CATS, mode="otro")


def test_parse_y_match_lvis() -> None:
    source = (
        "# comentario\nLVIS_CATEGORIES = ["
        "{'name': 'laptop_computer', 'id': 1,"
        " 'synonyms': ['laptop_computer', 'notebook_computer']},"
        "{'name': 'glass_(drink_container)', 'synonyms': ['drinking_glass'], 'id': 2},"
        "{'name': 'sofa', 'synonyms': ['sofa', 'couch'], 'id': 3}]  # noqa\n"
    )
    lvis = parse_lvis_categories(source)
    cats = [Category(1, "laptop"), Category(2, "drinking glass"), Category(3, "sofa")]
    syns = match_lvis_synonyms(cats, lvis, aliases={"laptop": "laptop_computer"})
    assert syns == {
        1: ["laptop_computer", "notebook_computer"],
        2: ["drinking_glass"],
        3: ["sofa", "couch"],
    }
    with pytest.raises(ValueError, match="laptop"):
        match_lvis_synonyms(cats, lvis)  # sin alias, laptop no aparece


def test_cache_de_sinonimos(tmp_path: Path) -> None:
    cache = tmp_path / "syn.json"
    cache.write_text(json.dumps({"synonyms": {"1": ["sofa", "couch"]}}))
    cfg = {"synonyms_cache": str(cache), "lvis_categories_url": "http://no-se-usa"}
    assert load_lvis_synonyms(cfg, [Category(1, "sofa")]) == {1: ["sofa", "couch"]}


@requires_data
def test_120_categorias_sin_colisiones(base_cfg) -> None:
    cats = load_vocabulary(base_cfg)
    clean = vocabulary_from_config(base_cfg, cats)
    assert len(clean.queries) == 120 and not clean.dropped
    if not LVIS_CACHE.is_file():
        pytest.skip("sin caché local de sinónimos LVIS")
    base_cfg["vocab"].update(mode="synonyms", synonyms_cache=str(LVIS_CACHE))
    vocab = vocabulary_from_config(base_cfg, cats)
    assert {q.category_id for q in vocab.queries} == {c.id for c in cats}
    texts = [q.text for q in vocab.queries]
    assert len(texts) == len(set(texts))
