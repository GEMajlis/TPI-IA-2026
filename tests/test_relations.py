import numpy as np

from scene.scene.properties import properties_from_mask
from scene.scene.relations import RelObject, box_gap, compute_relations, directional
from scene.scene.schema import RELATION_PREDICATES

H, W = 100, 100
CFG = {
    "h_margin": 0.02,
    "v_margin": 0.02,
    "overlap_min_iou": 0.01,
    "near_max_gap": 0.05,
    "on_top_min_x_overlap": 0.3,
    "contact_px": 3,
    "max_relations": 20000,
}


def obj(oid: str, y1: int, y2: int, x1: int, x2: int, conf: float = 0.9) -> RelObject:
    mask = np.zeros((H, W), bool)
    mask[y1:y2, x1:x2] = True
    p = properties_from_mask(mask, 0.00333, 0.03)
    return RelObject(oid, p.centroid, p.mask_bbox, mask, conf)


def triples(rels) -> set[tuple[str, str, str]]:
    return {(r.subject, r.predicate, r.object) for r in rels}


def test_izquierda_derecha_y_simetria() -> None:
    a, b = obj("a", 40, 50, 0, 10), obj("b", 40, 50, 80, 90)
    res = compute_relations([a, b], W, H, CFG)
    assert triples(res.relations) == {("a", "left_of", "b"), ("b", "right_of", "a")}


def test_arriba_abajo() -> None:
    a, b = obj("a", 0, 10, 40, 50), obj("b", 80, 90, 40, 50)
    assert triples(compute_relations([a, b], W, H, CFG).relations) == {
        ("a", "above", "b"),
        ("b", "below", "a"),
    }


def test_diagonal_da_dos_relaciones_por_sentido() -> None:
    a, b = obj("a", 0, 10, 0, 10), obj("b", 80, 90, 80, 90)
    rels = triples(compute_relations([a, b], W, H, CFG).relations)
    assert {("a", "left_of", "b"), ("a", "above", "b"), ("b", "right_of", "a")} <= rels
    assert len(rels) == 4


def test_centroides_alineados_no_dan_relacion_horizontal() -> None:
    a, b = obj("a", 0, 10, 40, 50), obj("b", 80, 90, 41, 51)  # dx = 1 px < margen de 2 px
    assert directional(a, b, W, H, 0.02, 0.02) == ["above"]


def test_overlaps_con_mascaras_que_se_solapan() -> None:
    a, b = obj("a", 40, 60, 40, 60), obj("b", 50, 70, 50, 70)
    rels = triples(compute_relations([a, b], W, H, CFG).relations)
    assert ("a", "overlaps", "b") in rels and ("b", "overlaps", "a") in rels


def test_cajas_que_se_tocan_sin_solape_no_es_overlaps() -> None:
    a, b = obj("a", 40, 60, 10, 30), obj("b", 40, 60, 30, 50)
    rels = triples(compute_relations([a, b], W, H, CFG).relations)
    assert not any(p == "overlaps" for _, p, _ in rels)


def test_near_y_on_top_of_van_en_extra() -> None:
    cup = obj("cup", 30, 50, 40, 50)  # apoyada sobre la mesa
    table = obj("table", 50, 80, 20, 80)
    far = obj("far", 0, 5, 95, 100)
    res = compute_relations([cup, table, far], W, H, CFG)
    extra = triples(res.extra_relations)
    assert ("cup", "on_top_of", "table") in extra
    assert ("table", "on_top_of", "cup") not in extra
    assert ("cup", "near", "table") in extra and ("table", "near", "cup") in extra
    assert not any("far" in (s, o) and p == "near" for s, p, o in extra)
    assert {r.predicate for r in res.relations} <= set(RELATION_PREDICATES)


def test_on_top_of_exige_contacto() -> None:
    cup = obj("cup", 20, 40, 40, 50)  # 10 px por encima de la mesa
    table = obj("table", 50, 80, 20, 80)
    extra = triples(compute_relations([cup, table], W, H, CFG).extra_relations)
    assert ("cup", "on_top_of", "table") not in extra


def test_tope_prioriza_confianza_y_no_rompe_pares() -> None:
    objs = [obj(f"o{i}", 10 * i, 10 * i + 5, 10 * i, 10 * i + 5, conf=1 - i / 10) for i in range(5)]
    res = compute_relations(objs, W, H, {**CFG, "max_relations": 9})
    # cada par diagonal aporta 4 relaciones: entran 2 pares completos (los de mayor confianza)
    assert len(res.relations) == 8
    assert res.stats["pairs_truncated"] == 8
    assert {r.subject for r in res.relations} | {r.object for r in res.relations} == {
        "o0",
        "o1",
        "o2",
    }


def test_box_gap() -> None:
    assert box_gap((0, 0, 10, 10), (5, 5, 15, 15)) == 0
    assert box_gap((0, 0, 10, 10), (13, 14, 20, 20)) == 5.0


def test_sin_objetos() -> None:
    res = compute_relations([], W, H, CFG)
    assert res.relations == [] and res.extra_relations == []
