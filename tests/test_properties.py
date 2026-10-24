import numpy as np
import pytest

from scene.masks import rle
from scene.scene.properties import (
    box_to_mask,
    ensure_nonempty,
    position_3x3,
    properties_from_config,
    properties_from_mask,
    properties_from_rle,
    size_label,
)
from scene.scene.schema import POSITIONS

H, W = 90, 120  # celdas de 40×30 px
SMALL, MEDIUM = 0.00333, 0.03


@pytest.mark.parametrize("cell", range(9))
def test_cada_una_de_las_9_regiones(cell: int) -> None:
    row, col = divmod(cell, 3)
    mask = np.zeros((H, W), dtype=bool)
    mask[row * 30 + 10 : row * 30 + 20, col * 40 + 15 : col * 40 + 25] = True
    assert properties_from_mask(mask, SMALL, MEDIUM).position == POSITIONS[cell]


def test_bordes_de_la_grilla_van_a_la_derecha_y_abajo() -> None:
    assert position_3x3(39.999, 29.999, W, H) == "top_left"
    assert position_3x3(40.0, 30.0, W, H) == "center"
    assert position_3x3(80.0, 60.0, W, H) == "bottom_right"
    assert position_3x3(W - 0.5, H - 0.5, W, H) == "bottom_right"
    assert position_3x3(0, 0, W, H) == "top_left"


def test_pixel_en_el_origen_como_el_ejemplo_del_schema() -> None:
    mask = np.zeros((H, W), dtype=bool)
    mask[0, 0] = True
    p = properties_from_mask(mask, SMALL, MEDIUM)
    assert p.centroid == (0.0, 0.0)
    assert p.area == 1 and p.area_ratio == pytest.approx(1 / (H * W))
    assert p.mask_bbox == (0, 0, 1, 1)
    assert p.position == "top_left" and p.size == "small"


def test_centroide_area_y_caja_de_un_rectangulo() -> None:
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 30:50] = True  # filas 10..19, columnas 30..49
    p = properties_from_mask(mask, SMALL, MEDIUM)
    assert p.area == 200
    assert p.centroid == (39.5, 14.5)
    assert p.mask_bbox == (30, 10, 50, 20)


@pytest.mark.parametrize(
    ("ratio", "label"),
    [(0.0001, "small"), (SMALL, "medium"), (0.01, "medium"), (MEDIUM, "large"), (0.5, "large")],
)
def test_umbrales_de_tamano(ratio: float, label: str) -> None:
    assert size_label(ratio, SMALL, MEDIUM) == label


def test_mascara_vacia_cae_a_la_caja() -> None:
    empty = np.zeros((H, W), dtype=bool)
    mask, source = ensure_nonempty(empty, (10.2, 20.0, 30.0, 40.7))
    assert source == "box_fallback"
    p = properties_from_mask(mask, SMALL, MEDIUM, source)
    assert p.mask_bbox == (10, 20, 30, 41)
    assert p.area_ratio > 0 and p.source == "box_fallback"


def test_mascara_vacia_sin_fallback_falla() -> None:
    with pytest.raises(ValueError):
        properties_from_mask(np.zeros((H, W), dtype=bool), SMALL, MEDIUM)


def test_caja_degenerada_o_fuera_de_la_imagen_da_al_menos_un_pixel() -> None:
    assert box_to_mask((50.0, 50.0, 50.0, 50.0), H, W).sum() == 1
    assert box_to_mask((500.0, 500.0, 600.0, 600.0), H, W).sum() == 1
    assert box_to_mask((-10.0, -10.0, 1000.0, 1000.0), H, W).all()


def test_mascara_no_vacia_no_se_toca() -> None:
    mask = np.zeros((H, W), dtype=bool)
    mask[5, 5] = True
    out, source = ensure_nonempty(mask, (0.0, 0.0, 100.0, 80.0))
    assert source == "mask" and out.sum() == 1


def test_desde_rle_igual_que_desde_mascara() -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((H, W)) > 0.7
    assert properties_from_rle(rle.encode(mask), SMALL, MEDIUM) == properties_from_mask(
        mask, SMALL, MEDIUM
    )


def test_desde_config(base_cfg: dict) -> None:
    mask = np.zeros((H, W), dtype=bool)
    mask[:, :] = True
    assert properties_from_config(mask, base_cfg).size == "large"
