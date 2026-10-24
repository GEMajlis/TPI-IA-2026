"""Tests del post-proceso de máscaras y de la caché (sin cargar SAM2)."""

import numpy as np

from scene.detection.base import Detection
from scene.segmentation.postprocess import (
    crop_to_box,
    fill_holes,
    largest_component,
    postprocess_mask,
)
from scene.segmentation.sam2 import (
    SegmentedObject,
    load_segmented,
    save_segmented,
    select_best_mask,
)

PP = {
    "crop_to_box": True,
    "box_margin": 0.0,
    "largest_component": True,
    "fill_holes": True,
    "max_hole_ratio": 0.05,
}


def test_recorte_a_la_caja_con_margen() -> None:
    mask = np.ones((20, 20), dtype=bool)
    assert crop_to_box(mask, (5, 5, 15, 15), 0.0).sum() == 100
    # margen 0,1 de una caja de 10 px → 1 px por lado → 12×12
    assert crop_to_box(mask, (5, 5, 15, 15), 0.1).sum() == 144


def test_recorte_no_se_sale_de_la_imagen() -> None:
    mask = np.ones((10, 10), dtype=bool)
    assert crop_to_box(mask, (-5, -5, 30, 30), 0.5).sum() == 100


def test_componente_principal() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:10, 0:10] = True  # 100 px
    mask[15:18, 15:18] = True  # 9 px, se descarta
    out = largest_component(mask)
    assert out.sum() == 100 and not out[16, 16]


def test_componente_principal_mascara_vacia() -> None:
    assert largest_component(np.zeros((5, 5), dtype=bool)).sum() == 0


def test_relleno_de_huecos_chicos_y_no_grandes() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True  # 900 px
    mask[10:12, 10:12] = False  # hueco de 4 px (< 5 % de 896) → se rellena
    mask[20:30, 20:30] = False  # hueco de 100 px → queda
    out = fill_holes(mask, 0.05)
    assert out[10, 10] and not out[25, 25]


def test_hueco_que_toca_el_borde_no_es_hueco() -> None:
    mask = np.ones((10, 10), dtype=bool)
    mask[0:2, 4:6] = False  # muesca en el borde superior
    assert not fill_holes(mask, 1.0)[0, 4]


def test_postprocess_completo_y_pasos_apagables() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True
    mask[25:28, 25:28] = True  # fuera de la caja
    box = (0.0, 0.0, 20.0, 20.0)
    assert postprocess_mask(mask, box, PP).sum() == 100
    off = {**PP, "crop_to_box": False, "largest_component": False}
    assert postprocess_mask(mask, box, off).sum() == 109


def test_seleccion_de_mascara_por_iou_predicho() -> None:
    masks = np.zeros((2, 3, 4, 4), dtype=bool)
    masks[0, 2] = True
    masks[1, 0, 0, 0] = True
    best, scores = select_best_mask(masks, np.array([[0.1, 0.2, 0.9], [0.8, 0.5, 0.1]]))
    assert best[0].all() and best[1].sum() == 1
    np.testing.assert_allclose(scores, [0.9, 0.8])


def test_cache_roundtrip(tmp_path) -> None:
    det = Detection((1.0, 1.0, 3.0, 3.0), 0.7, 5, "cup", "owlv2", "cup")
    mask = np.zeros((6, 5), dtype=bool)
    mask[1:3, 1:3] = True
    path = tmp_path / "masks.json"
    save_segmented([SegmentedObject(det, mask, 0.93, raw_area=4)], path)
    (obj,) = load_segmented(path)
    assert obj.detection == det and obj.mask_score == 0.93 and obj.raw_area == 4
    np.testing.assert_array_equal(obj.mask, mask)
