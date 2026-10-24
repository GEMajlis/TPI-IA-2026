import numpy as np
import pytest

from scene.detection.base import Detection, load_detections, save_detections
from scene.detection.postprocess import (
    apply_threshold,
    box_iou_matrix,
    cross_class_nms,
    nms,
    per_class_nms,
    postprocess,
)


def det(box, score, cid=1, name="cup", query=None) -> Detection:
    return Detection(tuple(map(float, box)), score, cid, name, "test", query or name)


DET_CFG = {
    "score_threshold": 0.2,
    "per_class_threshold": {},
    "nms_iou": 0.5,
    "cross_class_nms": {"enabled": False, "iou": 0.8},
    "max_detections": 100,
}


def test_iou_casos_basicos() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 0, 0]], dtype=float)
    np.testing.assert_allclose(box_iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0, 0.0])


def test_nms_suprime_solapadas_y_conserva_separadas() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.5, 0.9, 0.3])
    assert nms(boxes, scores, 0.5) == [1, 2]


def test_nms_iou_igual_al_umbral_no_suprime() -> None:
    # IoU exacto = 1/3: con umbral 1/3 se conservan ambas (se suprime solo si IoU > umbral).
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8]), 50 / 150) == [0, 1]


def test_umbral_global_y_por_clase() -> None:
    dets = [det([0, 0, 1, 1], 0.15, 1, "cup"), det([0, 0, 1, 1], 0.15, 2, "car")]
    assert apply_threshold(dets, 0.2) == []
    kept = apply_threshold(dets, 0.2, per_class={"car": 0.1})
    assert [d.category_name for d in kept] == ["car"]


def test_nms_por_clase_no_mezcla_clases_y_une_sinonimos() -> None:
    dets = [
        det([0, 0, 10, 10], 0.9, 1, "cup", "cup"),
        det([0, 0, 10, 10], 0.7, 1, "cup", "mug"),  # sinónimo, misma clase → se suprime
        det([0, 0, 10, 10], 0.8, 2, "drinking glass"),  # otra clase → se conserva
    ]
    kept = per_class_nms(dets, 0.5)
    assert [(d.category_name, d.raw_query) for d in kept] == [
        ("cup", "cup"),
        ("drinking glass", "drinking glass"),
    ]


def test_nms_entre_clases() -> None:
    dets = [det([0, 0, 10, 10], 0.9, 1, "cup"), det([0, 0, 10, 10], 0.8, 2, "drinking glass")]
    assert [d.category_name for d in cross_class_nms(dets, 0.8)] == ["cup"]


def test_postprocess_registra_conteos_por_filtro() -> None:
    dets = [
        det([0, 0, 10, 10], 0.9),
        det([1, 0, 11, 10], 0.8),  # duplicado de la anterior
        det([50, 50, 60, 60], 0.5),
        det([70, 70, 80, 80], 0.1),  # bajo el umbral
    ]
    res = postprocess(dets, {**DET_CFG, "max_detections": 1})
    assert res.counts == {"raw": 4, "threshold": 3, "nms_per_class": 2, "max_detections": 1}
    assert [d.score for d in res.detections] == [0.9]


def test_postprocess_con_nms_entre_clases_activo() -> None:
    dets = [det([0, 0, 10, 10], 0.9, 1, "cup"), det([0, 0, 10, 10], 0.8, 2, "drinking glass")]
    cfg = {**DET_CFG, "cross_class_nms": {"enabled": True, "iou": 0.8}}
    res = postprocess(dets, cfg)
    assert res.counts["nms_cross_class"] == 1
    assert len(res.detections) == 1


def test_postprocess_vacio() -> None:
    res = postprocess([], DET_CFG)
    assert res.detections == []
    assert res.counts["max_detections"] == 0


@pytest.mark.parametrize("n", [0, 3])
def test_roundtrip_json(tmp_path, n: int) -> None:
    dets = [det([i, i, i + 5, i + 5], 0.5 + i / 10) for i in range(n)]
    path = tmp_path / "dets.json"
    save_detections(dets, path)
    assert load_detections(path) == dets
