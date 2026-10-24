"""Tests de la conversión de salidas de OWLv2 (sin cargar el modelo)."""

import numpy as np

from scene.detection.owlv2 import decode_detections, owl_boxes_to_xyxy
from scene.detection.vocabulary import build_vocabulary
from scene.io.dataset import Category


def test_cajas_relativas_al_cuadrado_con_padding() -> None:
    # Imagen apaisada 640×480: el cuadrado con padding mide 640, la imagen ocupa la parte superior.
    boxes = np.array([[0.5, 0.375, 0.25, 0.25]])  # centro de la imagen real
    np.testing.assert_allclose(owl_boxes_to_xyxy(boxes, 640, 480), [[240, 160, 400, 320]])


def test_cajas_se_recortan_a_la_imagen() -> None:
    # Imagen vertical 300×600: una caja que cae en el padding derecho se recorta al ancho.
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
    np.testing.assert_allclose(owl_boxes_to_xyxy(boxes, 300, 600), [[240, 240, 300, 360]])


CATS = [Category(1, "cup"), Category(2, "car")]
VOCAB = build_vocabulary(CATS, mode="synonyms", synonyms={1: ["mug"]}, template="a photo of a {}")
BOXES = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 5, 5, 9]], dtype=float)  # la 3.ª es vacía


def test_decode_queda_el_maximo_por_clase_y_su_query() -> None:
    assert [q.text for q in VOCAB.queries] == ["cup", "mug", "car"]
    scores = np.array([[0.2, 0.6, 0.1], [0.05, 0.0, 0.3], [0.9, 0.9, 0.9]])
    dets = decode_detections(scores, BOXES, VOCAB, score_floor=0.1, raw_topk=10, labels_per_box=1)
    assert [(d.category_name, d.raw_query, round(d.score, 2)) for d in dets] == [
        ("cup", "mug", 0.6),
        ("car", "car", 0.3),
    ]
    assert dets[0].box_xyxy == (0.0, 0.0, 10.0, 10.0)


def test_decode_varias_etiquetas_por_caja_y_topk() -> None:
    scores = np.array([[0.2, 0.6, 0.5], [0.05, 0.0, 0.3], [0, 0, 0]])
    two = decode_detections(scores, BOXES, VOCAB, score_floor=0.1, raw_topk=10, labels_per_box=2)
    assert [(d.category_name, round(d.score, 2)) for d in two] == [
        ("cup", 0.6),
        ("car", 0.5),
        ("car", 0.3),
    ]
    top1 = decode_detections(scores, BOXES, VOCAB, score_floor=0.1, raw_topk=1, labels_per_box=2)
    assert len(top1) == 1
