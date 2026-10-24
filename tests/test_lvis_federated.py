import json

import numpy as np
import pytest

from eval.annotations import GTInstance, ImageGT, annotations_path, load_annotations
from eval.lvis_federated import (
    Prediction,
    evaluate,
    evaluate_from_config,
    match_image,
    predictions_from_scenes,
    prf,
    save_metrics,
)
from scene.masks import rle
from tests.conftest import requires_data

HORSE, CAR, CHAIR, PERSON = 1, 2, 3, 4
KW = {"iou_thr": 0.5, "size_thresholds": (0.01, 0.1), "freq_bins": (1, 5)}


def _box_poly(x0, y0, x1, y1) -> list:
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _inst(i: int, cid: int, box) -> GTInstance:
    x0, y0, x1, y1 = box
    return GTInstance(i, cid, box, _box_poly(*box), float((x1 - x0) * (y1 - y0)))


def _box_mask(box, h=100, w=100) -> dict:
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return rle.encode(m)


@pytest.fixture
def img() -> ImageGT:
    """Ejemplo federado: horse positivo, car negativo, chair sin verificar, person no exhaustivo."""
    return ImageGT(
        image_id="img",
        file_name="img.jpg",
        width=100,
        height=100,
        instances=[
            _inst(1, HORSE, (10, 10, 50, 50)),
            _inst(2, HORSE, (60, 60, 90, 90)),
            _inst(3, PERSON, (0, 0, 20, 20)),
        ],
        neg_category_ids={CAR},
        not_exhaustive_category_ids={PERSON},
    )


def _kinds(events) -> dict:
    out: dict = {}
    for e in events:
        out.setdefault((e.category_id, e.kind), 0)
        out[(e.category_id, e.kind)] += 1
    return out


def test_ejemplo_horse_car_chair(img) -> None:
    preds = [
        Prediction(HORSE, 0.9, (11, 11, 50, 50)),  # TP
        Prediction(HORSE, 0.8, (10, 10, 49, 49)),  # duplicado: el GT ya está tomado → FP
        Prediction(CAR, 0.7, (30, 30, 60, 60)),  # categoría negativa → FP
        Prediction(CHAIR, 0.6, (5, 5, 40, 40)),  # no verificada → ignorada
        Prediction(PERSON, 0.95, (0, 0, 20, 20)),  # TP
        Prediction(PERSON, 0.5, (70, 0, 95, 30)),  # positiva no exhaustiva sin match → ignorada
    ]
    events = match_image(preds, img, "bbox", 0.5)
    assert _kinds(events) == {
        (HORSE, "tp"): 1,
        (HORSE, "fp"): 1,
        (HORSE, "fn"): 1,  # el segundo caballo no se detectó
        (CAR, "fp"): 1,
        (CHAIR, "ignored"): 1,
        (PERSON, "tp"): 1,
        (PERSON, "ignored"): 1,
    }
    res = evaluate({"img": preds}, {"img": img}, iou_type="bbox", **KW)
    s = res.summary
    assert (s["tp"], s["fp"], s["fn"], s["ignored"]) == (2, 2, 1, 2)
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(2 / 3)
    assert s["evaluable_coverage"] == pytest.approx(4 / 6)


def test_matching_greedy_por_score(img) -> None:
    # La de menor score tiene más IoU, pero gana la de mayor score (orden greedy).
    preds = [Prediction(HORSE, 0.3, (10, 10, 50, 50)), Prediction(HORSE, 0.9, (12, 12, 50, 50))]
    events = [e for e in match_image(preds, img, "bbox", 0.5) if e.category_id == HORSE]
    tp = [e for e in events if e.kind == "tp"]
    assert len(tp) == 1 and tp[0].score == 0.9


def test_iou_por_debajo_del_umbral_es_fp(img) -> None:
    events = match_image([Prediction(HORSE, 0.9, (10, 10, 30, 30))], img, "bbox", 0.5)
    assert _kinds(events)[(HORSE, "fp")] == 1


def test_prediccion_perfecta_f1_1(img) -> None:
    preds = [
        Prediction(g.category_id, 1.0, g.bbox_xyxy, _box_mask(g.bbox_xyxy)) for g in img.instances
    ]
    for iou_type in ("bbox", "segm"):
        res = evaluate({"img": preds}, {"img": img}, iou_type=iou_type, **KW)
        assert res.summary["f1"] == pytest.approx(1.0)
        assert res.summary["evaluable_coverage"] == pytest.approx(1.0)


def test_segm_usa_iou_de_mascaras(img) -> None:
    # Caja correcta pero máscara chica: TP en bbox, FP en segm.
    pred = Prediction(HORSE, 0.9, (10, 10, 50, 50), _box_mask((10, 10, 25, 25)))
    bbox = match_image([pred], img, "bbox", 0.5)
    segm = match_image([pred], img, "segm", 0.5)
    assert ("tp" in {e.kind for e in bbox}) and ("tp" not in {e.kind for e in segm})


def test_imagen_sin_predicciones_cuenta_fn(img) -> None:
    res = evaluate({}, {"img": img}, image_ids=["img"], iou_type="bbox", **KW)
    assert res.summary["fn"] == 3 and res.summary["f1"] == 0.0


def test_prf_denominadores_cero() -> None:
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_desgloses_y_guardado(img, tmp_path) -> None:
    preds = [Prediction(HORSE, 0.9, (11, 11, 50, 50)), Prediction(CAR, 0.7, (0, 0, 5, 5))]
    res = evaluate({"img": preds}, {"img": img}, iou_type="bbox", **KW)
    # TP horse: área 1600/10000 = 0.16 → large; FP car: 25/10000 → small.
    assert res.by_size["large"]["tp"] == 1 and res.by_size["small"]["fp"] == 1
    path = save_metrics(res, tmp_path, category_names={HORSE: "horse", CAR: "car"})
    data = json.loads(path.read_text())
    assert data["summary"]["tp"] == 1 and set(data["by_size"]) == {"small", "medium", "large"}
    csv_text = (tmp_path / "metrics_detection_per_category.csv").read_text()
    assert "horse" in csv_text and "category_name" in csv_text


def test_predictions_from_scenes() -> None:
    scene = {
        "objects": [
            {
                "category_id": 3,
                "confidence": 0.4,
                "bbox": [1, 2, 3, 4],
                "mask": {"size": [1, 1], "counts": [1]},
            }
        ]
    }
    (p,) = predictions_from_scenes({"a": scene})["a"]
    assert (p.category_id, p.score, p.bbox_xyxy) == (3, 0.4, (1, 2, 3, 4))


@requires_data
def test_gt_contra_si_mismo_da_f1_1(base_cfg) -> None:
    """Sanidad con datos reales: usar el GT como predicción da F1 = 1 en cajas y máscaras."""
    gt = load_annotations(annotations_path(base_cfg))
    ids = sorted(gt)[:10]
    preds = {
        i: [
            Prediction(g.category_id, 1.0, g.bbox_xyxy, rle.encode(gt[i].mask(g)))
            for g in gt[i].instances
        ]
        for i in ids
    }
    cfg = {
        "scene": {"size_thresholds": {"small_max": 0.01, "medium_max": 0.1}},
        "eval": {"iou_thr": 0.5, "frequency_bins": {"rare_max": 1, "common_max": 5}},
    }
    for iou_type in ("bbox", "segm"):
        res = evaluate_from_config(preds, gt, cfg, iou_type)
        assert res.summary["f1"] == pytest.approx(1.0), iou_type
