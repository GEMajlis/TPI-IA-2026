import numpy as np

from eval.annotations import GTInstance, ImageGT
from eval.lvis_federated import Prediction, evaluate
from eval.segmentation import (
    gt_detections,
    mask_iou,
    oracle_rows,
    predicted_pairs_summary,
    summarize_oracle,
)
from scene.masks import rle
from scene.segmentation.sam2 import SegmentedObject

H, W = 50, 50
PP = {
    "crop_to_box": True,
    "box_margin": 0.0,
    "largest_component": True,
    "fill_holes": True,
    "max_hole_ratio": 0.05,
}


def square(x1, y1, x2, y2) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return m


def poly(x1, y1, x2, y2) -> list:
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def img_gt() -> ImageGT:
    return ImageGT(
        "img",
        "img.png",
        W,
        H,
        instances=[
            GTInstance(1, 3, (10.0, 10.0, 30.0, 30.0), poly(10, 10, 30, 30), 400.0),
            GTInstance(2, 4, (35.0, 35.0, 45.0, 45.0), poly(35, 35, 45, 45), 100.0),
        ],
    )


class FakeSam:
    """Devuelve la caja como máscara más ruido fuera de la caja (que el post-proceso limpia)."""

    def segment(self, image, dets):
        out = []
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
            m = square(x1, y1, x2, y2)
            m[0:3, 0:3] = True  # componente espuria lejos de la caja
            out.append(SegmentedObject(d, m, 0.9, int(m.sum())))
        return out


def test_mask_iou() -> None:
    assert mask_iou(square(0, 0, 10, 10), square(5, 0, 15, 10)) == 50 / 150
    assert mask_iou(np.zeros((H, W), bool), np.zeros((H, W), bool)) == 0.0


def test_detecciones_desde_gt_llevan_el_id_de_instancia() -> None:
    dets = gt_detections(img_gt())
    assert [d.raw_query for d in dets] == ["1", "2"] and all(d.score == 1.0 for d in dets)


def test_oraculo_mide_cruda_y_post_procesada() -> None:
    rows = oracle_rows(img_gt(), FakeSam(), np.zeros((H, W, 3), np.uint8), PP)
    assert [r["gt_id"] for r in rows] == [1, 2]
    assert all(r["iou_post"] > r["iou_raw"] for r in rows)
    assert rows[0]["iou_post"] > 0.9


def test_resumen_por_tamano(base_cfg) -> None:
    rows = [
        {"area_ratio": 0.001, "iou_raw": 0.4, "iou_post": 0.6, "mask_score": 0.5},
        {"area_ratio": 0.2, "iou_raw": 0.9, "iou_post": 0.9, "mask_score": 0.9},
        {"area_ratio": 0.2, "iou_raw": 0.7, "iou_post": 0.8, "mask_score": 0.8},
    ]
    s = summarize_oracle(rows, base_cfg)
    assert s["n"] == 3 and s["frac_iou_ge_thr_raw"] == 2 / 3 and s["frac_iou_ge_thr_post"] == 1.0
    assert s["by_size"]["small"]["n"] == 1 and s["by_size"]["large"]["n"] == 2
    assert s["by_size"]["medium"]["miou_post"] is None
    assert s["corr_mask_score_iou"] > 0.9


def test_tp_de_cajas_que_se_pierden_en_mascaras() -> None:
    gt = {"img": img_gt()}
    preds = {
        "img": [
            # caja y máscara buenas
            Prediction(3, 0.9, (10, 10, 30, 30), rle.encode(square(10, 10, 30, 30)), "obj_0"),
            # caja buena, máscara mala (solo una esquina)
            Prediction(4, 0.8, (35, 35, 45, 45), rle.encode(square(35, 35, 38, 38)), "obj_1"),
        ]
    }
    kw = {"iou_thr": 0.5, "size_thresholds": (0.00333, 0.03), "freq_bins": (4, 7)}
    box = evaluate(preds, gt, iou_type="bbox", **kw)
    segm = evaluate(preds, gt, iou_type="segm", **kw)
    s = predicted_pairs_summary(box, segm)
    assert (s["tp_box"], s["tp_mask"], s["box_tp_lost_in_mask"]) == (2, 1, 1)
    assert s["miou_mask_tp"] == 1.0
