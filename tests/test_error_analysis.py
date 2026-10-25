import numpy as np
import pytest

from eval.annotations import GTInstance, ImageGT
from eval.error_analysis import (
    draw_case,
    fp_types,
    funnel_rows,
    funnel_summary,
    per_image_scores,
    pick_gallery,
)
from eval.lvis_federated import evaluate_from_config, predictions_from_scenes
from scene.detection.base import Detection
from scene.detection.postprocess import postprocess
from scene.scene.builder import build_scene
from scene.segmentation.sam2 import SegmentedObject
from tests.conftest import ROOT

H, W = 100, 200


def poly(b) -> list:
    x1, y1, x2, y2 = b
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


BOXES = {
    "ok": (0, 0, 20, 20),
    "mask": (30, 0, 50, 20),
    "no_box": (60, 0, 80, 20),
    "wrong_class": (90, 0, 110, 20),
    "below": (120, 0, 140, 20),
}


@pytest.fixture
def cfg(base_cfg):
    base_cfg["paths"]["data_root"] = str(ROOT / "data" / "competition")
    base_cfg["paths"]["scene_schema"] = "scene_schema.json"
    return base_cfg


def det(box, score, cid) -> Detection:
    return Detection(tuple(map(float, box)), score, cid, f"c{cid}", "fake", f"c{cid}")


@pytest.fixture
def case(cfg):
    cats = {"ok": 1, "mask": 1, "no_box": 2, "wrong_class": 3, "below": 4}
    insts = [
        GTInstance(k, cats[name], tuple(map(float, b)), poly(b), 400.0)
        for k, (name, b) in enumerate(BOXES.items(), 1)
    ]
    img = ImageGT("img", "img.png", W, H, instances=insts, neg_category_ids={9})
    raw = [
        det((0, 0, 20, 15), 0.9, 1),  # TP de "ok" (IoU 0,75)
        det(BOXES["mask"], 0.8, 1),
        # Clase 2 es positiva (tiene GT en otro lugar): esta caja sobre la instancia de clase 3
        # es un FP por confusión de clase y además deja a la de clase 3 como "wrong_class".
        det(BOXES["wrong_class"], 0.7, 2),
        det(BOXES["below"], 0.05, 4),  # clase correcta, bajo el umbral
        # Duplicado de "ok": IoU 0,75 con el GT pero 0,5 con la otra caja (el NMS no lo saca).
        det((0, 5, 20, 20), 0.6, 1),
        det((150, 50, 190, 90), 0.5, 9),  # categoría negativa sin objeto debajo
    ]
    final = postprocess(raw, cfg["detection"]).detections
    segs = []
    for d in final:
        mask = np.zeros((H, W), bool)
        x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
        if d.box_xyxy == tuple(map(float, BOXES["mask"])):
            mask[y1 : y1 + 4, x1 : x1 + 4] = True  # máscara mala
        else:
            mask[y1:y2, x1:x2] = True
        segs.append(SegmentedObject(d, mask, 0.9, int(mask.sum())))
    scene, _ = build_scene("img", np.zeros((H, W, 3), np.uint8), segs, cfg)
    gt = {"img": img}
    preds = predictions_from_scenes({"img": scene})
    box = evaluate_from_config(preds, gt, cfg, iou_type="bbox", image_ids=["img"])
    segm = evaluate_from_config(preds, gt, cfg, iou_type="segm", image_ids=["img"])
    return gt, {"img": raw}, {"img": scene}, box, segm


def test_embudo_asigna_cada_instancia_a_su_etapa(case, cfg) -> None:
    gt, raw, _, box, segm = case
    rows = funnel_rows(gt, raw, box, segm, cfg)
    assert dict(zip(rows["gt_id"], rows["lost_at"], strict=True)) == {
        1: "ok",
        2: "mask",
        3: "no_box",
        4: "wrong_class",
        5: "below_threshold",
    }
    s = funnel_summary(rows)
    assert s["global"]["n"] == 5 and s["global"]["frac_ok"] == 0.2
    assert s["recall_after_stage"]["after_no_box"] == 0.8
    assert s["recall_after_stage"]["after_mask"] == 0.2


def test_tipos_de_falso_positivo(case, cfg) -> None:
    gt, _, scenes, box, _ = case
    fps = fp_types(scenes, gt, box, cfg)
    assert sorted(fps["fp_type"]) == ["class_confusion", "duplicate", "negative_category"]


def test_galeria_y_dibujo(case, cfg, tmp_path) -> None:
    gt, _, scenes, box, _ = case
    scores = per_image_scores(box)
    gallery = pick_gallery(scores, fp_types(scenes, gt, box, cfg), 1)
    assert gallery["failed"] == ["img"] and gallery["ambiguous"] == []
    path = draw_case(
        np.zeros((H, W, 3), np.uint8), gt["img"], scenes["img"], box, {}, tmp_path / "x.png"
    )
    assert path.is_file()
