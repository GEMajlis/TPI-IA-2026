import csv

import numpy as np

from eval.annotations import GTInstance, ImageGT
from eval.experiments import append_row, run_metrics, stage_cost_s
from eval.segmentation import oracle_rows
from scene.detection.base import Detection
from scene.scene.builder import build_scene
from scene.segmentation.postprocess import postprocess_mask
from scene.segmentation.sam2 import SegmentedObject
from tests.conftest import ROOT

H, W = 100, 100
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


def test_postproceso_solo_en_mascaras_chicas() -> None:
    noise = square(90, 90, 92, 92)  # componente espuria fuera de la caja
    small = square(10, 10, 15, 15) | noise  # 29 px = 0,29 %
    big = square(10, 10, 60, 60) | noise  # 2504 px = 25 %
    cfg = {**PP, "apply_below_area_ratio": 0.01}
    assert postprocess_mask(small, (10, 10, 15, 15), cfg).sum() == 25  # se limpia
    assert postprocess_mask(big, (10, 10, 60, 60), cfg).sum() == 2504  # queda cruda
    assert postprocess_mask(big, (10, 10, 60, 60), PP).sum() == 2500  # sin límite: se limpia


class FakeSam:
    def segment(self, image, dets):
        out = []
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
            m = square(x1, y1, x2, y2) | square(95, 0, 100, 5)
            out.append(SegmentedObject(d, m, 0.9, int(m.sum())))
        return out


def poly(x1, y1, x2, y2) -> list:
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def gt_img() -> ImageGT:
    return ImageGT(
        "img",
        "img.png",
        W,
        H,
        instances=[
            GTInstance(1, 1, (10.0, 10.0, 14.0, 14.0), poly(10, 10, 14, 14), 16.0),  # small
            GTInstance(2, 2, (20.0, 20.0, 80.0, 80.0), poly(20, 20, 80, 80), 3600.0),  # large
        ],
    )


def test_oraculo_variante_post_small() -> None:
    rows = oracle_rows(gt_img(), FakeSam(), np.zeros((H, W, 3), np.uint8), PP, small_max=0.01)
    small, large = rows
    assert small["iou_post_small"] == small["iou_post"]  # chica → post-procesada
    assert large["iou_post_small"] == large["iou_raw"]  # grande → cruda


def test_run_metrics_y_costo(base_cfg) -> None:
    base_cfg["paths"]["data_root"] = str(ROOT / "data" / "competition")
    base_cfg["paths"]["scene_schema"] = "scene_schema.json"
    det = Detection((20.0, 20.0, 80.0, 80.0), 0.9, 2, "x", "fake", "x")
    seg = SegmentedObject(det, square(20, 20, 80, 80), 0.9, 3600)
    scene, _ = build_scene("img", np.zeros((H, W, 3), np.uint8), [seg], base_cfg)
    m = run_metrics({"img": scene}, {"img": gt_img()}, base_cfg, {"holdout": ["img"]})
    assert m["holdout_f1_box"] == m["holdout_f1_mask"] == 2 / 3  # 1 TP, 1 FN
    assert m["holdout_f1_mask_large"] == 1.0 and m["consistency_issues"] == 0
    trace = [
        {"stage": "segmentation", "status": "ok", "duration_ms": 1500.0},
        {"stage": "segmentation", "status": "cached", "duration_ms": 5.0},
        {"stage": "detection", "status": "ok", "duration_ms": 500.0},
    ]
    assert stage_cost_s(trace) == {"detection": 0.5, "segmentation": 1.5, "scene": 0.0}


def test_append_row(tmp_path) -> None:
    path = tmp_path / "experiments.csv"
    append_row(path, {"name": "a", "delta": 0.1})
    append_row(path, {"name": "b", "delta": -0.2})
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["name"] for r in rows] == ["a", "b"]
