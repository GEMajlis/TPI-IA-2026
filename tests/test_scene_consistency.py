import copy

import numpy as np
import pytest

from eval.annotations import GTInstance, ImageGT
from eval.scene_consistency import (
    attribute_accuracy,
    grid_cell,
    internal_consistency,
    relation_issues,
)
from scene.detection.base import Detection
from scene.scene.builder import build_scene
from scene.segmentation.sam2 import SegmentedObject
from tests.conftest import ROOT

H, W = 60, 90


@pytest.fixture
def cfg(base_cfg: dict) -> dict:
    base_cfg["paths"]["data_root"] = str(ROOT / "data" / "competition")
    base_cfg["paths"]["scene_schema"] = "scene_schema.json"
    return base_cfg


def image() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :45] = (200, 30, 30)
    img[:, 45:] = (40, 80, 200)
    return img


def seg(box, score, cid, name) -> SegmentedObject:
    x1, y1, x2, y2 = box
    mask = np.zeros((H, W), bool)
    mask[y1:y2, x1:x2] = True
    det = Detection(tuple(map(float, box)), score, cid, name, "fake", name)
    return SegmentedObject(det, mask, 0.9, int(mask.sum()))


@pytest.fixture
def scene(cfg) -> dict:
    objs = [seg((5, 5, 25, 25), 0.9, 1, "cup"), seg((60, 30, 80, 55), 0.7, 2, "plate")]
    return build_scene("img", image(), objs, cfg)[0]


def test_escena_del_builder_es_consistente(scene, cfg) -> None:
    res = internal_consistency({"img": scene}, cfg)
    assert res["n_issues"] == 0, res["issues"]
    assert res["mask_outside_bbox"]["mean"] == 0.0


def test_detecta_campos_derivados_inconsistentes(scene, cfg) -> None:
    bad = copy.deepcopy(scene)
    bad["objects"][0]["position"] = "bottom_right"
    bad["objects"][0]["size"] = "small"  # 400 px de 5400 → large
    bad["objects"][1]["centroid"] = [0.0, 0.0]
    bad["objects"][1]["area_ratio"] = 0.5
    res = internal_consistency({"img": bad}, cfg)
    # mover el centroide también deja incoherentes las 4 relaciones direccionales del par
    assert res["issues"] == {
        "position": 1,
        "size": 1,
        "centroid": 1,
        "area_ratio": 1,
        "relation_direction": 4,
    }
    assert res["scenes_with_issues"] == ["img"]


def test_detecta_relaciones_incoherentes(scene) -> None:
    bad = copy.deepcopy(scene)
    bad["relations"] = [
        {
            "subject": "obj_0",
            "predicate": "right_of",
            "object": "obj_1",
        },  # dirección mal y sin inversa
        {"subject": "obj_0", "predicate": "near", "object": "obj_1"},  # fuera del enum
        {"subject": "obj_9", "predicate": "above", "object": "obj_1"},  # id inexistente
    ]
    assert sorted(relation_issues(bad)) == [
        "relation_direction",
        "relation_missing_inverse",
        "relation_predicate",
        "relation_unknown_id",
    ]


def test_mascara_fuera_de_la_caja_se_mide(scene, cfg) -> None:
    moved = copy.deepcopy(scene)
    moved["objects"][0]["bbox"] = [5.0, 5.0, 15.0, 25.0]  # deja afuera la mitad de la máscara
    res = internal_consistency({"img": moved}, cfg)
    assert res["mask_outside_bbox"]["frac_objects_gt_10pct"] == 0.5


def test_grilla_coincide_con_la_del_builder() -> None:
    assert grid_cell(0, 0, W, H) == "top_left"
    assert grid_cell(30, 20, W, H) == "center"
    assert grid_cell(89.9, 59.9, W, H) == "bottom_right"


def test_exactitud_de_atributos_contra_gt(scene, cfg) -> None:
    def inst(iid, cid, box):
        x1, y1, x2, y2 = box
        poly = [[x1, y1, x2, y1, x2, y2, x1, y2]]
        return GTInstance(iid, cid, tuple(map(float, box)), poly, (x2 - x1) * (y2 - y1))

    gt = {
        "img": ImageGT(
            "img",
            "img.png",
            W,
            H,
            instances=[inst(10, 1, (5, 5, 25, 25)), inst(11, 2, (60, 30, 80, 55))],
        )
    }
    res = attribute_accuracy({"img": scene}, gt, cfg, images={"img": image()})
    assert res["n_pairs"] == 2
    assert res["position_acc"] == 1.0 and res["size_acc"] == 1.0
    assert res["color_acc"] == 1.0 and res["n_color_pairs"] == 2
