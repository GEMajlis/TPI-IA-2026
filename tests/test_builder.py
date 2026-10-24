import numpy as np
import pytest

from scene.detection.base import Detection
from scene.masks import rle
from scene.scene.builder import build_scene
from scene.scene.schema import validate_scene
from scene.segmentation.sam2 import SegmentedObject
from tests.conftest import ROOT

SCHEMA = ROOT / "data" / "competition" / "scene_schema.json"
H, W = 60, 80


@pytest.fixture
def cfg(base_cfg: dict) -> dict:
    # El schema versionado en data/competition sirve aunque no esté la copia local del dataset.
    base_cfg["paths"]["data_root"] = str(ROOT / "data" / "competition")
    base_cfg["paths"]["scene_schema"] = "scene_schema.json"
    return base_cfg


def seg(box, score, cid=1, name="cup", fill=True) -> SegmentedObject:
    mask = np.zeros((H, W), bool)
    if fill:
        x1, y1, x2, y2 = (int(v) for v in box)
        mask[y1:y2, x1:x2] = True
    det = Detection(tuple(map(float, box)), score, cid, name, "owlv2", name)
    return SegmentedObject(det, mask, 0.9, int(mask.sum()))


def image() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :40] = (200, 30, 30)
    img[:, 40:] = (40, 80, 200)
    return img


def test_scene_valida_con_ids_por_confianza_y_relaciones(cfg: dict) -> None:
    objs = [seg((50, 10, 70, 30), 0.4, 2, "plate"), seg((5, 10, 25, 30), 0.8)]
    scene, stats = build_scene("tpi_9999", image(), objs, cfg)
    assert validate_scene(scene, SCHEMA) == []
    a, b = scene["objects"]
    assert (a["id"], a["category_name"], b["id"]) == ("obj_0", "cup", "obj_1")
    assert a["attributes"]["color"] == "red" and b["attributes"]["color"] == "blue"
    assert {"subject": "obj_0", "predicate": "left_of", "object": "obj_1"} in scene["relations"]
    assert scene["image"] == {"id": "tpi_9999", "width": W, "height": H}
    assert stats["n_relations"] == len(scene["relations"])


def test_propiedades_coinciden_con_la_mascara_entregada(cfg: dict) -> None:
    scene, _ = build_scene("x", image(), [seg((5, 10, 25, 30), 0.8)], cfg)
    obj = scene["objects"][0]
    mask = rle.decode(obj["mask"])
    assert mask.sum() == obj["area_px"] == 400
    assert obj["area_ratio"] == pytest.approx(400 / (H * W))
    assert obj["centroid"] == [14.5, 19.5]
    assert obj["provenance"]["mask"] == "observed"


def test_mascara_vacia_usa_la_caja_y_queda_valida(cfg: dict) -> None:
    scene, stats = build_scene("x", image(), [seg((5, 10, 25, 30), 0.8, fill=False)], cfg)
    obj = scene["objects"][0]
    assert validate_scene(scene, SCHEMA) == []
    assert obj["source"]["mask"] == "box_fallback" and obj["provenance"]["mask"] == "derived"
    assert rle.decode(obj["mask"]).sum() == 400
    assert stats["box_fallback_masks"] == 1


def test_imagen_sin_detecciones(cfg: dict) -> None:
    scene, stats = build_scene("x", image(), [], cfg)
    assert scene["objects"] == [] and scene["relations"] == []
    assert validate_scene(scene, SCHEMA) == []
    assert stats["n_objects"] == 0
