"""Orquestador con modelos falsos (caché y traza) y submission contra sample_submission.csv."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

from scene.config import start_run
from scene.detection.base import Detection
from scene.detection.vocabulary import build_vocabulary
from scene.io.dataset import Category, image_record_from_path
from scene.io.submission import build_submission, read_submission, sample_ids, write_submission
from scene.pipeline import Pipeline, load_scenes
from scene.segmentation.sam2 import SegmentedObject
from scene.tracing.tracer import Tracer, read_trace
from tests.conftest import ROOT

COMPETITION = ROOT / "data" / "competition"


@pytest.fixture
def cfg(base_cfg: dict, tmp_path) -> dict:
    base_cfg["paths"]["data_root"] = str(COMPETITION)
    base_cfg["paths"]["scene_schema"] = "scene_schema.json"
    base_cfg["paths"]["sample_submission"] = "sample_submission.csv"
    base_cfg["paths"]["cache"] = str(tmp_path / "cache")
    return base_cfg


class FakeDetector:
    source = model_id = "fake-det"

    def __init__(self) -> None:
        self.calls = 0

    def params(self) -> dict:
        return {"model_id": self.model_id}

    def detect(self, image):
        self.calls += 1
        return [
            Detection((5.0, 5.0, 25.0, 25.0), 0.9, 1, "cup", "fake", "cup"),
            Detection((40.0, 10.0, 60.0, 30.0), 0.3, 2, "plate", "fake", "plate"),
            Detection((0.0, 0.0, 2.0, 2.0), 0.02, 1, "cup", "fake", "cup"),  # bajo el umbral
        ]


class FakeSegmenter:
    model_id = "fake-seg"

    def __init__(self) -> None:
        self.calls = 0

    def params(self) -> dict:
        return {"model_id": self.model_id}

    def segment(self, image, dets):
        self.calls += 1
        out = []
        for d in dets:
            mask = np.zeros(image.shape[:2], bool)
            x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
            mask[y1:y2, x1:x2] = True
            out.append(SegmentedObject(d, mask, 0.95, int(mask.sum())))
        return out


def make_pipe(cfg, det, seg, **kw):
    ctx = start_run(cfg)
    tracer = Tracer(ctx.trace_path, ctx.run_id, ctx.config_hash)
    vocab = build_vocabulary([Category(1, "cup"), Category(2, "plate")])
    return Pipeline(cfg, ctx, tracer, vocab, lambda: det, lambda: seg, **kw), ctx


@pytest.fixture
def images(tmp_path):
    recs = []
    for i in range(2):
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(np.full((40, 80, 3), 100 + 50 * i, np.uint8)).save(path)
        recs.append(image_record_from_path(path))
    return recs


def test_pipeline_genera_escenas_validas_y_traza(cfg, images) -> None:
    det, seg = FakeDetector(), FakeSegmenter()
    pipe, ctx = make_pipe(cfg, det, seg)
    scenes = pipe.run(images)
    assert set(scenes) == {"img_0", "img_1"}
    assert [o["category_name"] for o in scenes["img_0"]["objects"]] == ["cup", "plate"]
    assert load_scenes(ctx.run_dir) == scenes
    stages = [r["stage"] for r in read_trace(ctx.trace_path)]
    assert stages == ["detection", "detection_postprocess", "segmentation", "scene"] * 2
    counts = read_trace(ctx.trace_path)[1]["extra"]["counts"]
    assert counts["raw"] == 3 and counts["threshold"] == 2


def test_cambiar_el_umbral_reutiliza_detecciones_crudas(cfg, images) -> None:
    det, seg = FakeDetector(), FakeSegmenter()
    make_pipe(cfg, det, seg)[0].run(images)
    assert (det.calls, seg.calls) == (2, 2)

    cfg2 = {**cfg, "detection": {**cfg["detection"], "score_threshold": 0.5}}
    det2, seg2 = FakeDetector(), FakeSegmenter()
    pipe, ctx = make_pipe(cfg2, det2, seg2)
    scenes = pipe.run(images)
    assert det2.calls == 0  # detecciones crudas desde la caché
    assert seg2.calls == 2  # cambió el post-proceso → SAM2 de nuevo
    assert [o["category_name"] for o in scenes["img_0"]["objects"]] == ["cup"]
    status = {(r["stage"], r["status"]) for r in read_trace(ctx.trace_path)}
    assert ("detection", "cached") in status and ("segmentation", "ok") in status


def test_misma_config_reutiliza_todo_y_from_stage_fuerza(cfg, images) -> None:
    make_pipe(cfg, FakeDetector(), FakeSegmenter())[0].run(images)
    det, seg = FakeDetector(), FakeSegmenter()
    make_pipe(cfg, det, seg)[0].run(images)
    assert (det.calls, seg.calls) == (0, 0)
    det, seg = FakeDetector(), FakeSegmenter()
    make_pipe(cfg, det, seg, from_stage="segmentation")[0].run(images)
    assert (det.calls, seg.calls) == (0, 2)


def test_una_imagen_que_falla_no_corta_el_run(cfg, images, tmp_path) -> None:
    broken = image_record_from_path(images[0].path)
    broken = type(broken)("roto", tmp_path / "no_existe.png", "external", 10, 10)
    pipe, _ = make_pipe(cfg, FakeDetector(), FakeSegmenter())
    scenes = pipe.run([broken, images[1]])
    assert list(scenes) == ["img_1"] and pipe.tracer.failed_images == ["roto"]


def _sample_scenes() -> dict:
    with (COMPETITION / "sample_submission.csv").open(encoding="utf-8", newline="") as f:
        return {r["image_id"]: json.loads(r["SceneJSON"]) for r in csv.DictReader(f)}


def test_submission_mismo_orden_e_ids_que_el_sample(cfg, tmp_path) -> None:
    scenes = _sample_scenes()
    rows, errors = build_submission(dict(reversed(list(scenes.items()))), cfg)
    assert errors == {}
    assert [iid for iid, _ in rows] == sample_ids(cfg) and len(rows) == 80
    path = write_submission(rows, tmp_path / "submission.csv")
    assert read_submission(path) == {iid: scenes[iid] for iid in sample_ids(cfg)}
    assert path.read_text(encoding="utf-8").splitlines()[0] == "image_id,SceneJSON"


def test_submission_reporta_faltantes_e_invalidos(cfg) -> None:
    scenes = _sample_scenes()
    del scenes["test_0001"]
    scenes["test_0002"]["objects"][0]["position"] = "arriba"
    _, errors = build_submission(scenes, cfg)
    assert set(errors) == {"test_0001", "test_0002"}
