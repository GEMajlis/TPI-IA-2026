"""Orquestador end-to-end: detección → post-proceso → SAM2 → SceneJSON, con traza y caché.

Caché por etapa en `paths.cache`, con una clave que es el hash de la parte de la config que
afecta a esa etapa (y a las anteriores). Así, cambiar el umbral reutiliza las detecciones
crudas, y cambiar las relaciones reutiliza también las máscaras.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scene.config import RunContext, config_hash
from scene.detection.base import load_detections, save_detections
from scene.detection.debug import draw_detections
from scene.detection.postprocess import postprocess
from scene.detection.vocabulary import Vocabulary
from scene.io.dataset import ImageRecord, load_image
from scene.scene.builder import build_scene
from scene.segmentation.debug import draw_masks
from scene.segmentation.sam2 import load_segmented, save_segmented
from scene.tracing.tracer import Tracer

STAGES = ("detection", "segmentation", "scene")


def stage_keys(cfg: dict) -> dict[str, str]:
    """Clave de caché de cada etapa: hash de su config y la de las etapas previas."""
    det = cfg["detection"]
    raw_part = {"detector": det["detector"], det["detector"]: det[det["detector"]]}
    raw_part["vocab"] = cfg["vocab"]
    pp_part = {k: v for k, v in det.items() if k not in ("detector", det["detector"])}
    seg_part = {"raw": raw_part, "pp": pp_part, "segmentation": cfg["segmentation"]}
    return {
        "detection": config_hash(raw_part)[:12],
        "segmentation": config_hash(seg_part)[:12],
    }


class Pipeline:
    """Corre la pipeline por imagen. Los modelos se cargan recién cuando hacen falta."""

    def __init__(
        self,
        cfg: dict,
        ctx: RunContext,
        tracer: Tracer,
        vocab: Vocabulary,
        make_detector: Callable[[], Any],
        make_segmenter: Callable[[], Any],
        from_stage: str | None = None,
        debug: bool = False,
    ) -> None:
        if from_stage is not None and from_stage not in STAGES:
            raise ValueError(f"--from-stage inválido {from_stage!r}; opciones: {STAGES}")
        self.cfg = cfg
        self.ctx = ctx
        self.tracer = tracer
        self.vocab = vocab
        self._make_detector, self._make_segmenter = make_detector, make_segmenter
        self._detector = self._segmenter = None
        # Las etapas desde `from_stage` en adelante ignoran la caché.
        self.force = set(STAGES[STAGES.index(from_stage) :]) if from_stage else set()
        self.debug = debug
        keys = stage_keys(cfg)
        cache = Path(cfg["paths"]["cache"])
        self.raw_dir = cache / "detections_raw" / keys["detection"]
        self.mask_dir = cache / "masks" / keys["segmentation"]
        self.scene_dir = ctx.run_dir / "scenes"
        self.models: dict[str, Any] = {}

    @property
    def detector(self):
        if self._detector is None:
            self._detector = self._make_detector()
            self.models["detector"] = self._detector.params()
            self.ctx.update_manifest("models", self.models)
        return self._detector

    @property
    def segmenter(self):
        if self._segmenter is None:
            self._segmenter = self._make_segmenter()
            self.models["segmenter"] = self._segmenter.params()
            self.ctx.update_manifest("models", self.models)
        return self._segmenter

    def _use_cache(self, stage: str, path: Path) -> bool:
        return stage not in self.force and path.is_file()

    def process(self, rec: ImageRecord) -> dict[str, Any]:
        """SceneJSON de una imagen (lo guarda en `scenes/<image_id>.json`)."""
        iid = rec.image_id
        image = load_image(rec)
        det_cfg = self.cfg["detection"]

        raw_path = self.raw_dir / f"{iid}.json"
        cached = self._use_cache("detection", raw_path)
        model_id = None if cached else self.detector.model_id
        params = {} if cached else self.detector.params()
        with self.tracer.stage(iid, "detection", params, model_id, n_in=1) as s:
            if cached:
                raw = load_detections(raw_path)
                s.status = "cached"
            else:
                raw = self.detector.detect(image)
                save_detections(raw, raw_path)
            s.n_out, s.artifact_path = len(raw), str(raw_path)

        pp_params = {k: v for k, v in det_cfg.items() if k not in ("detector", det_cfg["detector"])}
        with self.tracer.stage(iid, "detection_postprocess", pp_params, n_in=len(raw)) as s:
            res = postprocess(raw, det_cfg)
            s.n_out = len(res.detections)
            s.extra["counts"] = res.counts
        dets = res.detections

        mask_path = self.mask_dir / f"{iid}.json"
        cached = self._use_cache("segmentation", mask_path)
        model_id = None if cached else self.segmenter.model_id
        params = {} if cached else self.segmenter.params()
        with self.tracer.stage(iid, "segmentation", params, model_id, n_in=len(dets)) as s:
            if cached:
                objs = load_segmented(mask_path)
                s.status = "cached"
            else:
                objs = self.segmenter.segment(image, dets)
                save_segmented(objs, mask_path)
            s.n_out, s.artifact_path = len(objs), str(mask_path)
            s.extra["empty_masks"] = sum(1 for o in objs if not o.mask.any())

        scene_path = self.scene_dir / f"{iid}.json"
        with self.tracer.stage(iid, "scene", self.cfg["scene"]["relations"], n_in=len(objs)) as s:
            scene, stats = build_scene(iid, image, objs, self.cfg)
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
            s.n_out, s.artifact_path = len(scene["objects"]), str(scene_path)
            s.extra.update(stats)

        if self.debug:
            draw_detections(image, dets, self.ctx.run_dir / "debug" / f"{iid}_boxes.png")
            draw_masks(image, objs, self.ctx.run_dir / "debug" / f"{iid}_masks.png")
        return scene

    def run(self, images: list[ImageRecord]) -> dict[str, dict[str, Any]]:
        """Procesa todas las imágenes; una que falla queda en la traza y el run sigue."""
        scenes: dict[str, dict[str, Any]] = {}
        for rec in images:
            with self.tracer.image_guard(rec.image_id):
                scenes[rec.image_id] = self.process(rec)
        return scenes


def load_scenes(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Lee los SceneJSON de un run (`scenes/*.json`)."""
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(run_dir, "scenes").glob("*.json"))
    }
