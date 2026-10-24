"""SceneJSON "oráculo" construido desde el GT (solo evaluación, nunca inferencia).

Cada instancia anotada pasa por el mismo builder que las predicciones (propiedades, color y
relaciones con las mismas reglas), con confianza 1. Correr el mismo ejecutor sobre este JSON y
sobre el predicho permite atribuir un error de QA a la percepción (las respuestas difieren) o al
razonamiento (el oráculo también falla o el DSL interpretado es otro).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from eval.annotations import ImageGT
from scene.detection.base import Detection
from scene.scene.builder import build_scene
from scene.segmentation.sam2 import SegmentedObject


def oracle_scene(
    img: ImageGT, image: np.ndarray, cfg: dict, names: Mapping[int, str]
) -> dict[str, Any]:
    """SceneJSON con una entrada por instancia GT (sin crowd), validado contra el schema."""
    segmented = []
    for inst in img.instances:
        if inst.iscrowd:
            continue
        mask = img.mask(inst)
        det = Detection(
            tuple(float(v) for v in inst.bbox_xyxy),
            1.0,
            inst.category_id,
            names.get(inst.category_id, str(inst.category_id)),
            "gt",
            f"gt_{inst.id}",
        )
        segmented.append(SegmentedObject(det, mask, 1.0, int(mask.sum())))
    scene, _ = build_scene(img.image_id, image, segmented, cfg)
    return scene
