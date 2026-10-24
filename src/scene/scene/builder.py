"""Builder del SceneJSON: une detecciones, máscaras, propiedades, color y relaciones, y valida.

Cada objeto lleva `provenance` con el origen de sus campos:
- `observed`: salida directa de un modelo (caja y confianza del detector, máscara de SAM2);
- `derived`: calculado con código a partir de lo observado (centroide, posición, color, ...);
- `unknown`: no se pudo determinar (el valor queda en `null`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scene.config import data_path
from scene.masks import rle
from scene.scene.color import dominant_color_from_config
from scene.scene.properties import ensure_nonempty, properties_from_config
from scene.scene.relations import RelObject, compute_relations
from scene.scene.schema import assert_valid_scene
from scene.segmentation.sam2 import SegmentedObject

DECIMALS = 2  # redondeo de coordenadas en píxeles (cajas y centroides)


def _round(values, ndigits: int = DECIMALS) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def build_object(
    oid: str, seg: SegmentedObject, image: np.ndarray, cfg: dict
) -> tuple[dict[str, Any], RelObject]:
    """Un objeto del SceneJSON y su vista para el cálculo de relaciones."""
    det = seg.detection
    mask, mask_source = ensure_nonempty(seg.mask, det.box_xyxy)
    props = properties_from_config(mask, cfg, mask_source)
    color = dominant_color_from_config(image, mask, cfg)
    obj = {
        "id": oid,
        "category_id": det.category_id,
        "category_name": det.category_name,
        "confidence": round(det.score, 4),
        "bbox": _round(det.box_xyxy),
        "mask": rle.encode(mask),
        "centroid": _round(props.centroid),
        "area_ratio": props.area_ratio,
        "position": props.position,
        "size": props.size,
        "area_px": props.area,
        "mask_bbox": list(props.mask_bbox),
        "mask_score": round(seg.mask_score, 4),
        "attributes": {
            "color": color.name if color else None,
            "color_confidence": round(color.confidence, 3) if color else None,
            "color_rgb": list(color.rgb) if color else None,
        },
        "provenance": {
            "category_id": "observed",
            "confidence": "observed",
            "bbox": "observed",
            "mask": "observed" if mask_source == "mask" else "derived",
            "centroid": "derived",
            "area_ratio": "derived",
            "position": "derived",
            "size": "derived",
            "color": "derived" if color else "unknown",
        },
        "source": {"detector": det.source, "raw_query": det.raw_query, "mask": mask_source},
    }
    view = RelObject(oid, props.centroid, props.mask_bbox, mask, det.score)
    return obj, view


def build_scene(
    image_id: str,
    image: np.ndarray,
    segmented: list[SegmentedObject],
    cfg: dict,
    validate: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """SceneJSON de una imagen y estadísticas para la traza. Ids `obj_<n>` por confianza."""
    height, width = image.shape[:2]
    ordered = sorted(segmented, key=lambda s: -s.detection.score)
    objects, views = [], []
    for n, seg in enumerate(ordered):
        obj, view = build_object(f"obj_{n}", seg, image, cfg)
        objects.append(obj)
        views.append(view)
    rel = compute_relations(views, width, height, cfg["scene"]["relations"])
    scene = {
        "image": {"id": image_id, "width": int(width), "height": int(height)},
        "objects": objects,
        "relations": [r.to_dict() for r in rel.relations],
        "extra_relations": [r.to_dict() for r in rel.extra_relations],
    }
    if validate:
        assert_valid_scene(scene, data_path(cfg, "scene_schema"))
    stats = {
        **rel.stats,
        "box_fallback_masks": sum(o["source"]["mask"] == "box_fallback" for o in objects),
        "unknown_color": sum(o["attributes"]["color"] is None for o in objects),
    }
    return scene, stats
