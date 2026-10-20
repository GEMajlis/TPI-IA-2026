"""Carga del GT (anotaciones COCO/LVIS). SOLO para evaluación y EDA, nunca para inferencia."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class GTInstance:
    """Instancia anotada. `bbox_xyxy` en píxeles de la imagen original."""

    id: int
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    segmentation: list | dict
    area: float
    iscrowd: int = 0


@dataclass
class ImageGT:
    """GT de una imagen más su metadata federada LVIS."""

    image_id: str
    file_name: str
    width: int
    height: int
    instances: list[GTInstance] = field(default_factory=list)
    neg_category_ids: set[int] = field(default_factory=set)
    not_exhaustive_category_ids: set[int] = field(default_factory=set)

    @property
    def positive_category_ids(self) -> set[int]:
        """Categorías con al menos una anotación."""
        return {inst.category_id for inst in self.instances}

    def mask(self, inst: GTInstance) -> np.ndarray:
        """Máscara bool HxW de una instancia."""
        return segmentation_to_mask(inst.segmentation, self.height, self.width)


def segmentation_to_mask(segmentation: list | dict, height: int, width: int) -> np.ndarray:
    """Convierte polígonos o RLE (COCO) en una máscara bool HxW."""
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    return mask_utils.decode(rle).astype(bool)


def annotations_path(cfg: dict) -> Path:
    """Ruta del archivo de anotaciones de desarrollo según la config."""
    return Path(cfg["paths"]["data_root"]) / cfg["eval"]["annotations"]


def load_annotations(path: str | Path) -> dict[str, ImageGT]:
    """Lee `annotations.json` y devuelve el GT indexado por image_id (`tpi_0001`, ...)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    by_num: dict[int, ImageGT] = {}
    for img in data["images"]:
        by_num[img["id"]] = ImageGT(
            image_id=Path(img["file_name"]).stem,
            file_name=img["file_name"],
            width=img["width"],
            height=img["height"],
            neg_category_ids=set(img.get("neg_category_ids", [])),
            not_exhaustive_category_ids=set(img.get("not_exhaustive_category_ids", [])),
        )
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        by_num[ann["image_id"]].instances.append(
            GTInstance(
                id=ann["id"],
                category_id=ann["category_id"],
                bbox_xyxy=(x, y, x + w, y + h),
                segmentation=ann["segmentation"],
                area=ann["area"],
                iscrowd=ann.get("iscrowd", 0),
            )
        )
    return {gt.image_id: gt for gt in by_num.values()}
