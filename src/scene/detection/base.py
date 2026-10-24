"""Contrato de los detectores: `Detection` y la interfaz `Detector`."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Una caja predicha, en píxeles de la imagen original."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    category_id: int
    category_name: str
    source: str  # "owlv2", "yolo_world", ...
    raw_query: str  # texto/sinónimo que la disparó

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        return cls(**{**data, "box_xyxy": tuple(data["box_xyxy"])})


class Detector(ABC):
    """Detector open-vocabulary. El vocabulario se fija al construirlo."""

    source: str
    model_id: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detecta sobre una imagen RGB uint8 HxWx3. Devuelve detecciones crudas (sin NMS)."""

    def params(self) -> dict[str, Any]:
        """Parámetros que se registran en la traza."""
        return {}


def save_detections(dets: list[Detection], path: str | Path) -> None:
    """Guarda detecciones en JSON (caché por imagen)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([d.to_dict() for d in dets], ensure_ascii=False), encoding="utf-8")


def load_detections(path: str | Path) -> list[Detection]:
    return [Detection.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
