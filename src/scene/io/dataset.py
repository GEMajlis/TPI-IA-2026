"""Listado de imágenes, vocabulario y preguntas del benchmark.

Este módulo es parte de la inferencia: NO lee anotaciones de instancias. Las dimensiones
de cada imagen salen del propio archivo (no de la metadata de GT), así la pipeline puede
procesar una imagen arbitraria.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scene.config import data_path

SPLITS = ("dev", "test", "benchmark")


@dataclass(frozen=True)
class ImageRecord:
    """Imagen a procesar. `image_id` es el nombre del archivo sin extensión (p. ej. `tpi_0001`)."""

    image_id: str
    path: Path
    split: str
    width: int
    height: int


@dataclass(frozen=True)
class Category:
    """Categoría del vocabulario evaluable. Los sinónimos se agregan en E1-08."""

    id: int
    name: str
    synonyms: tuple[str, ...] = ()


def image_record_from_path(path: str | Path, split: str = "external") -> ImageRecord:
    """Arma un `ImageRecord` para cualquier imagen leyendo solo su cabecera."""
    path = Path(path)
    with Image.open(path) as img:
        width, height = img.size
    return ImageRecord(image_id=path.stem, path=path, split=split, width=width, height=height)


def load_image(record: ImageRecord) -> np.ndarray:
    """Carga la imagen como array RGB uint8 HxWx3 (sin rotar por EXIF, igual que COCO)."""
    with Image.open(record.path) as img:
        return np.asarray(img.convert("RGB"))


def load_vocabulary(cfg: dict) -> list[Category]:
    """Lee `categories.csv` (category_id, category_name) de la competencia."""
    with data_path(cfg, "categories").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [Category(id=int(r["category_id"]), name=r["category_name"].strip()) for r in rows]


def load_questions(cfg: dict) -> list[dict]:
    """Lee el benchmark de preguntas (`questions_dev.jsonl`). No contiene respuestas."""
    with data_path(cfg, "questions").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dev_images(cfg: dict) -> list[ImageRecord]:
    exts = {e.lower() for e in cfg["dataset"]["image_extensions"]}
    folder = data_path(cfg, "dev_images")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    return [image_record_from_path(p, split="dev") for p in files]


def _test_images(cfg: dict) -> list[ImageRecord]:
    folder = data_path(cfg, "test_images")
    with data_path(cfg, "test_index").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [image_record_from_path(folder / r["file_name"], split="test") for r in rows]


def _benchmark_images(cfg: dict) -> list[ImageRecord]:
    """Imágenes de desarrollo referenciadas por el benchmark, en orden de aparición."""
    folder = data_path(cfg, "dev_images")
    seen: dict[str, None] = {}
    for q in load_questions(cfg):
        seen.setdefault(q["image_file"], None)
    return [image_record_from_path(folder / name, split="benchmark") for name in seen]


def list_images(cfg: dict, split: str) -> list[ImageRecord]:
    """Lista las imágenes de un split: `dev` (120), `test` (60) o `benchmark` (20 de dev)."""
    loaders = {"dev": _dev_images, "test": _test_images, "benchmark": _benchmark_images}
    if split not in loaders:
        raise ValueError(f"Split desconocido {split!r}; opciones: {SPLITS}")
    return loaders[split](cfg)
