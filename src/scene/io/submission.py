"""Armado y validación de `submission.csv`: columnas `image_id,SceneJSON`, mismo orden e ids
que `sample_submission.csv` (60 de test + 20 del benchmark)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from scene.config import data_path
from scene.scene.schema import validate_scene

COLUMNS = ["image_id", "SceneJSON"]


def sample_ids(cfg: dict) -> list[str]:
    """Ids de `sample_submission.csv`, en orden."""
    with data_path(cfg, "sample_submission").open(encoding="utf-8", newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def build_submission(
    scenes: dict[str, dict[str, Any]], cfg: dict
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Filas del CSV y errores por image_id (falta la escena o no valida)."""
    schema = data_path(cfg, "scene_schema")
    rows: list[tuple[str, str]] = []
    errors: dict[str, list[str]] = {}
    for iid in sample_ids(cfg):
        scene = scenes.get(iid)
        if scene is None:
            errors[iid] = ["falta el SceneJSON de esta imagen"]
            continue
        if scene["image"]["id"] != iid:
            errors[iid] = [f"image.id {scene['image']['id']!r} no coincide con la fila"]
            continue
        errs = validate_scene(scene, schema)
        if errs:
            errors[iid] = errs
            continue
        rows.append((iid, json.dumps(scene, separators=(",", ":"), ensure_ascii=False)))
    return rows, errors


def write_submission(rows: list[tuple[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Las celdas pueden pasar el límite por defecto del módulo csv al leerlas.
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


def read_submission(path: str | Path) -> dict[str, dict[str, Any]]:
    """Lee un submission y parsea cada SceneJSON (para re-validarlo)."""
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Columnas {reader.fieldnames}, se esperaba {COLUMNS}")
        return {row["image_id"]: json.loads(row["SceneJSON"]) for row in reader}
