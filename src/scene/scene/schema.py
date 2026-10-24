"""Validación del SceneJSON contra `scene_schema.json` (normativo) y chequeos estructurales."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

# Enums del schema. Un test verifica que coincidan con `scene_schema.json`.
POSITIONS = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)
SIZES = ("small", "medium", "large")
RELATION_PREDICATES = ("left_of", "right_of", "above", "below", "overlaps")
QUESTION_TYPES = ("description", "count", "existence", "position", "size", "relation")


class SceneValidationError(ValueError):
    """El SceneJSON no cumple el schema o tiene inconsistencias estructurales."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors[:5]) + (" ..." if len(errors) > 5 else ""))
        self.errors = errors


@lru_cache(maxsize=4)
def _validator(schema_path: str) -> Draft202012Validator:
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def schema_errors(scene: dict, schema_path: str | Path) -> list[str]:
    """Errores de validación jsonschema, con la ruta del campo."""
    validator = _validator(str(schema_path))
    errors = sorted(validator.iter_errors(scene), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<raíz>'}: {e.message}" for e in errors]


def structural_errors(scene: dict) -> list[str]:
    """Chequeos que el schema no expresa: ids únicos, tamaño de máscara, referencias válidas."""
    errors: list[str] = []
    image = scene.get("image", {})
    hw = [image.get("height"), image.get("width")]
    objects = scene.get("objects", [])
    ids = [o.get("id") for o in objects]
    if len(ids) != len(set(ids)):
        errors.append("objects: hay ids repetidos")
    valid_ids = set(ids)
    for i, obj in enumerate(objects):
        mask = obj.get("mask", {})
        if mask.get("size") != hw:
            errors.append(f"objects/{i}/mask/size: {mask.get('size')} != [alto, ancho] {hw}")
        elif sum(mask.get("counts", [])) != hw[0] * hw[1]:
            errors.append(f"objects/{i}/mask/counts: la suma no coincide con alto*ancho")
    for i, rel in enumerate(scene.get("relations", [])):
        for key in ("subject", "object"):
            if rel.get(key) not in valid_ids:
                errors.append(f"relations/{i}/{key}: id inexistente {rel.get(key)!r}")
        if rel.get("subject") == rel.get("object"):
            errors.append(f"relations/{i}: sujeto y objeto son el mismo id")
    for field in ("questions", "questions_llm"):
        for i, qa in enumerate(scene.get(field, [])):
            for ev in qa.get("evidence", []):
                if ev not in valid_ids:
                    errors.append(f"{field}/{i}/evidence: id inexistente {ev!r}")
    return errors


def validate_scene(scene: dict, schema_path: str | Path, structural: bool = True) -> list[str]:
    """Todos los errores (schema y, opcionalmente, estructurales). Lista vacía = válido."""
    errors = schema_errors(scene, schema_path)
    if structural:
        errors += structural_errors(scene)
    return errors


def assert_valid_scene(scene: dict, schema_path: str | Path, structural: bool = True) -> None:
    """Igual que `validate_scene` pero lanza `SceneValidationError` si hay errores."""
    errors = validate_scene(scene, schema_path, structural=structural)
    if errors:
        raise SceneValidationError(errors)
