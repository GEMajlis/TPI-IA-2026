"""Plantillas deterministas para verbalizar resultados del ejecutor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

POSITION_LABELS = {
    "top_left": "arriba a la izquierda",
    "top_center": "arriba en el centro",
    "top_right": "arriba a la derecha",
    "middle_left": "en el medio a la izquierda",
    "center": "en el centro",
    "middle_right": "en el medio a la derecha",
    "bottom_left": "abajo a la izquierda",
    "bottom_center": "abajo en el centro",
    "bottom_right": "abajo a la derecha",
}
SIZE_LABELS = {"small": "pequeño", "medium": "mediano", "large": "grande"}


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _target_name(query: Any | None) -> str:
    target = _value(query, "target") if query is not None else None
    category = _value(target, "category") if target is not None else None
    return str(category) if category else "esa categoría"


def verbalize(result: Any, query: Any | None = None) -> str:
    """Convierte un resultado estructurado en una respuesta breve en español."""
    status = _value(result, "status", "ok")
    if status == "unsupported":
        return "La pregunta no está soportada."
    if status == "insufficient_information":
        return "No hay información suficiente para responder."
    if status == "ambiguous":
        return "La referencia es ambigua porque hay varias instancias posibles."

    normalized = _value(result, "normalized_answer", _value(result, "answer"))
    operation = _value(query, "op") if query is not None else None
    category = _target_name(query)
    if operation == "count" or isinstance(normalized, int):
        noun = "instancia" if normalized == 1 else "instancias"
        return f"Hay {normalized} {noun} de {category}."
    if operation in {"exists", "relation"} or normalized in {"yes", "no"}:
        return "Sí." if normalized == "yes" else "No."
    if operation == "position" or normalized in POSITION_LABELS:
        label = POSITION_LABELS.get(normalized, str(normalized))
        return f"La instancia está {label}."
    if operation == "size" or normalized in SIZE_LABELS:
        label = SIZE_LABELS.get(normalized, str(normalized))
        return f"La instancia tiene tamaño {label}."
    answer = _value(result, "answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return str(normalized)
