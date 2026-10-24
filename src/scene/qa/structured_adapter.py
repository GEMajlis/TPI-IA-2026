"""Modo determinista (E1-23): campos estructurados del benchmark → DSL, sin LLM.

Usa `type` y `class_en` de cada consigna; el predicado de las relaciones no viene estructurado,
así que se busca en el texto con las frases de `qa.adapter.relation_phrases`. Las respuestas
van a `SceneJSON.questions` conservando `question_id`, `type` y `question`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scene.qa.category_matcher import normalize_text
from scene.qa.dsl import QUESTION_TYPE_TO_OP, Query
from scene.qa.executor import Executor


def relation_from_text(question: str, phrases: Mapping[str, str]) -> str | None:
    """Predicado de la frase más larga que aparece en la pregunta (sin acentos ni mayúsculas)."""
    text = f" {normalize_text(question)} "
    for phrase in sorted(phrases, key=len, reverse=True):
        if f" {normalize_text(phrase)} " in text:
            return phrases[phrase]
    return None


def query_from_benchmark(question: Mapping[str, Any], phrases: Mapping[str, str]) -> Query:
    """DSL de una consigna; si no se puede traducir, `op="unsupported"` con el motivo."""
    op = QUESTION_TYPE_TO_OP.get(question["type"])
    classes = list(question.get("class_en") or [])
    if op is None:
        return Query(op="unsupported", reason=f"tipo de pregunta desconocido {question['type']!r}")
    if op == "describe":
        return Query(op="describe")
    if op == "relation":
        relation = relation_from_text(question["question"], phrases)
        if relation is None or len(classes) != 2:
            return Query(op="unsupported", reason="no se reconoce la relación o faltan clases")
        return Query(
            op="relation",
            target={"category": classes[0]},
            target2={"category": classes[1]},
            relation=relation,
        )
    if len(classes) != 1:
        return Query(op="unsupported", reason=f"se esperaba 1 clase y hay {len(classes)}")
    return Query(op=op, target={"category": classes[0]})


def benchmark_image_id(question: Mapping[str, Any]) -> str:
    """`tpi_0001.jpg` → `tpi_0001` (el `image_id` de la fila del submission)."""
    return Path(question["image_file"]).stem


def answer_questions(
    questions: list[Mapping[str, Any]],
    scene: Mapping[str, Any],
    executor: Executor,
    phrases: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Items de `questions` (con `interpreted_query` propio para trazabilidad)."""
    items = []
    for q in questions:
        query = query_from_benchmark(q, phrases)
        result = executor.execute(query, scene)
        item = result.to_question_answer(q["question_id"], q["type"], q["question"])
        item["interpreted_query"] = query.model_dump(mode="json")
        items.append(item)
    return items


def group_by_image(questions: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Consignas por `image_id`, en el orden del benchmark."""
    out: dict[str, list[Mapping[str, Any]]] = {}
    for q in questions:
        out.setdefault(benchmark_image_id(q), []).append(q)
    return out
