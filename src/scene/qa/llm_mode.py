"""Modo con LLM (E1-28): texto de la pregunta → LLM → DSL validado → mismo ejecutor determinista.

El LLM recibe **solo** el texto de la pregunta (nunca `type`, `class_en` ni la imagen) y su única
salida es el JSON intermedio. Los conteos, filtros y comparaciones los resuelve el ejecutor.
Las consignas de descripción (`op="describe"`) las redacta el descriptor a partir de un resumen
del SceneJSON y se verifican; si la descripción no es fiel, se usa la plantilla determinista
(una respuesta incompleta pero sustentada es preferible a una fluida inventada).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scene.llm.describer import SceneDescriber
from scene.llm.interpreter import InterpretationResult, Interpreter
from scene.qa.dsl import Query, QueryResult
from scene.qa.executor import Executor


def interpretation_to_query(interp: InterpretationResult) -> Query:
    """DSL que se ejecuta: el del LLM, o `op="unsupported"` con el motivo si no lo produjo."""
    if interp.status == "ok" and interp.query is not None:
        return interp.query
    reason = "; ".join(interp.errors) or "el intérprete no produjo una consulta válida"
    return Query(op="unsupported", reason=f"intérprete LLM: {reason}"[:500])


def _description_evidence(
    scene: Mapping[str, Any], mentioned: tuple[str, ...], names: Mapping[int, str]
) -> list[str]:
    """Ids de los objetos de las categorías que menciona la descripción (de mayor confianza)."""
    wanted = {m.lower() for m in mentioned}
    objs = sorted(scene.get("objects", []), key=lambda o: -o["confidence"])
    return [
        o["id"]
        for o in objs
        if (o.get("category_name") or names.get(o["category_id"], "")).lower() in wanted
    ]


def describe_result(
    scene: Mapping[str, Any],
    describer: SceneDescriber,
    executor: Executor,
    query: Query,
    names: Mapping[int, str],
) -> tuple[QueryResult, dict[str, Any]]:
    """Descripción con LLM verificada; si no es fiel, la plantilla determinista del ejecutor."""
    desc = describer.describe(scene)
    trace = {
        "describer_status": desc.status,
        "describer_attempts": desc.attempts,
        "fidelity_reasons": list(desc.fidelity.reasons),
    }
    if desc.status != "ok" or not desc.text:
        fallback = executor.execute(query, scene)
        detail = "descripción LLM no fiel; se usó la plantilla determinista"
        return fallback.model_copy(update={"detail": detail}), trace
    evidence = _description_evidence(scene, desc.fidelity.mentioned_categories, names)
    objs = {o["id"]: o for o in scene.get("objects", [])}
    conf = sum(objs[i]["confidence"] for i in evidence) / len(evidence) if evidence else 0.0
    result = QueryResult(
        answer=desc.text,
        normalized_answer=desc.text,
        evidence=evidence,
        confidence=round(conf, 4),
        status="ok",
    )
    return result, trace


def answer_questions_llm(
    questions: list[Mapping[str, Any]],
    scene: Mapping[str, Any],
    interpreter: Interpreter,
    executor: Executor,
    describer: SceneDescriber,
    category_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Items de `questions_llm`, con `interpreted_query` y la traza del intérprete."""
    items = []
    for q in questions:
        interp = interpreter.interpret(q["question"])  # solo el texto
        query = interpretation_to_query(interp)
        extra: dict[str, Any] = {}
        if query.op == "describe":
            result, extra = describe_result(scene, describer, executor, query, category_names)
        else:
            result = executor.execute(query, scene)
        item = result.to_question_answer(q["question_id"], q["type"], q["question"], query)
        item["interpreter"] = {
            "status": interp.status,
            "attempts": interp.attempts,
            "errors": list(interp.errors),
            "raw_output": interp.raw_output,
            **extra,
        }
        items.append(item)
    return items
