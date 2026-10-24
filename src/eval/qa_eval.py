"""Evaluación de QA (E1-24), para el modo determinista (`questions`) y el LLM (`questions_llm`).

- **Fidelidad:** se vuelve a ejecutar el `interpreted_query` de cada respuesta sobre el mismo
  SceneJSON (tiene que dar lo mismo) y se verifica que la evidencia sostenga la respuesta.
- **Exactitud:** el benchmark no trae respuestas, así que la referencia es la respuesta del mismo
  ejecutor, con la consulta estructurada, sobre el SceneJSON oráculo (`eval/oracle.py`). Mide si
  la pipeline llega a la misma respuesta que con una percepción perfecta, bajo nuestras reglas
  de posición, tamaño y relaciones (no las de la cátedra, que no conocemos).
- **Atribución de errores:** en modo determinista la consulta y el ejecutor son los mismos que en
  el oráculo, así que una diferencia es de percepción. En modo LLM, si el DSL interpretado difiere
  del estructurado el error es de interpretación; si coincide, de percepción.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from scene.qa.dsl import Query
from scene.qa.executor import Executor
from scene.qa.structured_adapter import query_from_benchmark

NON_OK = ("unsupported", "insufficient_information")


def _relations(scene: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    rels = list(scene.get("relations", [])) + list(scene.get("extra_relations", []))
    return {(r["subject"], r["predicate"], r["object"]) for r in rels}


def evidence_supports(item: Mapping[str, Any], scene: Mapping[str, Any]) -> tuple[bool, str]:
    """¿La evidencia sostiene `normalized_answer`? Devuelve (ok, motivo si no)."""
    objs = {o["id"]: o for o in scene.get("objects", [])}
    ev = list(item.get("evidence", []))
    if any(e not in objs for e in ev):
        return False, "evidencia con ids inexistentes"
    if item.get("status") in NON_OK:
        return True, ""
    q = item["interpreted_query"]
    op, ans = q.get("op"), item["normalized_answer"]
    if op == "count" and ans != len(ev):
        return False, f"cuenta {ans} con {len(ev)} objetos de evidencia"
    if op == "exists" and (ans == "yes") != bool(ev):
        return False, f"existencia {ans!r} con {len(ev)} objetos de evidencia"
    if op in ("position", "size"):
        if not ev or objs[ev[0]][op] != ans:
            return False, f"{op} {ans!r} no coincide con el objeto de evidencia"
    if op == "relation":
        if len(ev) < 2:
            return False, "relación con menos de 2 objetos de evidencia"
        holds = (ev[0], q["relation"], ev[1]) in _relations(scene)
        if holds != (ans == "yes"):
            return False, f"relación {ans!r} pero la tripla {'sí' if holds else 'no'} está"
    return True, ""


def reexecutes(item: Mapping[str, Any], scene: Mapping[str, Any], executor: Executor) -> bool:
    """Volver a ejecutar el DSL guardado da la misma respuesta (determinismo y trazabilidad)."""
    query = Query.model_validate(item["interpreted_query"])
    return executor.execute(query, scene).normalized_answer == item["normalized_answer"]


def _dsl_key(q: Query) -> tuple:
    """Lo que tiene que coincidir entre el DSL del LLM y el estructurado."""
    t = [
        (x.category, tuple(sorted(x.filters.active().items()))) for x in (q.target, q.target2) if x
    ]
    return (q.op, tuple(t), q.relation, q.attribute)


def evaluate_mode(
    mode: str,
    benchmark: list[Mapping[str, Any]],
    pred_scenes: Mapping[str, Mapping[str, Any]],
    oracle_scenes: Mapping[str, Mapping[str, Any]],
    executor: Executor,
    phrases: Mapping[str, str],
    resolve: Any,
) -> pd.DataFrame:
    """Una fila por consigna con respuesta, referencia oráculo, fidelidad y atribución.

    `resolve(texto) -> category_id` normaliza las categorías para comparar los DSL.
    """
    field = "questions" if mode == "deterministic" else "questions_llm"
    rows = []
    for q in benchmark:
        iid = q["image_file"].rsplit(".", 1)[0]
        scene = pred_scenes.get(iid)
        items = {it["question_id"]: it for it in (scene or {}).get(field) or []}
        item = items.get(q["question_id"])
        if item is None:
            continue
        structured = query_from_benchmark(q, phrases)
        ref = executor.execute(structured, oracle_scenes[iid])
        row: dict[str, Any] = {
            "mode": mode,
            "image_id": iid,
            "question_id": q["question_id"],
            "type": q["type"],
            "answer": item["normalized_answer"],
            "status": item.get("status", "ok"),
            "oracle_answer": ref.normalized_answer,
            "oracle_status": ref.status,
            "confidence": item["confidence"],
        }
        supported, why = evidence_supports(item, scene)
        row["evidence_supports"] = supported
        row["evidence_issue"] = why
        # La descripción del LLM es texto generado: re-ejecutar su DSL da la plantilla, no aplica.
        llm_text = mode == "llm" and q["type"] == "description"
        row["reexecutes"] = None if llm_text else reexecutes(item, scene, executor)
        llm_q = Query.model_validate(item["interpreted_query"])
        if mode == "llm":
            row["interpreter_status"] = item.get("interpreter", {}).get("status")

            def canon(query: Query) -> tuple:
                key = _dsl_key(query)
                cats = tuple((resolve(c), f) for c, f in key[1])
                return (key[0], cats, key[2], key[3])

            row["dsl_match"] = canon(llm_q) == canon(structured)
        if q["type"] == "description":
            row["correct"] = None  # texto libre: se mide con la fidelidad del descriptor
            row["error_source"] = None
        else:
            row["correct"] = item["normalized_answer"] == ref.normalized_answer
            if row["correct"]:
                row["error_source"] = None
            elif ref.status != "ok":
                row["error_source"] = "oracle_not_ok"  # ni con percepción perfecta se responde
            elif mode == "llm" and not row["dsl_match"]:
                row["error_source"] = "interpretation"
            else:
                row["error_source"] = "perception"
        rows.append(row)
    return pd.DataFrame(rows)


def _rate(s: pd.Series) -> float | None:
    """Proporción de True ignorando los casos donde no aplica (None)."""
    s = s.dropna()
    return float(s.astype(bool).mean()) if len(s) else None


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Exactitud por modo y tipo, estados, fidelidad, atribución y métricas del intérprete."""
    out: dict[str, Any] = {}
    for mode, g in df.groupby("mode"):
        closed = g[g["type"] != "description"]
        m: dict[str, Any] = {
            "n": len(g),
            "accuracy_vs_oracle": _rate(closed["correct"]),
            "by_type": {
                t: {
                    "n": len(x),
                    "accuracy_vs_oracle": _rate(x["correct"]),
                    "status": x["status"].value_counts().to_dict(),
                }
                for t, x in closed.groupby("type")
            },
            "unsupported_rate": float((g["status"] == "unsupported").mean()),
            "evidence_supports_rate": _rate(g["evidence_supports"]),
            "reexecutes_rate": _rate(g["reexecutes"]),
            "error_source": closed["error_source"].value_counts().to_dict(),
        }
        if mode == "llm":
            m["json_valid_rate"] = float((g["interpreter_status"] == "ok").mean())
            m["dsl_match_rate"] = _rate(g["dsl_match"])
        out[mode] = m
    return out
