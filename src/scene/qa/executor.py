"""Ejecutor determinista del DSL sobre un SceneJSON (E1-22).

Lo usan los dos modos de QA. Todo se calcula con código a partir del JSON; nada se adivina:
- categoría fuera del vocabulario, filtro de color desconocido u op no soportada → `unsupported`;
- pregunta singular sin instancias (posición, tamaño, color, relación) → `insufficient_information`;
- pregunta singular con varias instancias → `ambiguous` (D-10): se responde con la instancia de
  mayor confianza, la evidencia lleva todas y la confianza se reparte entre ellas.

Confianza (PLAN §4.5): `exists` → máx. de las instancias, o 1 − máx. de las descartadas por
filtros (sin ninguna: 1 − umbral del detector); `count` → promedio de las contadas (si son 0,
igual que `exists`); relaciones → producto de las confianzas de los dos objetos.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from scene.qa.dsl import Query, QueryResult, Target
from scene.qa.verbalize import POSITION_LABELS, SIZE_LABELS, verbalize

# Texto → category_id o None (p. ej. `CategoryMatcher.match_id`).
CategoryResolver = Callable[[str], int | None]
EXTRA_PREDICATES = ("near", "on_top_of")  # D-04: van en `extra_relations`


class _NotOk(Exception):
    """Corta la ejecución con un estado distinto de `ok`."""

    def __init__(self, status: str, detail: str, evidence: list[str] | None = None) -> None:
        super().__init__(detail)
        self.status, self.detail, self.evidence = status, detail, evidence or []


@dataclass
class Executor:
    resolve_category: CategoryResolver
    category_names: Mapping[int, str]
    palette: tuple[str, ...]  # nombres de color válidos (scene.color.palette)
    absent_confidence: float  # confianza de "no hay" cuando no queda ninguna instancia
    color_aliases: Mapping[str, str] = field(default_factory=dict)  # "rojo" → "red"
    display_names: Mapping[int, str] = field(default_factory=dict)  # nombre para describir
    describe_max_categories: int = 5

    def execute(self, query: Query, scene: Mapping[str, Any]) -> QueryResult:
        """Resultado validado; la evidencia siempre son ids de `scene['objects']`."""
        handler = getattr(self, f"_op_{query.op}", None)
        try:
            if handler is None or query.op == "unsupported":
                raise _NotOk("unsupported", query.reason or f"op {query.op!r} no soportada")
            normalized, evidence, confidence, status, detail = handler(query, scene)
        except _NotOk as stop:
            normalized, evidence, confidence = stop.status, stop.evidence, 0.0
            status, detail = stop.status, stop.detail
        result = QueryResult(
            answer=status,  # se reemplaza abajo por el texto en lenguaje natural
            normalized_answer=normalized,
            evidence=evidence,
            confidence=round(min(max(confidence, 0.0), 1.0), 4),
            status=status,
            detail=detail,
        )
        return result.model_copy(update={"answer": self._answer(result, query)})

    # --- selección de instancias ---------------------------------------------------------

    def _instances(self, target: Target, scene: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
        """(las que cumplen los filtros, las de la categoría que no), por confianza."""
        cid = self.resolve_category(target.category)
        if cid is None:
            raise _NotOk("unsupported", f"categoría fuera del vocabulario: {target.category!r}")
        color = target.filters.color
        if color is not None:
            color = self.color_aliases.get(color.lower(), color.lower())
            if color not in self.palette:
                raise _NotOk("unsupported", f"color no reconocido: {target.filters.color!r}")
        same = [o for o in scene.get("objects", []) if o.get("category_id") == cid]
        same.sort(key=lambda o: -o["confidence"])
        f = target.filters
        keep = [
            o
            for o in same
            if (f.position is None or o.get("position") == f.position)
            and (f.size is None or o.get("size") == f.size)
            and (color is None or (o.get("attributes") or {}).get("color") == color)
        ]
        kept = {o["id"] for o in keep}
        return keep, [o for o in same if o["id"] not in kept]

    def _single(self, target: Target, scene: Mapping[str, Any], exclude: str | None = None):
        """La instancia a la que se refiere una pregunta singular, y el estado/detalle."""
        found, _ = self._instances(target, scene)
        found = [o for o in found if o["id"] != exclude]
        if not found:
            raise _NotOk("insufficient_information", f"no se detectó {target.category!r}")
        top = found[0]
        if len(found) == 1:
            return top, [top["id"]], top["confidence"], "ok", None
        # D-10: la de mayor confianza, con su parte de la confianza total.
        share = top["confidence"] / sum(o["confidence"] for o in found)
        detail = f"{len(found)} instancias de {target.category!r}; se usó la de mayor confianza"
        return top, [o["id"] for o in found], top["confidence"] * share, "ambiguous", detail

    def _absent(self, rejected: list[dict]) -> float:
        return 1.0 - rejected[0]["confidence"] if rejected else self.absent_confidence

    # --- ops ------------------------------------------------------------------------------

    def _op_exists(self, query: Query, scene: Mapping[str, Any]):
        found, rejected = self._instances(query.target, scene)
        if found:
            return "yes", [o["id"] for o in found], found[0]["confidence"], "ok", None
        return "no", [], self._absent(rejected), "ok", None

    def _op_count(self, query: Query, scene: Mapping[str, Any]):
        found, rejected = self._instances(query.target, scene)
        if not found:
            return 0, [], self._absent(rejected), "ok", None
        mean = sum(o["confidence"] for o in found) / len(found)
        return len(found), [o["id"] for o in found], mean, "ok", None

    def _op_position(self, query: Query, scene: Mapping[str, Any]):
        top, evidence, conf, status, detail = self._single(query.target, scene)
        return top["position"], evidence, conf, status, detail

    def _op_size(self, query: Query, scene: Mapping[str, Any]):
        top, evidence, conf, status, detail = self._single(query.target, scene)
        return top["size"], evidence, conf, status, detail

    def _op_attribute(self, query: Query, scene: Mapping[str, Any]):
        top, evidence, conf, status, detail = self._single(query.target, scene)
        attrs = top.get("attributes") or {}
        if attrs.get("color") is None:
            raise _NotOk("insufficient_information", "color desconocido (máscara muy chica)")
        return (
            attrs["color"],
            evidence,
            conf * (attrs.get("color_confidence") or 1.0),
            status,
            detail,
        )

    def _op_relation(self, query: Query, scene: Mapping[str, Any]):
        subj, ev1, c1, st1, d1 = self._single(query.target, scene)
        obj, ev2, c2, st2, d2 = self._single(query.target2, scene, exclude=subj["id"])
        field_name = "extra_relations" if query.relation in EXTRA_PREDICATES else "relations"
        holds = any(
            r.get("subject") == subj["id"]
            and r.get("object") == obj["id"]
            and r.get("predicate") == query.relation
            for r in scene.get(field_name, [])
        )
        evidence = [subj["id"], obj["id"]] + [
            i for i in ev1 + ev2 if i not in (subj["id"], obj["id"])
        ]
        ambiguous = "ambiguous" in (st1, st2)
        detail = "; ".join(d for d in (d1, d2) if d) or None
        return (
            "yes" if holds else "no",
            evidence,
            c1 * c2,
            "ambiguous" if ambiguous else "ok",
            detail,
        )

    def _op_describe(self, query: Query, scene: Mapping[str, Any]):
        objects = sorted(scene.get("objects", []), key=lambda o: -o["confidence"])
        if not objects:
            return "No se detectaron objetos en la imagen.", [], self.absent_confidence, "ok", None
        counts = Counter(o["category_id"] for o in objects)
        top_cats = [cid for cid, _ in counts.most_common(self.describe_max_categories)]
        parts = [f"{self._display(cid)} ({counts[cid]})" for cid in top_cats]
        main = objects[0]
        n_obj = f"{len(objects)} objeto" + ("s" if len(objects) > 1 else "")
        n_cat = f"{len(counts)} categoría" + ("s" if len(counts) > 1 else "")
        text = (
            f"Se {'detectaron' if len(objects) > 1 else 'detectó'} {n_obj} de {n_cat}. "
            f"Las más frecuentes: {', '.join(parts)}. "
            f"El objeto más confiable es {self._display(main['category_id'])}, "
            f"{POSITION_LABELS[main['position']]} y de tamaño {SIZE_LABELS[main['size']]}."
        )
        used = [o for o in objects if o["category_id"] in top_cats]
        mean = sum(o["confidence"] for o in used) / len(used)
        return text, [o["id"] for o in used], mean, "ok", None

    # --- texto ----------------------------------------------------------------------------

    def _display(self, cid: int) -> str:
        return self.display_names.get(cid) or self.category_names.get(cid, str(cid))

    def _answer(self, result: QueryResult, query: Query) -> str:
        if query.op == "describe" and result.status == "ok":
            return str(result.normalized_answer)
        # Las plantillas nombran la categoría: se usa el nombre para mostrar (en español).
        cid = self.resolve_category(query.target.category) if query.target else None
        if cid is not None:
            target = query.target.model_copy(update={"category": self._display(cid)})
            query = query.model_copy(update={"target": target})
        if result.status == "ambiguous":
            text = verbalize(result.model_copy(update={"status": "ok"}), query)
            return f"{text} Hay varias instancias; se usó la de mayor confianza."
        return verbalize(result, query)


def executor_from_config(
    cfg: dict,
    resolve_category: CategoryResolver,
    category_names: Mapping[int, str],
    display_names: Mapping[int, str] | None = None,
) -> Executor:
    """Ejecutor según `cfg['qa']['executor']`, la paleta de color y el umbral del detector."""
    ecfg = cfg["qa"]["executor"]
    return Executor(
        resolve_category=resolve_category,
        category_names=category_names,
        palette=tuple(cfg["scene"]["color"]["palette"]),
        absent_confidence=1.0 - cfg["detection"]["score_threshold"],
        color_aliases=ecfg.get("color_aliases") or {},
        display_names=display_names or {},
        describe_max_categories=ecfg["describe_max_categories"],
    )
