"""Descripción textual de un SceneJSON y verificación de fidelidad."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

from scene.llm.model import TextGenerator

NUMBER_WORDS = {
    "cero": 0,
    "uno": 1,
    "una": 1,
    "un": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
}


@dataclass(frozen=True)
class FidelityReport:
    """Afirmaciones comprobables encontradas en una descripción."""

    faithful: bool
    mentioned_categories: tuple[str, ...]
    unsupported_categories: tuple[str, ...]
    mentioned_numbers: tuple[int | float, ...]
    unsupported_numbers: tuple[int | float, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DescriptionResult:
    """Texto generado junto con el resumen fuente y su verificación."""

    text: str
    summary: str
    fidelity: FidelityReport
    attempts: int
    status: str


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _object_category(obj: Mapping[str, Any], names: Mapping[int, str]) -> str:
    category = obj.get("category_name")
    if isinstance(category, str) and category:
        return category
    category_id = obj.get("category_id")
    return names.get(category_id, f"category_{category_id}")


def _object_color(obj: Mapping[str, Any]) -> str | None:
    for key in ("color", "dominant_color"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and isinstance(value.get("value"), str):
            return value["value"]
    attributes = obj.get("attributes")
    if isinstance(attributes, dict):
        color = attributes.get("color")
        if isinstance(color, str):
            return color
        if isinstance(color, dict) and isinstance(color.get("value"), str):
            return color["value"]
    return None


def build_scene_summary(
    scene: Mapping[str, Any],
    *,
    category_names: Mapping[int, str],
    max_objects: int,
    max_relations: int,
) -> str:
    """Construye el único contenido de escena que recibe el LLM."""
    objects = list(scene.get("objects", []))
    relations = list(scene.get("relations", []))
    limited_objects = objects[:max_objects]
    all_names = [_object_category(obj, category_names) for obj in objects]
    object_names = {
        str(obj.get("id")): _object_category(obj, category_names) for obj in limited_objects
    }
    counts = Counter(all_names)
    lines = [f"Total de objetos: {len(objects)}", "Conteos por categoría:"]
    if counts:
        lines.extend(f"- {name}: {count}" for name, count in sorted(counts.items()))
    else:
        lines.append("- ninguno")

    lines.append("Objetos:")
    for obj in limited_objects:
        obj_id = str(obj.get("id"))
        fields = [
            f"id={obj_id}",
            f"categoría={object_names[obj_id]}",
            f"posición={obj.get('position', 'desconocida')}",
            f"tamaño={obj.get('size', 'desconocido')}",
        ]
        color = _object_color(obj)
        if color:
            fields.append(f"color={color}")
        lines.append("- " + "; ".join(fields))

    lines.append("Relaciones:")
    relation_count = 0
    for relation in relations:
        subject = str(relation.get("subject"))
        target = str(relation.get("object"))
        if subject not in object_names or target not in object_names:
            continue
        lines.append(
            f"- {object_names[subject]} ({subject}) {relation.get('predicate')} "
            f"{object_names[target]} ({target})"
        )
        relation_count += 1
        if relation_count >= max_relations:
            break
    if relation_count == 0:
        lines.append("- ninguna")
    return "\n".join(lines)


def _category_variants(name: str) -> tuple[str, ...]:
    normalized = _normalized(name)
    variants = {normalized, f"{normalized}s", f"{normalized}es"}
    return tuple(sorted(variants, key=len, reverse=True))


def _mentioned_categories(text: str, vocabulary: Iterable[str]) -> set[str]:
    normalized = _normalized(text)
    found: set[str] = set()
    for name in sorted(set(vocabulary), key=len, reverse=True):
        for variant in _category_variants(name):
            if re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized):
                found.add(name)
                break
    return found


def _mentioned_numbers(text: str) -> list[int | float]:
    normalized = _normalized(text)
    values: list[int | float] = []
    for raw in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", normalized):
        value = float(raw.replace(",", "."))
        values.append(int(value) if value.is_integer() else value)
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"(?<!\w){word}(?!\w)", normalized):
            values.append(value)
    return values


def verify_description(
    description: str,
    scene: Mapping[str, Any],
    *,
    category_names: Mapping[int, str],
) -> FidelityReport:
    """Comprueba categorías del vocabulario y números contra el SceneJSON."""
    scene_categories = [_object_category(obj, category_names) for obj in scene.get("objects", [])]
    allowed_categories = set(scene_categories)
    vocabulary = set(category_names.values()) | allowed_categories
    mentioned_categories = _mentioned_categories(description, vocabulary)
    unsupported_categories = mentioned_categories - allowed_categories

    counts = Counter(scene_categories)
    allowed_numbers = {len(scene_categories), *counts.values()}
    mentioned_numbers = _mentioned_numbers(description)
    unsupported_numbers = [value for value in mentioned_numbers if value not in allowed_numbers]

    reasons: list[str] = []
    if not description.strip():
        reasons.append("descripción vacía")
    if unsupported_categories:
        reasons.append("categorías no presentes: " + ", ".join(sorted(unsupported_categories)))
    if unsupported_numbers:
        reasons.append("números no sustentados: " + ", ".join(map(str, unsupported_numbers)))
    return FidelityReport(
        faithful=not reasons,
        mentioned_categories=tuple(sorted(mentioned_categories)),
        unsupported_categories=tuple(sorted(unsupported_categories)),
        mentioned_numbers=tuple(mentioned_numbers),
        unsupported_numbers=tuple(unsupported_numbers),
        reasons=tuple(reasons),
    )


class SceneDescriber:
    """Genera una descripción sin exponer la imagen al modelo de lenguaje."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        category_names: Mapping[int, str],
        prompt_version: str = "describer_v1.txt",
        max_new_tokens: int,
        max_objects: int,
        max_relations: int,
        retry_on_unfaithful: int,
    ) -> None:
        self.generator = generator
        self.category_names = dict(category_names)
        self.prompt_template = (
            resources.files("scene.llm")
            .joinpath("prompts", prompt_version)
            .read_text(encoding="utf-8")
        )
        self.max_new_tokens = max_new_tokens
        self.max_objects = max_objects
        self.max_relations = max_relations
        self.retry_on_unfaithful = retry_on_unfaithful

    def describe(self, scene: Mapping[str, Any]) -> DescriptionResult:
        """Genera y, si hace falta, reintenta indicando las afirmaciones inválidas."""
        summary = build_scene_summary(
            scene,
            category_names=self.category_names,
            max_objects=self.max_objects,
            max_relations=self.max_relations,
        )
        base_prompt = self.prompt_template.replace("{{SCENE_SUMMARY}}", summary)
        prompt = base_prompt
        text = ""
        report = verify_description(text, scene, category_names=self.category_names)
        attempts = 1 + self.retry_on_unfaithful
        for attempt in range(1, attempts + 1):
            text = self.generator.generate(prompt, max_new_tokens=self.max_new_tokens).strip()
            report = verify_description(text, scene, category_names=self.category_names)
            if report.faithful:
                return DescriptionResult(text, summary, report, attempt, "ok")
            prompt = (
                base_prompt
                + "\n\nLa descripción anterior no fue fiel: "
                + "; ".join(report.reasons)
                + f"\nDescripción anterior: {text}\nCorregila usando solo el resumen."
            )
        return DescriptionResult(text, summary, report, attempts, "unfaithful")


def build_describer(
    cfg: dict,
    generator: TextGenerator,
    *,
    category_names: Mapping[int, str],
) -> SceneDescriber:
    """Construye el descriptor desde `llm.describer`."""
    describer_cfg = cfg["llm"]["describer"]
    return SceneDescriber(
        generator,
        category_names=category_names,
        prompt_version=describer_cfg["prompt_version"],
        max_new_tokens=describer_cfg["max_new_tokens"],
        max_objects=describer_cfg["max_objects"],
        max_relations=describer_cfg["max_relations"],
        retry_on_unfaithful=describer_cfg["retry_on_unfaithful"],
    )
