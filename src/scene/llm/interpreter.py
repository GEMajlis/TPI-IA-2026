"""Interpretación de preguntas en texto al DSL validado de consultas."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

from pydantic import ValidationError

from scene.llm.model import TextGenerator

CategoryMatcher = Callable[[str], int | None]


@dataclass(frozen=True)
class InterpretationResult:
    """Resultado del intérprete; `query` solo existe cuando el estado es `ok`."""

    status: str
    query: Any | None
    attempts: int
    raw_output: str
    errors: tuple[str, ...] = ()

    @property
    def interpreted_query(self) -> dict[str, Any]:
        """Versión serializable para `questions_llm[].interpreted_query`."""
        if self.query is None:
            return {"status": self.status, "errors": list(self.errors)}
        if hasattr(self.query, "model_dump"):
            return self.query.model_dump(mode="json")
        if isinstance(self.query, dict):
            return self.query
        raise TypeError("El modelo DSL no se puede serializar")


def _load_prompt(name: str) -> str:
    prompt = resources.files("scene.llm").joinpath("prompts", name)
    return prompt.read_text(encoding="utf-8")


def _first_json_object(text: str) -> dict[str, Any]:
    """Extrae el primer objeto JSON aun si el modelo agregó fences o texto."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("la salida no contiene un objeto JSON válido")


def _default_query_model() -> type:
    try:
        from scene.qa.dsl import Query
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "E1-20 todavía no está disponible: pase query_model al Interpreter"
        ) from exc
    return Query


class Interpreter:
    """LLM -> JSON -> resolución de categorías -> validación pydantic."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        category_matcher: CategoryMatcher,
        category_names: Mapping[int, str],
        query_model: type | None = None,
        prompt_version: str = "interpreter_v1.txt",
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser al menos 1")
        self.generator = generator
        self.category_matcher = category_matcher
        self.category_names = dict(category_names)
        self.query_model = query_model or _default_query_model()
        self.prompt_template = _load_prompt(prompt_version)
        self.max_attempts = max_attempts

    def _prompt(self, question: str, previous: str | None, error: str | None) -> str:
        prompt = self.prompt_template.replace("{{QUESTION}}", question)
        if previous is not None:
            prompt += (
                "\n\nLa salida anterior fue inválida. Corregila y devolvé solo JSON."
                f"\nError de validación: {error}\nSalida anterior: {previous}"
            )
        return prompt

    def _canonical_category(self, mention: Any) -> str:
        if not isinstance(mention, str) or not mention.strip():
            raise ValueError("target.category debe ser texto no vacío")
        category_id = self.category_matcher(mention)
        if category_id is None:
            raise ValueError(f"categoría fuera del vocabulario: {mention!r}")
        if category_id not in self.category_names:
            raise ValueError(f"category matcher devolvió un id desconocido: {category_id}")
        return self.category_names[category_id]

    def _resolve_categories(self, payload: dict[str, Any]) -> dict[str, Any]:
        """E1-21 decide el id; el LLM solamente copia la mención de la pregunta."""
        resolved = json.loads(json.dumps(payload))
        for key in ("target", "target2"):
            target = resolved.get(key)
            if target is None:
                continue
            if not isinstance(target, dict):
                raise ValueError(f"{key} debe ser un objeto o null")
            if target.get("category") is not None:
                target["category"] = self._canonical_category(target["category"])
        return resolved

    def _validate(self, raw: str) -> Any:
        payload = _first_json_object(raw)
        if "unsupported_reason" in payload:
            raise UnsupportedQuery(str(payload["unsupported_reason"]))
        payload = self._resolve_categories(payload)
        validator = getattr(self.query_model, "model_validate", None)
        if validator is None:
            raise TypeError("query_model debe ser un modelo pydantic v2")
        return validator(payload)

    def interpret(self, question: str) -> InterpretationResult:
        """Interpreta una pregunta; reintenta una vez por defecto ante salida inválida."""
        if not question.strip():
            return InterpretationResult("unsupported", None, 0, "", ("pregunta vacía",))

        errors: list[str] = []
        previous: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw = self.generator.generate(
                self._prompt(question, previous, errors[-1] if errors else None)
            )
            try:
                query = self._validate(raw)
            except UnsupportedQuery as exc:
                return InterpretationResult("unsupported", None, attempt, raw, (str(exc),))
            except (TypeError, ValueError, ValidationError) as exc:
                errors.append(str(exc))
                previous = raw
                continue
            return InterpretationResult("ok", query, attempt, raw, tuple(errors))
        return InterpretationResult(
            "unsupported", None, self.max_attempts, previous or "", tuple(errors)
        )


class UnsupportedQuery(ValueError):
    """El propio modelo marcó la consulta como no soportada."""


def build_interpreter(
    cfg: dict,
    generator: TextGenerator,
    *,
    category_matcher: CategoryMatcher,
    category_names: Mapping[int, str],
    query_model: type | None = None,
) -> Interpreter:
    """Construye el intérprete desde `llm.interpreter`."""
    interpreter_cfg = cfg["llm"]["interpreter"]
    return Interpreter(
        generator,
        category_matcher=category_matcher,
        category_names=category_names,
        query_model=query_model,
        prompt_version=interpreter_cfg["prompt_version"],
        max_attempts=interpreter_cfg["max_attempts"],
    )
