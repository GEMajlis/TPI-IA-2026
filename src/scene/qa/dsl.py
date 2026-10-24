"""DSL de consultas (PLAN §3) y resultado del ejecutor.

Lo usan los dos modos de QA: el adapter determinista (campos estructurados) y el intérprete LLM
(solo texto). Las categorías van como texto; las resuelve `qa.category_matcher` en código.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scene.scene.schema import POSITIONS, SIZES

# `Literal[tupla]` equivale a `Literal["a", "b", ...]`: así los enums salen del schema.
Position = Literal[POSITIONS]
Size = Literal[SIZES]
Op = Literal[
    "count", "exists", "position", "size", "relation", "attribute", "describe", "unsupported"
]
Relation = Literal["left_of", "right_of", "above", "below", "near", "overlaps", "on_top_of"]
Attribute = Literal["color"]
OutputType = Literal["int", "yes_no", "region", "size_label", "text"]
Status = Literal["ok", "unsupported", "insufficient_information", "ambiguous"]

# Tipo de salida de cada op (el `output_type` de una consulta tiene que coincidir).
OUTPUT_TYPE_BY_OP: dict[str, str] = {
    "count": "int",
    "exists": "yes_no",
    "position": "region",
    "size": "size_label",
    "relation": "yes_no",
    "attribute": "text",
    "describe": "text",
    "unsupported": "text",
}
# Tipo de pregunta del benchmark → op, y `answer_format` → output_type.
QUESTION_TYPE_TO_OP: dict[str, str] = {
    "count": "count",
    "existence": "exists",
    "position": "position",
    "size": "size",
    "relation": "relation",
    "description": "describe",
}
ANSWER_FORMAT_TO_OUTPUT: dict[str, str] = {
    "integer": "int",
    "boolean": "yes_no",
    "position_3x3": "region",
    "size_class": "size_label",
    "free_text": "text",
}


class Filters(BaseModel):
    """Filtros opcionales sobre las instancias de la categoría (`null` = sin filtro)."""

    model_config = ConfigDict(extra="forbid")

    color: str | None = None
    position: Position | None = None
    size: Size | None = None

    def active(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Target(BaseModel):
    """Objeto al que se refiere la pregunta: texto de la categoría más filtros."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    filters: Filters = Field(default_factory=Filters)


class Query(BaseModel):
    """Consulta ejecutable sobre un SceneJSON.

    `relation` y `target2` solo van con `op="relation"`; `attribute` solo con `op="attribute"`.
    `unsupported` lo usa el intérprete cuando la pregunta no entra en ninguna op (con `reason`).
    """

    model_config = ConfigDict(extra="forbid")

    op: Op
    target: Target | None = None
    target2: Target | None = None
    relation: Relation | None = None
    attribute: Attribute | None = None
    output_type: OutputType | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check_op(self) -> Query:
        if self.op in ("describe", "unsupported"):
            if self.target is not None or self.target2 is not None:
                raise ValueError(f"op={self.op!r} no lleva target")
        elif self.target is None:
            raise ValueError(f"op={self.op!r} necesita target")
        if (self.op == "relation") != (self.relation is not None):
            raise ValueError("relation va si y solo si op='relation'")
        if (self.op == "relation") != (self.target2 is not None):
            raise ValueError("target2 va si y solo si op='relation'")
        if (self.op == "attribute") != (self.attribute is not None):
            raise ValueError("attribute va si y solo si op='attribute'")
        expected = OUTPUT_TYPE_BY_OP[self.op]
        if self.output_type is None:
            self.output_type = expected
        elif self.output_type != expected:
            raise ValueError(f"output_type {self.output_type!r} no corresponde a op={self.op!r}")
        return self

    def categories(self) -> list[str]:
        """Textos de categoría en orden [target, target2]."""
        return [t.category for t in (self.target, self.target2) if t is not None]


class QueryResult(BaseModel):
    """Salida del ejecutor. `evidence` son ids de objetos del mismo SceneJSON."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    normalized_answer: int | str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: Status = "ok"
    detail: str | None = None  # motivo de un estado distinto de `ok`

    # `before`: si no, pydantic convierte True → 1 antes de llegar acá.
    @field_validator("normalized_answer", mode="before")
    @classmethod
    def _check_normalized(cls, v: int | str) -> int | str:
        if isinstance(v, bool) or (isinstance(v, int) and v < 0):
            raise ValueError("normalized_answer entero tiene que ser ≥ 0")
        if isinstance(v, str) and not v:
            raise ValueError("normalized_answer string no puede ser vacío")
        return v

    @field_validator("evidence")
    @classmethod
    def _check_evidence(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("evidence tiene ids repetidos")
        return v

    def to_question_answer(
        self, question_id: str, qtype: str, question: str, query: Query | None = None
    ) -> dict:
        """Item de `questions` (o de `questions_llm` si se pasa `query`) según el schema."""
        item = {
            "question_id": question_id,
            "type": qtype,
            "question": question,
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "status": self.status,
        }
        if self.detail is not None:
            item["status_detail"] = self.detail
        if query is not None:
            item["interpreted_query"] = query.model_dump(mode="json")
        return item
