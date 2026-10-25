from typing import Literal

from pydantic import BaseModel

from scene.llm.interpreter import Interpreter


class Target(BaseModel):
    category: str
    filters: dict


class Query(BaseModel):
    op: Literal["count", "exists", "position", "size", "relation", "attribute", "describe"]
    target: Target
    target2: Target | None
    relation: str | None
    attribute: str | None
    output_type: str


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        return next(self.outputs)


def _interpreter(generator: FakeGenerator) -> Interpreter:
    ids = {"tazas": 1, "taza": 1, "reloj": 2, "anillo": 3}
    return Interpreter(
        generator,
        category_matcher=ids.get,
        category_names={1: "cup", 2: "clock", 3: "ring"},
        query_model=Query,
        max_attempts=2,
    )


def test_interpreta_json_y_category_matcher_decide_nombre() -> None:
    generator = FakeGenerator(
        [
            '```json\n{"op":"count","target":{"category":"tazas","filters":{}},'
            '"target2":null,"relation":null,"attribute":null,"output_type":"int"}\n```'
        ]
    )
    result = _interpreter(generator).interpret("¿Cuántas tazas aparecen?")
    assert result.status == "ok"
    assert result.query.target.category == "cup"
    assert result.interpreted_query["op"] == "count"
    assert result.attempts == 1
    assert "SceneJSON" not in generator.prompts[0]


def test_reintenta_una_vez_ante_json_invalido() -> None:
    valid = (
        '{"op":"exists","target":{"category":"reloj","filters":{}},'
        '"target2":null,"relation":null,"attribute":null,"output_type":"yes_no"}'
    )
    generator = FakeGenerator(["esto no es json", valid])
    result = _interpreter(generator).interpret("¿Hay un reloj?")
    assert result.status == "ok"
    assert result.attempts == 2
    assert len(result.errors) == 1
    assert "salida anterior" in generator.prompts[1].lower()


def test_dos_fallos_devuelven_unsupported() -> None:
    generator = FakeGenerator(["mal", "todavía mal"])
    result = _interpreter(generator).interpret("¿Hay un reloj?")
    assert result.status == "unsupported"
    assert result.query is None
    assert result.attempts == 2
    assert len(result.errors) == 2
    assert result.interpreted_query["status"] == "unsupported"


def test_categoria_fuera_de_vocabulario_termina_unsupported() -> None:
    raw = (
        '{"op":"exists","target":{"category":"dragón","filters":{}},'
        '"target2":null,"relation":null,"attribute":null,"output_type":"yes_no"}'
    )
    result = _interpreter(FakeGenerator([raw, raw])).interpret("¿Hay un dragón?")
    assert result.status == "unsupported"
    assert "fuera del vocabulario" in result.errors[-1]


def test_marca_consulta_no_soportada_sin_reintentar() -> None:
    result = _interpreter(
        FakeGenerator(['{"unsupported_reason":"requiere inferir una causa"}'])
    ).interpret("¿Por qué está contenta?")
    assert result.status == "unsupported"
    assert result.attempts == 1
