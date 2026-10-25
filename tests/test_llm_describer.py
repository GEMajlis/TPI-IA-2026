from scene.llm.describer import SceneDescriber, build_scene_summary, verify_description

CATEGORIES = {1: "cup", 2: "table", 3: "chair"}
SCENE = {
    "image": {"id": "x", "width": 100, "height": 100},
    "objects": [
        {"id": "obj_1", "category_id": 1, "position": "top_left", "size": "small"},
        {"id": "obj_2", "category_id": 1, "position": "center", "size": "small"},
        {"id": "obj_3", "category_id": 2, "position": "center", "size": "large"},
    ],
    "relations": [{"subject": "obj_1", "predicate": "above", "object": "obj_3"}],
}


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        return next(self.outputs)


def test_resumen_solo_contiene_campos_estructurados() -> None:
    summary = build_scene_summary(
        SCENE, category_names=CATEGORIES, max_objects=10, max_relations=10
    )
    assert "cup: 2" in summary
    assert "table: 1" in summary
    assert "cup (obj_1) above table (obj_3)" in summary
    assert "bbox" not in summary


def test_verificador_acepta_categorias_y_conteos_presentes() -> None:
    report = verify_description(
        "Hay 2 cup y una table; una cup está arriba.", SCENE, category_names=CATEGORIES
    )
    assert report.faithful
    assert report.unsupported_categories == ()
    assert report.unsupported_numbers == ()


def test_verificador_detecta_categoria_y_numero_inventados() -> None:
    report = verify_description("Hay 7 chair junto a la table.", SCENE, category_names=CATEGORIES)
    assert not report.faithful
    assert report.unsupported_categories == ("chair",)
    assert report.unsupported_numbers == (7,)


def test_descriptor_reintenta_y_registra_fidelidad() -> None:
    generator = FakeGenerator(["Hay 7 chair.", "Hay 2 cup y una table."])
    describer = SceneDescriber(
        generator,
        category_names=CATEGORIES,
        max_new_tokens=50,
        max_objects=10,
        max_relations=10,
        retry_on_unfaithful=1,
    )
    result = describer.describe(SCENE)
    assert result.status == "ok"
    assert result.attempts == 2
    assert result.fidelity.faithful
    assert "no fue fiel" in generator.prompts[1]
    assert "image" not in generator.prompts[0].lower()
