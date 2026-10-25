from scene.qa.verbalize import verbalize


def test_verbaliza_conteo_con_categoria() -> None:
    result = {"normalized_answer": 2, "status": "ok"}
    query = {"op": "count", "target": {"category": "cup"}}
    assert verbalize(result, query) == "Hay 2 instancias de cup."


def test_verbaliza_booleanos_posicion_y_tamano() -> None:
    assert verbalize({"normalized_answer": "yes", "status": "ok"}) == "Sí."
    assert (
        verbalize({"normalized_answer": "bottom_right", "status": "ok"})
        == "La instancia está abajo a la derecha."
    )
    assert (
        verbalize({"normalized_answer": "medium", "status": "ok"})
        == "La instancia tiene tamaño mediano."
    )


def test_verbaliza_estados_explicitos() -> None:
    assert "soportada" in verbalize({"status": "unsupported"})
    assert "suficiente" in verbalize({"status": "insufficient_information"})
    assert "ambigua" in verbalize({"status": "ambiguous"})
