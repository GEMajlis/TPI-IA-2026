import json

import pytest

from scripts.validate_delivery import DEFAULT_ARTIFACTS, DEFAULT_OUTPUT, validate_delivery


def test_validacion_guardada_pasa_todos_los_checks() -> None:
    result = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    assert result["passed"] is True
    assert result["checks"]
    assert all(item["ok"] for item in result["checks"].values())
    assert len(result["manual_actions"]) == 3


@pytest.mark.skipif(not DEFAULT_ARTIFACTS.is_dir(), reason="no están los artefactos de Kaggle")
def test_validacion_coincide_con_los_artefactos_finales() -> None:
    expected = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    assert validate_delivery(DEFAULT_ARTIFACTS) == expected
