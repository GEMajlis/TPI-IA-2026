import json
import re

import pytest

from scripts.build_report import (
    DEFAULT_ARTIFACTS,
    DEFAULT_DATA,
    DEFAULT_REPORT,
    collect_report_data,
    render_report,
)


def test_reporte_se_regenera_desde_el_snapshot() -> None:
    data = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))

    assert render_report(data) == DEFAULT_REPORT.read_text(encoding="utf-8")
    assert data["detection"]["summary"]["f1"] >= 0.5
    assert data["submission"]["rows"] == 80
    assert data["submission"]["schema_errors"] == 0
    assert data["traces"]["dev"]["errors"] == 0
    assert data["qa"]["llm"]["json_valid_rate"] == 1.0
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in data["sources"].values())


@pytest.mark.skipif(not DEFAULT_ARTIFACTS.is_dir(), reason="no están los artefactos de Kaggle")
def test_snapshot_coincide_con_los_artefactos_finales() -> None:
    expected = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))

    assert collect_report_data(DEFAULT_ARTIFACTS) == expected


def test_reporte_tiene_la_estructura_pedida() -> None:
    report = DEFAULT_REPORT.read_text(encoding="utf-8")

    for heading in (
        "## 1. Portada",
        "## 2. Análisis del dataset",
        "## 3. Metodología",
        "## 4. Resultados",
        "## 5. Discusión y análisis de errores",
        "## 6. Conclusiones y trabajo futuro",
        "## Trazabilidad",
    ):
        assert heading in report
    assert "TODO" not in report
