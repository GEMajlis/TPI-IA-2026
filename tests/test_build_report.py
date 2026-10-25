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


def test_seccion_de_umbral_cuando_el_zip_trae_el_ajuste() -> None:
    data = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
    point = {"precision": 0.72, "recall": 0.62, "f1": 0.666, "preds_per_image": 28.2}
    data["tuning"] = {
        "chosen_threshold": 0.18,
        "baseline_threshold": 0.10,
        "holdout_baseline": {
            "precision": 0.55,
            "recall": 0.71,
            "f1": 0.619,
            "preds_per_image": 57.0,
        },
        "holdout_chosen": point,
        "tuning_chosen": {**point, "f1": 0.646},
        "cross_variant": "cross_nms_0.8",
        "holdout_cross": {**point, "f1": 0.6667},
        "n_thresholds": 25,
    }
    report = render_report(data)
    assert "**Mejora adoptada: umbral de detección.**" in report
    assert "mejora el F1 de holdout en 4,7 puntos" in report
    assert "Variante descartada" in report and "prompt v1 → v2" in report
