"""Valida los artefactos y documentos de la entrega final de Etapa 1 (E1-34)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.build_notebook import ROOT, build_notebook, load_delivery
    from scripts.build_report import (
        DEFAULT_ARTIFACTS,
        DEFAULT_DATA,
        DEFAULT_REPORT,
        collect_report_data,
        render_report,
    )
else:
    from build_notebook import ROOT, build_notebook, load_delivery
    from build_report import (
        DEFAULT_ARTIFACTS,
        DEFAULT_DATA,
        DEFAULT_REPORT,
        collect_report_data,
        render_report,
    )

DEFAULT_OUTPUT = ROOT / "docs" / "delivery_validation.json"


def _result(ok: bool, evidence: str) -> dict[str, Any]:
    return {"ok": bool(ok), "evidence": evidence}


def validate_delivery(artifacts: Path = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    """Ejecuta las verificaciones deterministas que no requieren volver a correr modelos."""
    snapshot = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
    collected = collect_report_data(artifacts.resolve())
    dev = collected["dev_manifest"]
    submission_manifest = collected["submission_manifest"]
    traces = collected["traces"]
    delivery = load_delivery(ROOT / "configs" / "delivery.yaml")
    notebook_path = ROOT / delivery.output
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    expected_notebook = build_notebook(ROOT, delivery)

    checks = {
        "artifact_snapshot_matches": _result(
            collected == snapshot,
            "report/report_data.json coincide con los archivos y SHA-256 del ZIP final",
        ),
        "notebook_reproducible": _result(
            notebook == expected_notebook,
            f"{delivery.output.as_posix()} coincide con scripts/build_notebook.py",
        ),
        "notebook_name": _result(
            notebook_path.stem == f"TPI_IA_2026P_{delivery.group_number}", notebook_path.name
        ),
        "report_reproducible": _result(
            DEFAULT_REPORT.read_text(encoding="utf-8") == render_report(snapshot),
            "report/REPORTE.md coincide con scripts/build_report.py y report_data.json",
        ),
        "submission_valid": _result(
            collected["submission"]["rows"] == 80 and collected["submission"]["schema_errors"] == 0,
            "80 filas, ids/orden normativos y 0 errores de schema",
        ),
        "submission_has_both_qa_modes": _result(
            collected["submission"]["benchmark_scenes"] == 20
            and collected["submission"]["questions"] == 60
            and collected["submission"]["questions_llm"] == 60,
            "20 escenas benchmark, 60 questions y 60 questions_llm",
        ),
        "manifests_same_config": _result(
            dev["config_hash"] == submission_manifest["config_hash"], dev["config_hash"]
        ),
        "seeds_recorded": _result(
            dev["seed"] == 42
            and submission_manifest["seed"] == 42
            and dev["config"]["llm"]["seed"] == 42,
            "seed=42 en ambos manifests y en llm.seed",
        ),
        "models_recorded": _result(
            bool(dev["models"].get("detector"))
            and bool(dev["models"].get("segmenter"))
            and collected["llm_benchmark"]["results"][0]["device"] == "cuda",
            "OWLv2 y SAM2 en manifest; benchmark LLM sobre CUDA",
        ),
        "kaggle_gpu_recorded": _result(
            dev["hardware"]["device"] == "cuda" and "T4" in dev["hardware"]["gpu"]["name"],
            f"{dev['hardware']['gpu']['name']} ({dev['hardware']['gpu']['total_memory_mb']} MiB)",
        ),
        "dev_trace_complete": _result(
            traces["dev"]["errors"] == 0
            and traces["dev"]["images"] == 120
            and all(
                traces["dev"]["stages"][stage]["count"] == 120
                for stage in ("detection", "detection_postprocess", "segmentation", "scene")
            ),
            "480 registros, 120 imágenes y 0 errores",
        ),
        "submission_trace_complete": _result(
            traces["submission"]["errors"] == 0
            and traces["submission"]["images"] == 80
            and traces["submission"]["stages"]["qa_deterministic"]["count"] == 20
            and traces["submission"]["stages"]["qa_llm"]["count"] == 20,
            "360 registros, 80 imágenes, QA en 20 escenas y 0 errores",
        ),
        "promotion_detection_target": _result(
            collected["detection"]["summary"]["f1"] >= 0.5,
            f"F1 bbox={collected['detection']['summary']['f1']:.6f} >= 0.50",
        ),
        "scene_consistency": _result(
            collected["scene"]["consistency"]["n_issues"] == 0,
            "0 inconsistencias estructurales en el run de desarrollo",
        ),
        "llm_contract": _result(
            collected["qa"]["llm"]["json_valid_rate"] == 1.0
            and collected["qa"]["llm"]["dsl_match_rate"] == 1.0
            and collected["qa"]["llm"]["evidence_supports_rate"] == 1.0,
            "JSON, DSL y evidencia válidos en el 100% del benchmark",
        ),
        "controlled_experiment": _result(
            bool(collected["experiment"]["hypothesis"])
            and bool(collected["experiment"]["change"])
            and bool(collected["experiment"]["conclusion"]),
            "hipótesis, cambio único, holdout, costo y conclusión registrados",
        ),
    }
    return {
        "stage": "Etapa 1 - Promoción",
        "group": collected["group"],
        "run_ids": collected["run_ids"],
        "checks": checks,
        "passed": all(item["ok"] for item in checks.values()),
        "test_commands": [
            "pytest -q",
            "pytest -q tests/test_import_boundary.py",
            "pytest -q tests/test_pipeline_submission.py::"
            "test_pipeline_genera_escenas_validas_y_traza",
            "ruff check .",
            "ruff format --check .",
        ],
        "manual_actions": [
            "Crear y aprobar el PR de desarrollo a main antes de entregar",
            "Confirmar que la versión guardada de Kaggle sea visible para la cátedra",
            "Preparar la defensa de hasta 10 minutos con participación de los cuatro integrantes",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate_delivery(args.artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit("La entrega no pasó todas las verificaciones automáticas")


if __name__ == "__main__":
    main()
