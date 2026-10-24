"""Genera la notebook final autocontenida de la Etapa 1.

La notebook escribe el código y las configs con `%%writefile`; en Kaggle solo necesita el dataset
oficial y acceso a los modelos públicos. No se edita a mano: se regenera con este script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCRIPTS = (
    "analyze_errors.py",
    "benchmark_llm.py",
    "compare_runs.py",
    "eval_qa.py",
    "eval_segmentation.py",
    "make_submission.py",
    "run_eda.py",
    "run_eval.py",
    "run_pipeline.py",
    "run_qa.py",
    "run_qa_llm.py",
    "tune_thresholds.py",
)


@dataclass(frozen=True)
class Member:
    """Integrante que aparece en la portada."""

    name: str
    student_id: str


@dataclass(frozen=True)
class DeliveryConfig:
    """Metadatos y opciones de la notebook final."""

    group_number: str
    course: str
    kaggle_team: str
    members: tuple[Member, ...]
    output: Path
    run_oracle_segmentation: bool
    run_threshold_tuning: bool
    run_controlled_experiment: bool
    run_llm_benchmark: bool


def load_delivery(path: str | Path) -> DeliveryConfig:
    """Carga y valida `configs/delivery.yaml`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = raw.get("delivery", {}) if isinstance(raw, dict) else {}
    required = ("group_number", "course", "kaggle_team", "members", "notebook")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"Faltan campos de entrega: {missing}")
    members = tuple(
        Member(str(item["name"]).strip(), str(item["student_id"]).strip())
        for item in data["members"]
    )
    if not members or any(not member.name or not member.student_id for member in members):
        raise ValueError("Cada integrante necesita nombre y legajo")
    notebook = data["notebook"]
    return DeliveryConfig(
        group_number=str(data["group_number"]),
        course=str(data["course"]),
        kaggle_team=str(data["kaggle_team"]),
        members=members,
        output=Path(notebook["output"]),
        run_oracle_segmentation=bool(notebook["run_oracle_segmentation"]),
        run_threshold_tuning=bool(notebook["run_threshold_tuning"]),
        run_controlled_experiment=bool(notebook["run_controlled_experiment"]),
        run_llm_benchmark=bool(notebook["run_llm_benchmark"]),
    )


def runtime_files(root: Path = ROOT) -> list[Path]:
    """Archivos necesarios para ejecutar la pipeline sin clonar el repositorio."""
    paths = {
        root / "pyproject.toml",
        root / "configs" / "base.yaml",
        root / "configs" / "kaggle.yaml",
        root / "configs" / "lexicon_es.yaml",
        root / "configs" / "delivery.yaml",
    }
    paths.update((root / "configs" / "experiments").glob("*.yaml"))
    paths.update(path for path in (root / "src").rglob("*") if path.suffix in {".py", ".txt"})
    paths.update(root / "scripts" / name for name in RUNTIME_SCRIPTS)
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Faltan archivos para la notebook: {missing}")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _cell_id(kind: str, source: str, index: int) -> str:
    digest = hashlib.sha1(f"{kind}:{index}:{source}".encode()).hexdigest()[:12]
    return f"c{index:03d}-{digest}"


def markdown(source: str, *, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": "",
        "metadata": {"tags": tags or []},
        "source": source.strip() + "\n",
    }


def code(source: str, *, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "",
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


def _source_tag(relative: Path) -> str:
    path = relative.as_posix()
    if path.startswith(("src/eval/", "src/eda/")):
        return "ground-truth"
    if path.startswith("src/scene/"):
        return "inference"
    return "project-file"


def writefile_cell(path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Crea una celda que reconstruye exactamente un archivo del proyecto."""
    relative = path.relative_to(root)
    content = path.read_text(encoding="utf-8")
    return code(
        f"%%writefile {relative.as_posix()}\n{content}",
        tags=["source", _source_tag(relative)],
    )


def _cover(cfg: DeliveryConfig) -> str:
    members = "\n".join(f"- {member.name} ({member.student_id})" for member in cfg.members)
    return f"""
# TPI IA 2026P · Comprensión visual de escenas

**Grupo {cfg.group_number} · Curso {cfg.course}**

**Equipo Kaggle:** {cfg.kaggle_team}

## Integrantes

{members}

Esta notebook es la entrega reproducible de la **Etapa 1 — Promoción**. Implementa la pipeline
imagen → detección open-vocabulary → SAM2 → SceneJSON → QA determinista/LLM, junto con
trazabilidad, evaluación por etapa, experimento controlado y análisis de propagación de errores.
"""


def _execution_cells(
    cfg: DeliveryConfig,
) -> tuple[list[dict[str, Any]], str, str]:
    flags = f"""
RUN_ORACLE_SEGMENTATION = {cfg.run_oracle_segmentation!r}
RUN_THRESHOLD_TUNING = {cfg.run_threshold_tuning!r}
RUN_CONTROLLED_EXPERIMENT = {cfg.run_controlled_experiment!r}
RUN_LLM_BENCHMARK = {cfg.run_llm_benchmark!r}
"""
    helpers = """
import json
import subprocess
import sys
from pathlib import Path

from IPython.display import Image, display

BASE_CONFIGS = ["configs/base.yaml", "configs/kaggle.yaml"]
RUNS_DIR = Path("/kaggle/working/outputs/runs")


def run_script(script, *args, configs=BASE_CONFIGS):
    command = [sys.executable, script, "--config", *configs, *map(str, args)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_pipeline(*args, configs=BASE_CONFIGS):
    before = set(RUNS_DIR.glob("*")) if RUNS_DIR.exists() else set()
    run_script("scripts/run_pipeline.py", *args, configs=configs)
    candidates = set(RUNS_DIR.glob("*")) - before
    if not candidates:
        candidates = {path for path in RUNS_DIR.glob("*") if (path / "scenes").is_dir()}
    if not candidates:
        raise RuntimeError("La pipeline no creó un run")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
"""
    cells = [
        markdown(
            """
## 1. Preparación y reproducibilidad

El código de inferencia se reconstruye en celdas etiquetadas `inference`. Los módulos que leen
anotaciones se etiquetan `ground-truth` y solo se usan en EDA/evaluación. Ninguna predicción,
respuesta o SceneJSON usa GT. Las semillas, versiones, hardware y modelos quedan en el manifest.
""",
            tags=["documentation"],
        ),
        code(
            """
from pathlib import Path
import os

PROJECT_ROOT = Path("/kaggle/working/tpi-ia-2026")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
print("Directorio de trabajo:", Path.cwd())
""",
            tags=["bootstrap"],
        ),
    ]
    return cells, flags, helpers


def build_notebook(root: Path, cfg: DeliveryConfig) -> dict[str, Any]:
    """Arma el documento nbformat 4 listo para guardar."""
    files = runtime_files(root)
    parents = sorted({path.relative_to(root).parent.as_posix() for path in files if path.parent})
    cells, flags, helpers = _execution_cells(cfg)
    cells[1] = code(
        cells[1]["source"]
        + "\nfor directory in "
        + repr(parents)
        + ":\n    Path(directory).mkdir(parents=True, exist_ok=True)\n",
        tags=["bootstrap"],
    )
    cells.extend(writefile_cell(path, root) for path in files)
    cells.extend(
        [
            code('%pip install -q -e ".[llm]"', tags=["dependencies"]),
            code(flags, tags=["parameters"]),
            code(helpers, tags=["helpers"]),
            markdown(
                """
## 2. EDA — solo ground truth

Esta sección caracteriza las anotaciones de desarrollo. Sus resultados justifican los umbrales,
el tope de detecciones y el desglose por tamaño; no alimentan la inferencia por imagen.
""",
                tags=["ground-truth", "documentation"],
            ),
            code(
                """
run_script("scripts/run_eda.py", "--out", "/kaggle/working/outputs/eda")
for figure in sorted(Path("/kaggle/working/outputs/eda/figures").glob("*.png")):
    print(figure.name)
    display(Image(filename=str(figure)))
""",
                tags=["ground-truth", "eda"],
            ),
            markdown(
                """
### Hallazgos principales

- 120 imágenes, 1790 instancias y 120 categorías.
- El 71 % de los objetos ocupa menos del 1 % de la imagen: el recall de objetos chicos es el
  principal riesgo.
- La evaluación LVIS es federada; las categorías no verificadas se ignoran.
- El GT contiene objetos con dos etiquetas superpuestas, por lo que el NMS entre clases se deja
  apagado en la configuración final.
""",
                tags=["ground-truth", "documentation"],
            ),
            markdown(
                """
## 3. Metodología

**Percepción.** OWLv2 recibe las 120 categorías, aplica umbral global 0,18 y NMS por clase.
SAM2.1 hiera-large recibe exclusivamente las cajas predichas y produce máscaras.

**Representación.** Área, centroide, posición, tamaño, color y relaciones se derivan de las
máscaras predichas.

**QA.** El modo determinista traduce los campos estructurados al DSL. En el modo LLM,
Qwen2.5-3B recibe solo la pregunta, genera el mismo DSL validado y el ejecutor calcula la respuesta.
El descriptor recibe solo un resumen textual del SceneJSON.

**Trazabilidad.** Cada etapa escribe tiempos, entradas, salidas, errores y artefactos en JSONL.
""",
                tags=["documentation"],
            ),
            markdown(
                """
## 4. Pipeline final sobre desarrollo

La inferencia siguiente no abre anotaciones. `--eval` ejecuta la evaluación recién después de
guardar las predicciones y mantiene la frontera en los paquetes `eval`/`eda`.
""",
                tags=["documentation"],
            ),
            code(
                'DEV_RUN = run_pipeline("--split", "dev", "--eval")\nprint("Run dev:", DEV_RUN)',
                tags=["inference", "gpu"],
            ),
            code(
                'run_script("scripts/run_eval.py", "--run", DEV_RUN)',
                tags=["ground-truth", "evaluation"],
            ),
            code(
                """
dev_metrics = {
    "detection": read_json(DEV_RUN / "metrics_detection.json")["summary"],
    "segmentation": read_json(DEV_RUN / "metrics_segmentation.json")["summary"],
    "scene": read_json(DEV_RUN / "metrics_scene.json"),
}
dev_metrics
""",
                tags=["ground-truth", "results"],
            ),
            markdown(
                """
## 5. Diagnóstico, ajuste y propagación de errores

Las cajas GT se usan solamente como oráculo de evaluación para aislar el error de SAM2. El
barrido selecciona el umbral en tuning y reporta una única vez sobre holdout. El embudo identifica
la primera etapa en la que se pierde cada instancia.
""",
                tags=["ground-truth", "documentation"],
            ),
            code(
                """
if RUN_ORACLE_SEGMENTATION:
    run_script("scripts/eval_segmentation.py", "--run", DEV_RUN)
else:
    print("Oráculo omitido por configuración")

if RUN_THRESHOLD_TUNING:
    run_script("scripts/tune_thresholds.py")
else:
    print("Tuning omitido por configuración")

run_script("scripts/analyze_errors.py", "--run", DEV_RUN)
error_summary = read_json(DEV_RUN / "error_analysis" / "summary.json")
error_summary
""",
                tags=["ground-truth", "evaluation"],
            ),
            code(
                """
for figure in sorted((DEV_RUN / "error_analysis" / "gallery").glob("*.png")):
    print(figure.name)
    display(Image(filename=str(figure)))
""",
                tags=["ground-truth", "results"],
            ),
            markdown(
                """
## 6. Experimento controlado

Hipótesis: aplicar el post-proceso de SAM2 solo a máscaras chicas mejora su F1 sin perjudicar
medium/large. Se modifica una sola variable y se decide con holdout. El experimento previo fue
descartado: mejoró con cajas GT, pero el F1 end-to-end de máscaras bajó 0,614 → 0,609.
""",
                tags=["ground-truth", "documentation"],
            ),
            code(
                """
EXPERIMENT_RUN = None
if RUN_CONTROLLED_EXPERIMENT:
    experiment_configs = BASE_CONFIGS + ["configs/experiments/e1_30_pp_solo_chicas.yaml"]
    EXPERIMENT_RUN = run_pipeline("--split", "dev", configs=experiment_configs)
    run_script(
        "scripts/compare_runs.py",
        "--base", DEV_RUN,
        "--exp", EXPERIMENT_RUN,
        "--name", "e1_30_pp_solo_chicas",
        "--hypothesis", "post-proceso solo en máscaras chicas sube el F1 de máscaras",
        "--change", "apply_below_area_ratio: null -> 0.00333",
        "--conclusion", "descartado: baja el F1 de máscaras en holdout",
    )
else:
    print("Experimento omitido por configuración")
""",
                tags=["ground-truth", "experiment", "gpu"],
            ),
            markdown(
                """
## 7. Costo del LLM

Se comparan los candidatos con los mismos prompts, una pasada de calentamiento y tres repeticiones.
La decisión previa eligió Qwen2.5-3B fp16: ~5994 MiB y 5,3 s promedio, frente a 2,6 GiB y
10,9 s de Qwen3-4B 4-bit en una T4.
""",
                tags=["documentation"],
            ),
            code(
                """
if RUN_LLM_BENCHMARK:
    run_script(
        "scripts/benchmark_llm.py",
        "--output", "/kaggle/working/outputs/llm_benchmark.json",
    )
    display(read_json("/kaggle/working/outputs/llm_benchmark.json"))
else:
    print("Benchmark LLM omitido por configuración")
""",
                tags=["evaluation", "gpu"],
            ),
            markdown(
                """
## 8. Inferencia de test/benchmark y preguntas

Estas 80 imágenes se procesan sin GT. El LLM nunca recibe la imagen ni el SceneJSON completo:
solo el texto de cada pregunta; el mismo ejecutor determinista calcula la respuesta.
""",
                tags=["documentation"],
            ),
            code(
                """
SUBMISSION_RUN = run_pipeline("--split", "test", "benchmark")
run_script("scripts/run_qa.py", "--run", SUBMISSION_RUN)
run_script("scripts/run_qa_llm.py", "--run", SUBMISSION_RUN)
print("Run submission:", SUBMISSION_RUN)
""",
                tags=["inference", "gpu"],
            ),
            code(
                'run_script("scripts/eval_qa.py", "--run", SUBMISSION_RUN)',
                tags=["ground-truth", "evaluation"],
            ),
            code(
                """
SUBMISSION = Path("/kaggle/working/submission.csv")
run_script("scripts/make_submission.py", "--run", SUBMISSION_RUN, "--out", SUBMISSION)
qa_metrics = read_json(SUBMISSION_RUN / "metrics_qa.json")
qa_metrics
""",
                tags=["inference", "submission"],
            ),
            markdown(
                """
## 9. Resultados y discusión

En la corrida de referencia, el umbral 0,18 obtuvo F1 de cajas **0,653** y F1 de máscaras
**0,596** sobre las 120 imágenes. La consistencia estructural fue perfecta en 6905 objetos. El QA
determinista y LLM alcanzaron 72,5 % contra el oráculo; las 11 respuestas erróneas provinieron de
percepción, no de razonamiento. La submission aceptada obtuvo `PipelineScore = 0.6509`.

El embudo muestra que la mayor pérdida está en detección y se concentra en objetos chicos. Esto
motiva tiling tipo SAHI como extensión de Etapa 2, pero no se usa en la entrega de Etapa 1.
""",
                tags=["results", "documentation"],
            ),
            code(
                f"""
from scene.io.submission import read_submission

submission_rows = read_submission(SUBMISSION)
assert len(submission_rows) == 80
required = [
    DEV_RUN / "run_manifest.json",
    DEV_RUN / "trace.jsonl",
    DEV_RUN / "metrics_detection.json",
    DEV_RUN / "metrics_segmentation.json",
    DEV_RUN / "metrics_scene.json",
    DEV_RUN / "error_analysis" / "summary.json",
    SUBMISSION_RUN / "metrics_qa.json",
    SUBMISSION,
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, missing

final_summary = {{
    "group": {cfg.group_number!r},
    "course": {cfg.course!r},
    "kaggle_team": {cfg.kaggle_team!r},
    "dev_run": str(DEV_RUN),
    "experiment_run": str(EXPERIMENT_RUN) if EXPERIMENT_RUN else None,
    "submission_run": str(SUBMISSION_RUN),
    "submission_rows": len(submission_rows),
    "reference_pipeline_score": 0.6509,
    "metrics": {{"dev": dev_metrics, "qa": qa_metrics}},
}}
summary_path = Path("/kaggle/working/outputs/final_summary.json")
summary_path.write_text(json.dumps(final_summary, indent=2, default=str), encoding="utf-8")

import zipfile

artifact_zip = Path("/kaggle/working/grupo15_entrega_artifacts.zip")
artifact_files = [SUBMISSION, summary_path]
for run in (DEV_RUN, SUBMISSION_RUN):
    for pattern in ("metrics*.json", "*.csv", "trace.jsonl", "run_manifest.json"):
        artifact_files.extend(run.glob(pattern))
for directory in (
    Path("/kaggle/working/outputs/eda"),
    Path("/kaggle/working/outputs/tuning"),
    DEV_RUN / "error_analysis",
):
    if directory.exists():
        artifact_files.extend(path for path in directory.rglob("*") if path.is_file())
for optional in (
    Path("/kaggle/working/outputs/experiments.csv"),
    Path("/kaggle/working/outputs/llm_benchmark.json"),
):
    if optional.is_file():
        artifact_files.append(optional)
with zipfile.ZipFile(artifact_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(artifact_files)):
        archive.write(path, path.relative_to("/kaggle/working"))

print("Entrega validada. Resumen:", summary_path)
print("Artefactos para reporte/checklist:", artifact_zip)
display(final_summary)
""",
                tags=["validation", "results"],
            ),
            markdown(
                """
## 10. Conclusiones y limitaciones

La pipeline supera la meta de F1 de detección, entrega máscaras y SceneJSON consistentes, y
responde mediante dos modos que comparten el mismo razonador verificable. La trazabilidad permite
atribuir los errores a la etapa correspondiente. Las principales limitaciones son el recall de
objetos chicos, la sensibilidad de SAM2 a cajas imprecisas y la dependencia de un léxico español
manual. El LLM no corrige percepción ni ve la imagen.
""",
                tags=["documentation"],
            ),
        ]
    )
    cells.insert(0, markdown(_cover(cfg), tags=["title", "documentation"]))
    for index, cell in enumerate(cells):
        cell["id"] = _cell_id(cell["cell_type"], cell["source"], index)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "tpi_ia": {
                "group_number": cfg.group_number,
                "course": cfg.course,
                "kaggle_team": cfg.kaggle_team,
                "generated_by": "scripts/build_notebook.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", default="configs/delivery.yaml")
    parser.add_argument("--out", help="override de delivery.notebook.output")
    args = parser.parse_args()
    cfg = load_delivery(ROOT / args.delivery)
    notebook = build_notebook(ROOT, cfg)
    output = ROOT / (Path(args.out) if args.out else cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{output}: {len(notebook['cells'])} celdas")


if __name__ == "__main__":
    main()
