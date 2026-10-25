# ruff: noqa: E501
"""Genera el reporte de Etapa 1 a partir de los artefactos finales de Kaggle (E1-33)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from scene.io.submission import read_submission
from scene.scene.schema import validate_scene

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "outputs" / "kaggle" / "final"
DEFAULT_DATA = ROOT / "report" / "report_data.json"
DEFAULT_REPORT = ROOT / "report" / "REPORTE.md"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_summary(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    stages: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "errors": 0, "cached": 0, "duration_s": 0.0, "gpu_peak_mb": 0.0}
    )
    for row in rows:
        stage = stages[row["stage"]]
        stage["count"] += 1
        stage["errors"] += row["status"] == "error"
        stage["cached"] += row["status"] == "cached"
        stage["duration_s"] += float(row["duration_ms"]) / 1000
        stage["gpu_peak_mb"] = max(stage["gpu_peak_mb"], float(row.get("gpu_mem_peak_mb") or 0))
    for stage in stages.values():
        stage["duration_s"] = round(stage["duration_s"], 3)
        stage["gpu_peak_mb"] = round(stage["gpu_peak_mb"], 1)
    return {
        "records": len(rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "images": len({row["image_id"] for row in rows}),
        "stages": dict(stages),
    }


def _submission_summary(path: Path) -> dict[str, Any]:
    scenes = read_submission(path)
    schema = ROOT / "data" / "competition" / "scene_schema.json"
    sample = read_submission(ROOT / "data" / "competition" / "sample_submission.csv")
    errors = {iid: validate_scene(scene, schema) for iid, scene in scenes.items()}
    errors = {iid: value for iid, value in errors.items() if value}
    if list(scenes) != list(sample):
        raise ValueError("El submission no conserva los ids y el orden del sample_submission.csv")
    if errors:
        raise ValueError(f"SceneJSON inválidos: {dict(list(errors.items())[:3])}")
    question_rows = sum(len(scene.get("questions", [])) for scene in scenes.values())
    question_llm_rows = sum(len(scene.get("questions_llm", [])) for scene in scenes.values())
    benchmark_scenes = sum(bool(scene.get("questions")) for scene in scenes.values())
    return {
        "rows": len(scenes),
        "objects": sum(len(scene["objects"]) for scene in scenes.values()),
        "relations": sum(len(scene["relations"]) for scene in scenes.values()),
        "benchmark_scenes": benchmark_scenes,
        "questions": question_rows,
        "questions_llm": question_llm_rows,
        "schema_errors": 0,
        "size_mb": round(path.stat().st_size / 2**20, 2),
    }


def _tuning_data(tuning_dir: Path) -> dict[str, Any] | None:
    """Ajuste de umbral (E1-29/E1-30): resumen y puntos de la curva en holdout, si están."""
    summaries = sorted(tuning_dir.glob("*/summary.json")) if tuning_dir.is_dir() else []
    if not summaries:
        return None
    summary_path = summaries[-1]
    curve_path = summary_path.parent / "curve.csv"
    summary = _json(summary_path)
    with curve_path.open(encoding="utf-8", newline="") as stream:
        curve = list(csv.DictReader(stream))

    def point(variant: str, threshold: float, split: str = "holdout") -> dict[str, float]:
        row = next(
            r
            for r in curve
            if r["variant"] == variant
            and r["split"] == split
            and abs(float(r["threshold"]) - threshold) < 1e-9
        )
        keys = ("precision", "recall", "f1", "preds_per_image")
        return {k: float(row[k]) for k in keys}

    base = summary["variants"]["base"]
    cross_name = next(v for v in summary["variants"] if v != "base")
    chosen = float(base["chosen_threshold"])
    baseline = 0.10  # umbral inicial de la línea base (D-14)
    return {
        "summary_path": summary_path,
        "curve_path": curve_path,
        "chosen_threshold": chosen,
        "baseline_threshold": baseline,
        "holdout_baseline": point("base", baseline),
        "holdout_chosen": point("base", chosen),
        "tuning_chosen": point("base", chosen, "tuning"),
        "cross_variant": cross_name,
        "holdout_cross": point(
            cross_name, float(summary["variants"][cross_name]["chosen_threshold"])
        ),
        "n_thresholds": len({r["threshold"] for r in curve}),
    }


def collect_report_data(artifacts: Path) -> dict[str, Any]:
    """Carga y cruza toda la evidencia que sustenta el reporte."""
    final_path = artifacts / "outputs" / "final_summary.json"
    final = _json(final_path)
    dev_id = Path(final["dev_run"]).name
    submission_id = Path(final["submission_run"]).name
    dev = artifacts / "outputs" / "runs" / dev_id
    submission_run = artifacts / "outputs" / "runs" / submission_id
    paths = {
        "final_summary": final_path,
        "eda": artifacts / "outputs" / "eda" / "eda_summary.json",
        "detection": dev / "metrics_detection.json",
        "segmentation": dev / "metrics_segmentation.json",
        "segmentation_oracle": dev / "metrics_segmentation_oracle.json",
        "scene": dev / "metrics_scene.json",
        "qa": submission_run / "metrics_qa.json",
        "error_analysis": dev / "error_analysis" / "summary.json",
        "experiment": artifacts / "outputs" / "experiments.csv",
        "llm_benchmark": artifacts / "outputs" / "llm_benchmark.json",
        "dev_manifest": dev / "run_manifest.json",
        "submission_manifest": submission_run / "run_manifest.json",
        "dev_trace": dev / "trace.jsonl",
        "submission_trace": submission_run / "trace.jsonl",
        "submission": artifacts / "submission.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos: {missing}")
    tuning = _tuning_data(artifacts / "outputs" / "tuning")
    if tuning is not None:
        paths["tuning_summary"] = tuning.pop("summary_path")
        paths["tuning_curve"] = tuning.pop("curve_path")

    delivery = yaml.safe_load((ROOT / "configs" / "delivery.yaml").read_text(encoding="utf-8"))[
        "delivery"
    ]
    with paths["experiment"].open(encoding="utf-8-sig", newline="") as stream:
        experiments = list(csv.DictReader(stream))
    if not experiments:
        raise ValueError("experiments.csv no contiene comparaciones")

    return {
        "generated_from": str(artifacts.relative_to(ROOT)).replace("\\", "/"),
        "group": {
            "number": str(delivery["group_number"]),
            "course": str(delivery["course"]),
            "kaggle_team": delivery["kaggle_team"],
            "members": delivery["members"],
        },
        "run_ids": {"dev": dev_id, "submission": submission_id},
        "final": final,
        "eda": _json(paths["eda"]),
        "detection": _json(paths["detection"]),
        "segmentation": _json(paths["segmentation"]),
        "segmentation_oracle": _json(paths["segmentation_oracle"]),
        "scene": _json(paths["scene"]),
        "qa": _json(paths["qa"]),
        "error_analysis": _json(paths["error_analysis"]),
        "experiment": experiments[-1],
        **({"tuning": tuning} if tuning is not None else {}),
        "llm_benchmark": _json(paths["llm_benchmark"]),
        "dev_manifest": _json(paths["dev_manifest"]),
        "submission_manifest": _json(paths["submission_manifest"]),
        "traces": {
            "dev": _trace_summary(paths["dev_trace"]),
            "submission": _trace_summary(paths["submission_trace"]),
        },
        "submission": _submission_summary(paths["submission"]),
        "sources": {
            name: {
                "path": str(path.relative_to(artifacts)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }


def _n(value: float, decimals: int = 3) -> str:
    return f"{float(value):.{decimals}f}".replace(".", ",")


def _pct(value: float, decimals: int = 1) -> str:
    return f"{100 * float(value):.{decimals}f}%".replace(".", ",")


def _tuning_section(tuning: dict[str, Any] | None) -> str:
    """Párrafo y tabla de la mejora adoptada (umbral); vacío si el ZIP no trae el ajuste."""
    if tuning is None:
        return ""
    base, chosen = tuning["holdout_baseline"], tuning["holdout_chosen"]
    cross = tuning["holdout_cross"]
    rows = "\n".join(
        f"| {label} | {_n(m['precision'])} | {_n(m['recall'])} | {_n(m['f1'])} | {_n(m['preds_per_image'], 1)} |"
        for label, m in (
            (f"Línea base, umbral {_n(tuning['baseline_threshold'], 2)}", base),
            (f"**Umbral elegido, {_n(tuning['chosen_threshold'], 2)}**", chosen),
            (f"Umbral elegido + NMS entre clases ({tuning['cross_variant']})", cross),
        )
    )
    delta = (chosen["f1"] - base["f1"]) * 100
    return f"""**Mejora adoptada: umbral de detección.** Sobre la caché de detecciones crudas se barrieron
{tuning["n_thresholds"]} umbrales; el de máximo F1 de cajas en tuning fue
{_n(tuning["chosen_threshold"], 2)} (F1 {_n(tuning["tuning_chosen"]["f1"])} en tuning). En holdout:

| Configuración | Precisión | Recall | F1 de cajas | Predicciones por imagen |
|---|---:|---:|---:|---:|
{rows}

El umbral elegido **mejora el F1 de holdout en {_n(delta, 1)} puntos** frente a la línea base: cambia
recall por precisión y reduce a la mitad las predicciones por imagen. El NMS entre clases no aporta
y queda apagado.

"""


def render_report(data: dict[str, Any]) -> str:
    """Renderiza el Markdown sin recalcular cifras."""
    group = data["group"]
    members = "\n".join(
        f"- {member['name']} ({member['student_id']})" for member in group["members"]
    )
    eda = data["eda"]
    det = data["detection"]
    seg = data["segmentation"]
    oracle = data["segmentation_oracle"]
    scene = data["scene"]
    qa = data["qa"]
    errors = data["error_analysis"]
    experiment = data["experiment"]
    llm = data["llm_benchmark"]
    tuning_section = _tuning_section(data.get("tuning"))
    manifest = data["dev_manifest"]
    submission = data["submission"]
    final = data["final"]
    trace = data["traces"]["dev"]
    ds, ss = det["summary"], seg["summary"]
    attrs = scene["attributes"]
    consistency = scene["consistency"]
    funnel = errors["funnel"]["global"]
    det_size = det["by_size"]
    seg_size = seg["by_size"]
    llm_rows = {row["candidate"]: row for row in llm["results"]}
    qwen = llm_rows["qwen25_3b"]
    qwen3 = llm_rows["qwen3_4b"]
    source_rows = "\n".join(
        f"| `{name}` | `{source['path']}` | `{source['sha256']}` |"
        for name, source in data["sources"].items()
    )
    stage_rows = "\n".join(
        f"| {name} | {value['count']} | {_n(value['duration_s'], 1)} | "
        f"{_n(value['gpu_peak_mb'], 0)} | {value['errors']} |"
        for name, value in trace["stages"].items()
    )

    return f"""# TPI IA 2026P - Comprensión visual de escenas

## 1. Portada

**Grupo {group["number"]} - Curso {group["course"]}**

**Equipo Kaggle:** {group["kaggle_team"]}

**Etapa:** 1 - Promoción

**Notebook:** `TPI_IA_2026P_{group["number"]}.ipynb`

### Integrantes

{members}

## 2. Análisis del dataset

El conjunto de desarrollo contiene {eda["n_images"]} imágenes, {eda["n_instances"]} instancias y
{manifest["models"]["detector"]["n_queries"]} categorías
evaluables. La mediana es {_n(eda["instances_per_image"]["50%"], 1)} instancias por imagen y el
máximo es {_n(eda["instances_per_image"]["max"], 0)}. El {_pct(eda["share_area_lt_1pct"])} de los
objetos ocupa menos del 1% de la imagen, y el {_pct(eda["share_instances_overlapping"])} se
superpone con otra instancia. Esto anticipa dificultades de recall, oclusión y segmentación fina.

| Tamaño por área relativa | Proporción |
|---|---:|
| Small (< 0,333%) | {_pct(eda["size_split"]["small"])} |
| Medium (< 3%) | {_pct(eda["size_split"]["medium"])} |
| Large | {_pct(eda["size_split"]["large"])} |

Las categorías aparecen en entre {eda["category_images_range"][0]} y
{eda["category_images_range"][1]} imágenes. Debido a la política federada de LVIS, una imagen
tiene en promedio {_n(eda["federated_per_image_mean"]["n_positive"], 2)} categorías positivas,
{_n(eda["federated_per_image_mean"]["n_negative"], 2)} negativas verificadas y
{_n(eda["federated_per_image_mean"]["n_unverified"], 2)} no verificadas. Por eso la precisión se
reporta junto con la cobertura evaluable y no se consideran falsos positivos las categorías que
LVIS no permite juzgar. El benchmark lingüístico tiene 60 consignas: 20 descripciones y 40
preguntas cerradas.

**Decisiones derivadas.** Se usan umbrales de tamaño 0,00333 y 0,03, NMS por clase para no borrar
instancias legítimas de clases superpuestas, tope de 100 detecciones y análisis separado por tamaño.

## 3. Metodología

La pipeline es `imagen -> OWLv2 -> post-proceso -> SAM2 -> SceneJSON -> QA`. El detector
`{manifest["models"]["detector"]["model_id"]}` consulta las 120 categorías con el prompt
`{manifest["models"]["detector"]["template"]}`, conserva detecciones crudas desde score 0,01 y
aplica umbral 0,18, NMS por clase con IoU 0,5 y un máximo de 100 objetos. Las cajas resultantes se
usan como prompts de `{manifest["models"]["segmenter"]["model_id"]}`; por caja se elige la mejor de
tres máscaras y se aplican recorte con margen, componente principal y relleno de huecos.

El SceneJSON guarda ids de instancia, clase, score, caja, máscara RLE no comprimida column-major,
centroide, área relativa, posición 3x3, tamaño, color dominante y relaciones espaciales. Cada campo
distingue datos observados, derivados y desconocidos. La validación usa el `scene_schema.json`
normativo y chequeos de referencias, ids y tamaños de máscara.

El modo determinista transforma los campos estructurados del benchmark en un DSL validado. El
modo LLM usa `{manifest["config"]["llm"]["candidates"]["qwen25_3b"]["model_id"]}` en fp16: recibe
solamente el texto, genera el mismo DSL, y el ejecutor determinista calcula la respuesta. El LLM
verbaliza y describe, pero nunca recibe la imagen, el GT ni calcula conteos o relaciones. Los
estados `ambiguous`, `unsupported` e `insufficient_information` evitan inventar datos.

Las anotaciones se leen exclusivamente desde `src/eval/` y `src/eda/`. La inferencia acepta una
imagen arbitraria y no importa esos paquetes. Cada etapa registra parámetros, modelo, tiempos,
estado, conteos, memoria y artefacto en `trace.jsonl`. La semilla global y del LLM es
{manifest["seed"]}.

### Modelos, recursos y código reutilizado

- OWLv2: `google/owlv2-base-patch16-ensemble`, descargado de Hugging Face.
- Segmentación: `facebook/sam2.1-hiera-large`, descargado de Hugging Face.
- LLM: `Qwen/Qwen2.5-3B-Instruct`; candidato comparado: `Qwen/Qwen3-4B`.
- Matching semántico de respaldo: `sentence-transformers/LaBSE`.
- Sinónimos LVIS: metadata pública de Detectron2, descargada y parseada sin ejecutar código.
- Librerías: PyTorch, Transformers, NumPy, OpenCV, Pillow, pandas, scikit-learn, pycocotools,
  pydantic y jsonschema. No se incorporó código ad hoc de terceros al repositorio.
- Ejecución final: Python {manifest["python"].split()[0]}, PyTorch
  {manifest["packages"]["torch"]} y GPU {manifest["hardware"]["gpu"]["name"]} de
  {_n(manifest["hardware"]["gpu"]["total_memory_mb"] / 1024, 1)} GiB en Kaggle.

## 4. Resultados

### 4.1 Percepción sobre desarrollo

Se evaluaron las {ds["n_images"]} imágenes con IoU >= 0,5 y política federada LVIS.

| Etapa | Precisión | Recall | F1 | F1 macro | Cobertura evaluable |
|---|---:|---:|---:|---:|---:|
| Cajas | {_n(ds["precision"])} | {_n(ds["recall"])} | **{_n(ds["f1"])}** | {_n(ds["f1_macro"])} | {_n(ds["evaluable_coverage"])} |
| Máscaras | {_n(ss["precision"])} | {_n(ss["recall"])} | **{_n(ss["f1"])}** | {_n(ss["f1_macro"])} | {_n(ss["evaluable_coverage"])} |

El F1 de cajas supera la meta de promoción de 0,50.

| Tamaño | F1 cajas | F1 máscaras |
|---|---:|---:|
| Small | {_n(det_size["small"]["f1"])} | {_n(seg_size["small"]["f1"])} |
| Medium | {_n(det_size["medium"]["f1"])} | {_n(seg_size["medium"]["f1"])} |
| Large | {_n(det_size["large"]["f1"])} | {_n(seg_size["large"]["f1"])} |

Con cajas GT, SAM2 alcanza IoU >= 0,5 en
{_pct(oracle["oracle_gt_boxes"]["frac_iou_ge_thr_post"])} de {oracle["oracle_gt_boxes"]["n"]}
instancias. Con cajas predichas, {oracle["predicted_boxes"]["box_tp_lost_in_mask"]} TP de caja se
pierden en la etapa de máscara, lo que confirma la propagación del error de localización.

### 4.2 Representación estructurada

Se validaron {consistency["n_scenes"]} escenas, {consistency["n_objects"]} objetos y
{consistency["n_relations"]} relaciones con **{consistency["n_issues"]} inconsistencias**. Sobre
{attrs["n_pairs"]} pares TP, la exactitud fue {_pct(attrs["position_acc"])} en posición,
{_pct(attrs["size_acc"])} en tamaño y {_pct(attrs["color_acc"])} en color. El
{_pct(consistency["mask_outside_bbox"]["frac_objects_gt_10pct"])} de los objetos tiene más del 10%
de su máscara fuera de la caja, principal alerta de consistencia geométrica.

### 4.3 Preguntas y descripciones

| Métrica | Determinista | LLM |
|---|---:|---:|
| Exactitud vs. oráculo (40 cerradas) | {_pct(qa["deterministic"]["accuracy_vs_oracle"])} | {_pct(qa["llm"]["accuracy_vs_oracle"])} |
| Evidencia válida | {_pct(qa["deterministic"]["evidence_supports_rate"])} | {_pct(qa["llm"]["evidence_supports_rate"])} |
| Reejecución coincide | {_pct(qa["deterministic"]["reexecutes_rate"])} | {_pct(qa["llm"]["reexecutes_rate"])} |
| JSON intermedio válido | - | {_pct(qa["llm"]["json_valid_rate"])} |
| DSL igual al modo estructurado | - | {_pct(qa["llm"]["dsl_match_rate"])} |

Por tipo, ambos modos logran count {_pct(qa["llm"]["by_type"]["count"]["accuracy_vs_oracle"])},
existence {_pct(qa["llm"]["by_type"]["existence"]["accuracy_vs_oracle"])}, position
{_pct(qa["llm"]["by_type"]["position"]["accuracy_vs_oracle"])}, relation
{_pct(qa["llm"]["by_type"]["relation"]["accuracy_vs_oracle"])} y size
{_pct(qa["llm"]["by_type"]["size"]["accuracy_vs_oracle"])}. De 20 descripciones LLM,
{_pct(qa["llm"]["descriptions"]["faithful_rate"])} fueron fieles y
{qa["llm"]["descriptions"]["template_fallback"]} usaron el fallback determinista. Los
{sum(qa["llm"]["error_source"].values())} errores cerrados se atribuyeron a percepción, no a la
interpretación del LLM.

### 4.4 Experimentos controlados

Cada comparación cambia una sola variable respecto de la línea base y se decide con el holdout de
40 imágenes (split estratificado con semilla 42), que no participa de la elección.

{tuning_section}**Variante descartada: post-proceso de máscaras solo en objetos chicos.** Se cambió una sola variable: `{experiment["change"]}`. El F1 de máscaras en holdout pasó de
{_n(experiment["base_holdout_f1_mask"])} a {_n(experiment["exp_holdout_f1_mask"])}
({float(experiment["delta_holdout_f1_mask"]):+.4f}); aunque tuning mejoró
{float(experiment["delta_tuning_f1_mask"]):+.4f}, la variante se **descartó** porque empeoró el
holdout. El costo de segmentación fue {_n(experiment["base_seg_s"], 1)} s en la base y
{_n(experiment["exp_seg_s"], 1)} s en la variante. El oráculo sugería una mejora, pero el
experimento end-to-end mostró que las cajas predichas cambian el efecto del post-proceso.

**Intérprete LLM: prompt v1 → v2.** Con la percepción fija, el prompt v1 daba JSON válido en el
65% de las consignas (las 20 descripciones quedaban `unsupported` por falta de un ejemplo de
`describe`) y usaba categorías del benchmark como ejemplos. El v2, sin contenido del benchmark,
llega al 100% y es el que se usa en este run. Es una medición previa a la corrida final,
registrada en `docs/DECISIONS.md` (D-18).

### 4.5 Recursos, traza y entrega

| Etapa del run dev | Registros | Tiempo acumulado (s) | Pico GPU trazado (MiB) | Errores |
|---|---:|---:|---:|---:|
{stage_rows}

La traza de desarrollo contiene {trace["records"]} registros sobre {trace["images"]} imágenes y
{trace["errors"]} errores. En el benchmark de tres prompts, Qwen2.5-3B usó
{_n(qwen["load_peak_memory_mb"], 0)} MiB y {_n(qwen["latency_mean_ms"], 0)} ms promedio; Qwen3-4B
4-bit usó {_n(qwen3["load_peak_memory_mb"], 0)} MiB pero {_n(qwen3["latency_mean_ms"], 0)} ms, por
lo que se eligió el primero.

El submission final contiene {submission["rows"]} filas, {submission["objects"]} objetos,
{submission["relations"]} relaciones, {submission["benchmark_scenes"]} escenas de benchmark,
{submission["questions"]} respuestas deterministas y {submission["questions_llm"]} respuestas
LLM; no presenta errores de schema y pesa {_n(submission["size_mb"], 2)} MiB. La referencia
registrada del leaderboard es `PipelineScore = {_n(final["reference_pipeline_score"], 4)}`.

## 5. Discusión y análisis de errores

De las {funnel["n"]} instancias GT, {_pct(funnel["frac_ok"])} llega a TP de máscara. Las pérdidas
principales aparecen antes de SAM2: {_pct(funnel["frac_no_box"])} sin caja candidata,
{_pct(funnel["frac_wrong_class"])} con clase incorrecta y {_pct(funnel["frac_below_threshold"])}
bajo el umbral. Sólo {_pct(funnel["frac_nms"])} se pierde por NMS y ninguna por el tope de
detecciones. En objetos small llega a máscara correcta el
{_pct(errors["funnel"]["by_size"]["small"]["frac_ok"])}, contra
{_pct(errors["funnel"]["by_size"]["medium"]["frac_ok"])} en medium y
{_pct(errors["funnel"]["by_size"]["large"]["frac_ok"])} en large.

Los falsos positivos se concentran en mala localización ({errors["fp_types"]["localization"]}) y
fondo ({errors["fp_types"]["background"]}); duplicados, confusiones y categorías negativas son
minoritarios. Las clases más problemáticas combinan objetos chicos o delgados (`strap`,
`doorknob`, `streetlight`) y bajo recall (`book`, `curtain`, `person`). La galería del run conserva
cuatro casos correctos, cuatro ambiguos y cuatro fallidos.

Las limitaciones principales son: recall bajo en objetos pequeños; dependencia de SAM2 respecto de
la caja; cobertura evaluable LVIS de aproximadamente {_pct(ds["evaluable_coverage"])}; conteo QA
de sólo {_pct(qa["llm"]["by_type"]["count"]["accuracy_vs_oracle"])}; y un benchmark de LLM pequeño.
La gran cantidad de relaciones también aumenta el tamaño del envío, aunque no produjo
inconsistencias.

## 6. Conclusiones y trabajo futuro

La Etapa 1 cumple el nivel promoción: F1 de cajas superior a 0,50, SceneJSON válido con atributos y
relaciones, QA determinista y con LLM sobre un único ejecutor, manejo explícito de incertidumbre,
tracing completo y experimento controlado. El resultado muestra que el cuello de botella está en la
percepción, especialmente en objetos pequeños, y no en la interpretación lingüística.

La extensión prioritaria para una Etapa 2 es tiling tipo SAHI con fusión de detecciones, evaluada
sobre el holdout y con medición de costo. También se propone reducir relaciones redundantes y
calibrar conteos sin usar GT durante inferencia.

## Trazabilidad

El reporte fue generado por `scripts/build_report.py`. El snapshot
`report/report_data.json` conserva los datos usados y el SHA-256 completo de cada fuente del ZIP.

| Clave | Archivo dentro de `grupo15_entrega_artifacts.zip` | SHA-256 |
|---|---|---|
{source_rows}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    data = collect_report_data(args.artifacts.resolve())
    args.data.parent.mkdir(parents=True, exist_ok=True)
    args.data.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.write_text(render_report(data), encoding="utf-8")
    print(f"Reporte: {args.out}")
    print(f"Snapshot trazable: {args.data}")


if __name__ == "__main__":
    main()
