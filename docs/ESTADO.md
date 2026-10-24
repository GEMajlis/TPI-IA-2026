# Estado del proyecto

_Última actualización: 2026-10-24_

Resumen para el grupo: qué está hecho, qué sigue y qué falta. El detalle de cada tarea está en [TASKS.md](TASKS.md), las decisiones técnicas en [PLAN.md](PLAN.md) y los desvíos del plan en [DECISIONS.md](DECISIONS.md).

## Cómo arrancar

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                        # 112 tests, todos tienen que pasar
ruff check . && ruff format --check .
```

**Datos:** `data/dataset/` no se versiona. Descargá los datos de la competencia de Kaggle y descomprimilos ahí manteniendo la estructura (`development/`, `tpi_ia_2026_competition_public/`, `tpi_2026_qa_public/`, `scene_schema.json`, `sample_submission.csv`). Los tests que necesitan datos se saltean si la carpeta no está.

**Regla de oro (motivo de desaprobación):** el GT solo se lee desde `src/eval/` y `src/eda/`. Nada en `src/scene/` puede importarlos ni tocar anotaciones. Lo verifica `tests/test_import_boundary.py`; no lo desactiven. Las demás reglas de trabajo están en [CLAUDE.md](../CLAUDE.md).

## Qué hicimos (Etapa 1: 24 de 35 tareas, y E1-18/E1-23 con el código listo)

| Tarea | Qué quedó | Dónde |
|---|---|---|
| E1-00 Bootstrap | Estructura del repo, dependencias, configs `base.yaml` y `kaggle.yaml` | `pyproject.toml`, `configs/` |
| E1-01 Config | Carga de YAML con overrides, semillas, `run_id`, `run_manifest.json` | `src/scene/config.py` |
| E1-02 Datos y frontera GT | Listado de imágenes y vocabulario sin leer GT; carga de GT solo en eval; test de frontera | `src/scene/io/dataset.py`, `src/eval/annotations.py` |
| E1-03 Schema | Validación del SceneJSON y resumen de los archivos normativos | `src/scene/scene/schema.py`, [schema_notes.md](schema_notes.md) |
| E1-04 RLE | Codificación de máscaras igual a la de `pycocotools` | `src/scene/masks/rle.py` |
| E1-05 Tracer | Traza por imagen y etapa en `trace.jsonl`; un error no corta el run | `src/scene/tracing/tracer.py` |
| E1-06 EDA | 8 hallazgos con su implicancia, 7 figuras | `scripts/run_eda.py`, `src/eda/`, [eda_findings.md](eda_findings.md) |
| E1-07 Evaluador | P, R y F1 federados LVIS (cajas y máscaras), por tamaño, frecuencia y categoría | `src/eval/lvis_federated.py` |
| E1-08 Vocabulario | Nombres limpios y sinónimos LVIS opcionales (120 o 215 queries) | `src/scene/detection/vocabulary.py` |
| E1-09 OWLv2 | Detector base con caché de detecciones crudas e imágenes de debug | `src/scene/detection/owlv2.py`, `scripts/run_detection.py` |
| E1-11 SAM2 | Máscaras con las cajas como prompt, post-proceso y caché RLE; F1 de máscaras | `src/scene/segmentation/`, `scripts/run_segmentation.py` |
| E1-13 Propiedades | Área, área relativa, centroide, posición 3×3, tamaño y caja de la máscara; caja rasterizada si la máscara queda vacía | `src/scene/scene/properties.py` |
| E1-14 Color | Color dominante por k-means en Lab, paleta de 11 colores, `null` si la máscara es muy chica | `src/scene/scene/color.py` |
| E1-15 Relaciones | `left_of`, `right_of`, `above`, `below`, `overlaps` (schema) y `near`, `on_top_of` (campo aparte), con tope priorizado | `src/scene/scene/relations.py` |
| E1-16 Builder | SceneJSON validado con origen observado/derivado/desconocido por campo | `src/scene/scene/builder.py` |
| E1-17 Orquestador | Pipeline completa con caché por etapa y traza; acepta imágenes sueltas | `src/scene/pipeline.py`, `scripts/run_pipeline.py` |
| E1-18 Submission (falta envío) | 80 filas en el orden del sample, todas validadas | `src/scene/io/submission.py`, `scripts/make_submission.py` |
| E1-19 Consistencia | Chequeo del SceneJSON contra su máscara y exactitud de posición/tamaño/color contra GT | `src/eval/scene_consistency.py`, `scripts/run_eval.py` |
| E1-12 Oráculo de segmentación | SAM2 con cajas GT: separa el error de SAM2 del del detector | `src/eval/segmentation.py`, `scripts/eval_segmentation.py` |
| E1-29 Ajuste de umbral (0,18) | Barrido de umbral y NMS entre clases en tuning (80) / holdout (40) | `src/eval/tuning.py`, `scripts/tune_thresholds.py` |
| E1-20 DSL | Modelos pydantic de la consulta y del resultado; fixture con el DSL de las 60 consignas del benchmark | `src/scene/qa/dsl.py` |
| E1-21 Category matcher | Texto en inglés o español → categoría: nombre, plurales, léxico español, sinónimos LVIS y LaBSE como último recurso (D-13) | `src/scene/qa/category_matcher.py`, `configs/lexicon_es.yaml` |
| E1-30 Experimentos | Umbral 0,18 adoptado (+4,7 F1 holdout); post-proceso solo en chicas descartado (D-17) | `configs/experiments/`, `scripts/compare_runs.py` |
| E1-22 Ejecutor | Ejecuta el DSL sobre el SceneJSON con filtros, evidencia, confianza y estados `unsupported` / `insufficient_information` / `ambiguous` | `src/scene/qa/executor.py` |
| E1-23 Modo determinista (falta Kaggle) | Benchmark → DSL sin LLM y `questions` en las 20 escenas del benchmark | `src/scene/qa/structured_adapter.py`, `scripts/run_qa.py` |
| E1-31 Análisis de errores (falta QA) | Embudo por instancia, tipos de FP, peores clases y galería; conclusiones en [error_analysis.md](error_analysis.md) | `src/eval/error_analysis.py`, `scripts/analyze_errors.py` |
| E1-24 Evaluación de QA | Fidelidad de la evidencia, re-ejecución del DSL, exactitud contra el SceneJSON oráculo y atribución percepción/interpretación | `src/eval/qa_eval.py`, `src/eval/oracle.py`, `scripts/eval_qa.py` |
| E1-28 `questions_llm` (falta T4) | Pregunta en texto → LLM → DSL → mismo ejecutor; descripción verificada con respaldo determinista | `src/scene/qa/llm_mode.py`, `scripts/run_qa_llm.py` |
| E1-10 Post-proceso | Umbral, NMS por clase, NMS entre clases (apagado) y tope, con conteos en la traza | `src/scene/detection/postprocess.py` |

**Resultados en Kaggle (120 imágenes de dev):**

| Configuración | F1 de cajas | F1 de máscaras | Nota |
|---|---|---|---|
| Línea base (umbral 0,1) | 0,599 | 0,543 | P 0,52 · R 0,71; ~57 predicciones por imagen |
| **Umbral 0,18 (E1-29), holdout de 40 imágenes** | **0,666** | (falta medir) | P 0,72 · R 0,62; ~28 predicciones por imagen |
| **Umbral 0,18, dev completo (120)** | **0,653** | **0,596** | P 0,70 · R 0,61; +5,4 y +5,2 puntos sobre la línea base |

- **Por tamaño (línea base):** F1 de máscaras small 0,42 · medium 0,64 · large 0,76. Los objetos chicos son el cuello de botella.
- **Oráculo (SAM2 con cajas GT):** el 84 % de las instancias llega a IoU ≥ 0,5, pero solo el 72–75 % de las chicas. El post-proceso ayuda en small y perjudica en medium/large → experimento para E1-30 (D-15).
- **Consistencia (E1-19):** 0 inconsistencias en 6905 objetos; posición 97 %, tamaño 95 %, color 90 % contra GT (D-16).

## En desarrollo

- **E1-25 a E1-27 (rama `task/E1-25-27-llm`):** carga lazy de Qwen, benchmark para T4,
  intérprete pregunta→DSL, verbalización, descriptor desde SceneJSON y verificación de
  fidelidad ya tienen implementación y tests. Siguen abiertas hasta integrar E1-16/E1-20/E1-21
  y ejecutar las mediciones de aceptación en Kaggle.

## Lo más importante que aprendimos

- **Hay diferencias con el plan original** ([DECISIONS.md](DECISIONS.md)):
  - el schema solo acepta 5 relaciones (`left_of`, `right_of`, `above`, `below`, `overlaps`): `near` y `on_top_of` van en un campo aparte;
  - el benchmark tiene **60 consignas**, no 40;
  - las preguntas están en **español**.
- **Puntaje de Kaggle:** 40 % F1 de cajas, 35 % F1 de máscaras, 15 % consistencia estructural y 10 % QA. La percepción pesa el 75 %.
- **Los objetos son muy chicos:** el 71 % ocupa menos del 1 % de la imagen. El recall en objetos chicos va a ser el punto débil.
- **La evaluación federada ignora la mayoría de las predicciones:** de las 120 categorías, unas 113 por imagen no están verificadas. Por eso el umbral óptimo probablemente sea bajo.
- **Sin NMS entre clases en la línea base:** el GT tiene objetos anotados con dos categorías a la vez (cup y drinking glass, hat y baseball cap, TV y monitor).
- **Umbrales de tamaño propuestos:** small por debajo de 0,333 %, medium por debajo de 3 %, large el resto. Se cambian en `configs/base.yaml` si el enunciado define otros.

## Próximos pasos (en este orden)

La idea es armar primero un **corte vertical mínimo** y probarlo en Kaggle con GPU, antes de completar escena y QA. El riesgo principal es que los modelos no rindan o no entren en la T4, y el primer F1 real define en qué conviene invertir tiempo.

1. ~~**E1-09 y E1-10 · Detector OWLv2 y post-proceso.**~~ Hecho.
2. ~~**Primera corrida en Kaggle.**~~ Hecho: F1 de cajas 0,60.
3. ~~**SAM2 en Kaggle y E1-13 · propiedades básicas.**~~ Hecho: F1 de máscaras 0,54.
4. **Primer envío (hito H1):** `submission.csv` generado en Kaggle con umbral 0,18 (80 filas válidas, 2271 objetos, 164 796 relaciones, 15,3 MB). Falta subirlo y anotar el puntaje. Ese submission no tiene `questions`: para sumarlas (E1-23) alcanza con correr sobre el mismo run, sin GPU, `python scripts/run_qa.py --config configs/base.yaml configs/kaggle.yaml --run latest` y después `python scripts/make_submission.py --config configs/base.yaml configs/kaggle.yaml --run latest --out /kaggle/working/submission.csv`.

En paralelo, porque no dependen de los modelos (se trabajan con SceneJSON sintéticos):
- ~~**E1-20 a E1-23:** DSL, category matcher, ejecutor y modo determinista.~~ Hecho (E1-23 falta correrlo en Kaggle). Sigue **E1-24**, la evaluación de QA.
- **E1-25:** carga del LLM y benchmark de memoria y latencia en T4.

## Qué queda por hacer

**Etapa 1 (promoción): pendiente.**

| Área | Tareas |
|---|---|
| Percepción | E1-09 OWLv2 · E1-10 post-proceso · E1-11 SAM2 · E1-12 eval. de segmentación con oráculo |
| Escena | E1-13 propiedades · E1-14 color · E1-15 relaciones · E1-16 builder · E1-19 chequeo de consistencia |
| Pipeline y entrega | E1-17 orquestador · E1-18 submission (**H1**) |
| Preguntas | E1-20 DSL · E1-21 category matcher · E1-22 ejecutor · E1-23 modo determinista (**H2**) · E1-24 eval. de QA |
| LLM | E1-25 carga del modelo · E1-26 intérprete · E1-27 verbalización y descripción · E1-28 `questions_llm` |
| Experimentos y análisis | E1-29 ajuste de umbral · E1-30 mejora controlada · E1-31 análisis de errores |
| Entrega final | E1-32 notebook de Kaggle · E1-33 reporte · E1-34 checklist (**H3**) |

**Etapa 2 (sobresaliente):** E2-01 a E2-04. Arranca solo con H3 cumplido. La extensión se elige según el análisis de errores; hoy el candidato más fuerte es el tiling tipo SAHI, por los objetos chicos.

## Pendientes del grupo

- [x] **Subir el enunciado del TPI** (PDF) a `docs/` → `docs/enunciado_TPI_2026P.pdf`. Confirma el ejemplo horse/car/chair de E1-07 y **no define umbrales de tamaño** (sigue D-06). Hace falta para el ejemplo horse/car/chair (E1-07), para el checklist de promoción (E1-34) y para ver si define umbrales de tamaño.
- [ ] **Definir el número de grupo** (nombre de la notebook: `TPI_IA_2026P_<grupo>`).
- [ ] **Crear una notebook en la competencia** con GPU T4 e Internet activado, y confirmar la ruta de los datos con `!ls /kaggle/input`.
- [ ] **Decidir cómo llevar el código a Kaggle** hasta tener el generador de notebook (E1-32): un script autocontenido o el repo subido como dataset.
- [ ] **Revisar y aprobar las decisiones abiertas de [DECISIONS.md](DECISIONS.md):**
  - **D-06:** umbrales de tamaño;
  - **D-09:** sin NMS entre clases en la línea base;
  - **D-10:** qué responder cuando una pregunta en singular encuentra varias instancias (implementada en E1-22: la de mayor confianza, con estado `ambiguous`).
- [x] **Repartir las tareas** (todo en la rama `desarrollo`, un commit por tarea, `git pull` antes de empezar y antes de cada push; cada uno toca su carpeta):
  - **QA:** E1-20, E1-21, E1-22 y E1-23 (`src/scene/qa/`).
  - **LLM:** E1-25, E1-26 y E1-27 (`src/scene/llm/`).
  - **Evaluación:** E1-12, E1-19 y E1-29 (`src/eval/`, `scripts/`). Después: E1-24, E1-28, E1-30 y E1-31.
