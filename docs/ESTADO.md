# Estado del proyecto

_Última actualización: 2026-10-24 · Entrega: **1 de noviembre de 2026, 23:59**_

Resumen para el grupo: qué está hecho, qué resultados tenemos y qué falta. El detalle de cada tarea está en [TASKS.md](TASKS.md), las decisiones técnicas en [PLAN.md](PLAN.md) y los desvíos del plan en [DECISIONS.md](DECISIONS.md).

## Dónde estamos

**Etapa 1 (promoción): 32 de 35 tareas.** La pipeline completa corre en Kaggle de punta a punta (detección → SAM2 → SceneJSON → preguntas en modo determinista y con LLM). El `submission.csv` fue aceptado con `PipelineScore = 0.6509` (**H1 cumplido**). Falta armar la **entrega** (notebook, reporte y checklist).

| Pendiente | Qué falta | Quién |
|---|---|---|
| **E1-32 · Notebook de Kaggle** | `TPI_IA_2026P_<grupo>` que corra de punta a punta sin archivos externos, con secciones en Markdown y todas las salidas guardadas | a repartir |
| **E1-33 · Reporte técnico** | `report/REPORTE.md` (portada, dataset, metodología, resultados, discusión y errores, conclusiones); cada número sale del run final | a repartir |
| **E1-34 · Checklist final** | Condiciones de validez y checklist de promoción, punto por punto | a repartir |

**Etapa 2 (sobresaliente):** E2-01 a E2-04, solo si queda tiempo después de E1-34. El análisis de errores recomienda **tiling tipo SAHI** ([error_analysis.md](error_analysis.md) §6).

## Resultados (Kaggle, Tesla T4)

**Percepción (120 imágenes de dev, IoU ≥ 0,5, política federada LVIS):**

| Configuración | F1 de cajas | F1 de máscaras | Nota |
|---|---|---|---|
| Línea base (umbral 0,10) | 0,599 | 0,543 | P 0,52 · R 0,71 · ~57 predicciones por imagen |
| **Umbral 0,18 (E1-29), holdout de 40 imágenes** | **0,666** | 0,614 | Elegido en tuning (80), reportado en holdout (D-14) |
| **Umbral 0,18, dev completo** | **0,653** | **0,596** | P 0,70 · R 0,61 · ~28 predicciones por imagen |

- **Por tamaño:** los objetos chicos son el cuello de botella. Solo el 40 % de las instancias chicas llega a TP de máscara, contra ~72 % en medium y large.
- **Dónde se pierde cada instancia (E1-31):** 55,5 % llega a TP de máscara; se pierde sobre todo en la detección (11,6 % sin ninguna caja, 8,4 % con otra clase, 16,6 % bajo el umbral). NMS y tope casi no pierden nada (1 %).
- **SAM2 con cajas GT (E1-12):** 84 % de las instancias con IoU ≥ 0,5; 72–75 % en las chicas (D-15).
- **Consistencia del SceneJSON (E1-19):** 0 inconsistencias en 6905 objetos; posición 97 %, tamaño 95 % y color 90 % contra GT (D-16).

**Preguntas (20 imágenes del benchmark, 60 consignas):**

| | Determinista (`questions`) | LLM (`questions_llm`, Qwen2.5-3B) |
|---|---|---|
| Coincidencia con el SceneJSON oráculo (40 cerradas) | 72,5 % | 72,5 % |
| Por tipo | count 40 % · existence 90 % · position 100 % · relation 80 % · size 60 % | igual |
| JSON válido / mismo DSL que el estructurado | — | 100 % / 100 % |
| Descripciones | plantilla determinista | 18/20 fieles; 2 con plantilla |
| Evidencia que sostiene la respuesta | 100 % | 100 % |

Los 11 errores son de **percepción** (conteos con instancias de más o de menos, una existencia alucinada y 4 objetos no detectados); el LLM no agrega errores de interpretación (E1-24).

**Submission:** 80 filas válidas, 2271 objetos, 164 796 relaciones, 15,4 MB, con `questions` y `questions_llm`. Aceptado por Kaggle el 2026-10-24 con **`PipelineScore = 0.6509`** (H1).

## Experimentos controlados (una variable cada uno, decididos con el holdout)

| Experimento | Resultado | Decisión |
|---|---|---|
| Umbral de detección 0,10 → 0,18 | F1 de cajas en holdout 0,619 → **0,666** | **Adoptado** (D-14) |
| NMS entre clases (IoU 0,8) | 0,6661 → 0,6667 | Descartado: no aporta (D-09, D-14) |
| Post-proceso de máscaras solo en chicas | Con cajas GT mejora (84,1 % → 85,2 %), end-to-end no (F1 de máscaras en holdout 0,614 → 0,609) | Descartado (D-17) |
| Prompt del intérprete v1 → v2 | JSON válido 0,65 → **1,00**; descripciones 0/20 → 18/20 fieles | **Adoptado** (D-18) |

## Checklist de promoción

| Requisito del enunciado | Estado |
|---|---|
| F1 de detección ≥ 0,50 | ✅ 0,653 en dev (0,666 en holdout) |
| Atributos y relaciones consultables de forma determinista | ✅ color, posición, tamaño y 7 relaciones |
| `questions_llm` con un LLM liviano ejecutable en Kaggle que recibe solo el texto | ✅ Qwen2.5-3B fp16, 6 GB, ~5 s por consulta en T4 (D-12) |
| Ubicación, comparación y relaciones en los dos modos | ✅ |
| Manejo explícito de preguntas ambiguas, no soportadas o sin información | ✅ estados `ambiguous`, `unsupported`, `insufficient_information` |
| Tracing completo usado para analizar la propagación de errores | ✅ `trace.jsonl` por etapa y embudo de E1-31 |
| Mejora justificada contra una línea base | ✅ tabla de experimentos |
| Submission válido evaluado por Kaggle | ✅ `PipelineScore = 0.6509` (H1) |
| Notebook, reporte y defensa | ⏳ E1-32, E1-33, E1-34 |

## Cómo correr todo en Kaggle

```python
!git clone -b desarrollo https://github.com/GEMajlis/TPI-IA-2026.git   # o: git pull
%cd /kaggle/working/TPI-IA-2026
!pip install -q -e ".[llm]"
C = "--config configs/base.yaml configs/kaggle.yaml"
!python scripts/run_pipeline.py {C} --split dev --eval          # percepción en dev + F1
!python scripts/run_pipeline.py {C} --split test benchmark      # las 80 imágenes del submission
R = "<carpeta que imprime la línea anterior>"
!python scripts/run_qa.py {C} --run {R}                         # questions
!python scripts/run_qa_llm.py {C} --run {R}                     # questions_llm
!python scripts/eval_qa.py {C} --run {R}                        # evaluación de QA
!python scripts/make_submission.py {C} --run {R} --out /kaggle/working/submission.csv
```

Evaluación sobre un run de dev: `run_eval.py` (F1, consistencia y atributos), `tune_thresholds.py` (umbral), `eval_segmentation.py` (oráculo de SAM2), `analyze_errors.py` (embudo y galería) y `compare_runs.py` (experimentos). La caché de detecciones y máscaras se reutiliza entre runs.

## Desarrollo local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                        # 459 tests, todos tienen que pasar
ruff check . && ruff format --check .
```

**Datos:** `data/dataset/` no se versiona. Descargá los datos de la competencia y descomprimilos ahí manteniendo la estructura (`development/`, `tpi_ia_2026_competition_public/`, `tpi_2026_qa_public/`, `scene_schema.json`, `sample_submission.csv`).

**Regla de oro (motivo de desaprobación):** el GT solo se lee desde `src/eval/` y `src/eda/`. Nada en `src/scene/` puede importarlos ni tocar anotaciones. Lo verifica `tests/test_import_boundary.py`; no lo desactiven. Las demás reglas están en [CLAUDE.md](../CLAUDE.md).

**Forma de trabajo:** todo en la rama `desarrollo`, un commit por tarea, `git pull` antes de empezar y antes de cada push. Antes de agregar una decisión, mirar el último número en `DECISIONS.md` (ya chocaron D-12 y D-13 una vez).

## Pendientes del grupo

- [x] **Subir el submission** y anotar el puntaje: `PipelineScore = 0.6509`.
- [ ] **Definir el número de grupo** (nombre de la notebook: `TPI_IA_2026P_<grupo>`).
- [ ] **Decidir si el repo de GitHub es público.** Si lo es, la notebook final puede clonarlo (es un recurso público que descarga la propia notebook). Si es privado, E1-32 tiene que meter el código en la notebook con `%%writefile`.
- [ ] **Repartir E1-32, E1-33 y E1-34.**
- [ ] **Pasar `desarrollo` a `main`** con un PR antes de entregar.
- [ ] **Preparar la defensa:** 10 minutos; todos tienen que poder explicar cualquier parte.

## Lo hecho, tarea por tarea

| Tarea | Qué quedó | Dónde |
|---|---|---|
| E1-00 a E1-05 · Infraestructura | Repo, config con semillas y `run_id`, carga de datos sin GT, schema, RLE, tracer | `src/scene/config.py`, `io/`, `masks/`, `tracing/`, `scene/schema.py` |
| E1-06 · EDA | 8 hallazgos con su implicancia | [eda_findings.md](eda_findings.md), `scripts/run_eda.py` |
| E1-07 · Evaluador | P, R y F1 federados LVIS para cajas y máscaras, por tamaño, frecuencia y categoría | `src/eval/lvis_federated.py` |
| E1-08 a E1-11 · Percepción | Vocabulario, OWLv2, post-proceso, SAM2 con caché | `src/scene/detection/`, `src/scene/segmentation/` |
| E1-12 · Oráculo de segmentación | SAM2 con cajas GT para separar el error de SAM2 del detector | `src/eval/segmentation.py`, `scripts/eval_segmentation.py` |
| E1-13 a E1-16 · Escena | Propiedades, color, relaciones y builder del SceneJSON con origen por campo | `src/scene/scene/` |
| E1-17 · Orquestador | Pipeline completa con caché por etapa y traza; acepta imágenes sueltas | `src/scene/pipeline.py`, `scripts/run_pipeline.py` |
| E1-18 · Submission | 80 filas validadas y aceptadas por Kaggle; `PipelineScore = 0.6509` (H1) | `src/scene/io/submission.py`, `scripts/make_submission.py` |
| E1-19 · Consistencia | Chequeo del SceneJSON contra su máscara y exactitud de atributos contra GT | `src/eval/scene_consistency.py`, `scripts/run_eval.py` |
| E1-20 a E1-23 · QA determinista | DSL, category matcher en español, ejecutor y `questions` | `src/scene/qa/`, `scripts/run_qa.py` |
| E1-24 · Evaluación de QA | Oráculo desde GT, fidelidad de la evidencia y atribución de errores | `src/eval/qa_eval.py`, `src/eval/oracle.py`, `scripts/eval_qa.py` |
| E1-25 a E1-28 · LLM | Qwen2.5-3B, intérprete (prompt v2), descriptor verificado y `questions_llm` | `src/scene/llm/`, `src/scene/qa/llm_mode.py`, `scripts/run_qa_llm.py` |
| E1-29 · Ajuste de umbral | Barrido en tuning/holdout sobre la caché cruda | `src/eval/tuning.py`, `scripts/tune_thresholds.py` |
| E1-30 · Experimentos | Comparación controlada de runs y registro en `experiments.csv` | `configs/experiments/`, `scripts/compare_runs.py` |
| E1-31 · Análisis de errores | Embudo por instancia, tipos de FP, peores clases, galería y atribución de QA | [error_analysis.md](error_analysis.md), `scripts/analyze_errors.py` |
