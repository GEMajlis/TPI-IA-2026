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

## Qué hicimos (Etapa 1: 11 de 35 tareas)

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
| E1-10 Post-proceso | Umbral, NMS por clase, NMS entre clases (apagado) y tope, con conteos en la traza | `src/scene/detection/postprocess.py` |

OWLv2 ya corre de punta a punta en CPU (smoke test: 2 imágenes de dev, ~12 s por imagen, F1 de cajas 0,65 en esa muestra chica). Falta la primera corrida completa en Kaggle.

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
2. **Primera corrida en Kaggle:** `python scripts/run_detection.py --config configs/base.yaml configs/kaggle.yaml --split dev --eval` (120 imágenes de dev y F1 de cajas). Ahí sabemos qué tan lejos estamos del 0,50.
3. **E1-11 y E1-13 · SAM2 y propiedades básicas** (centroide, área, posición, tamaño), lo que habilita el F1 de máscaras.
4. **E1-16, E1-17 y E1-18 en versión mínima:** builder, orquestador y submission, para subir a Kaggle y obtener el primer puntaje (**hito H1**).

En paralelo, porque no dependen de los modelos (se trabajan con SceneJSON sintéticos):
- **E1-14 y E1-15:** color dominante y relaciones espaciales.
- **E1-20, E1-21 y E1-22:** DSL, category matcher y ejecutor determinista. El matcher tiene que entender nombres en español.
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
  - **D-10:** qué responder cuando una pregunta en singular encuentra varias instancias.
- [ ] **Repartir las tareas** según las líneas de [TASKS.md](TASKS.md): PER, ESC, QA y LLM se pueden llevar en paralelo. Cada tarea va en su rama `task/<ID>-<slug>`.
