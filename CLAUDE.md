# CLAUDE.md — Reglas para agentes

Proyecto: **TPI Inteligencia Artificial 2026 (UTN FRC) — Comprensión visual de escenas.**
Pipeline: imagen → detección open-vocabulary → SAM2 → SceneJSON → preguntas (determinista y LLM).

Antes de tomar una tarea leé `docs/PLAN.md` (decisiones técnicas) y la tarea correspondiente en `docs/TASKS.md`.

## Regla de oro (motivo de desaprobación si se rompe)

- **Las anotaciones (GT) solo se leen desde `src/eval/` y `src/eda/`.**
- Nada en `src/scene/` (inferencia) puede importar `src/eval/` ni `src/eda/`, ni abrir archivos de anotaciones.
- La inferencia debe poder procesar **una imagen arbitraria** sin metadatos de LVIS más allá del vocabulario de categorías.
- Está prohibido usar GT para elegir clases presentes, completar objetos, corregir cajas o máscaras, construir el JSON predicho o responder preguntas.
- El LLM **nunca recibe la imagen** y nunca calcula respuestas: solo produce el JSON intermedio y verbaliza.
- Existe un test (`tests/test_import_boundary.py`) que verifica esta separación. No lo desactives.

## Flujo de trabajo por tarea

1. Leé la tarea completa: objetivo, dependencias, entregables y criterios de aceptación.
2. Verificá que las dependencias estén terminadas. Si no lo están, avisá en lugar de improvisar interfaces.
3. Trabajá en la rama `desarrollo` (una sola rama para todas las tareas; no se crean ramas por tarea). Un commit por tarea, con el ID en el mensaje.
4. Escribí los tests primero o junto con el código. `pytest` tiene que pasar completo antes de cerrar.
5. Actualizá el estado de la tarea en `docs/TASKS.md` (`[ ]` → `[x]`) y agregá una nota breve si cambió alguna interfaz.
6. Si una decisión de `docs/PLAN.md` resulta inviable, **no la cambies en silencio**: documentá el motivo en `docs/DECISIONS.md` (fecha, decisión, alternativa, evidencia).

## Convenciones

- Python 3.10+ (versión de Kaggle). Type hints en funciones públicas y dataclasses o pydantic para los contratos.
- Estilo con `ruff` y formato con `ruff format`.
- Todo parámetro ajustable va en `configs/*.yaml`. Nada de números mágicos en el código.
- Semillas fijas mediante `scene.config.set_seed`. Cada ejecución genera un `run_id`.
- Las salidas van a `outputs/runs/<run_id>/` (en gitignore). Los resultados cuantitativos se guardan en CSV o JSON, nunca solo impresos.
- Los modelos se descargan desde Hugging Face u otra fuente pública **en tiempo de ejecución**. No se commitean pesos.
- **GPU:** el desarrollo local puede ser en CPU. Todo wrapper de modelo debe aceptar `device` y permitir un smoke test con 2 o 3 imágenes. Las corridas completas se hacen en Kaggle.
- Comentarios y docstrings en español, breves. Evitá código o comentarios que el grupo no pueda explicar en la defensa.
- Cada campo del SceneJSON indica si es **observado** (salida de modelo), **derivado** (calculado con código) o **desconocido** (`null`). Nunca se inventa un valor.

## Archivos normativos

`data/competition/` contiene `scene_schema.json`, `sample_submission.csv` y el benchmark de preguntas. **Son la referencia normativa** para nombres de campos, enums y formatos. Ante una duda, gana el schema y no este documento. Su resumen está en `docs/schema_notes.md` (tarea E1-03).

## Comandos

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
python scripts/run_pipeline.py --config configs/base.yaml --split dev --limit 3   # smoke test
python scripts/run_eval.py --run outputs/runs/<run_id>
python scripts/make_submission.py --run outputs/runs/<run_id>
python scripts/build_notebook.py   # genera notebooks/TPI_IA_2026P_<grupo>.ipynb
```
