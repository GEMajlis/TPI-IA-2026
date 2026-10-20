# Decisiones y desvíos respecto de `docs/PLAN.md`

Formato: fecha · decisión · alternativa descartada · evidencia. Las entradas marcadas **pendiente** se cierran en la tarea indicada.

## 2026-10-18 · Fundaciones (E1-00 a E1-05)

### D-01 · Ubicación de los datos
- **Decisión:** `data/dataset/` es una copia completa de la fuente de datos de la competencia, con la misma estructura que monta Kaggle (`development/`, `tpi_ia_2026_competition_public/`, `tpi_2026_qa_public/`, `scene_schema.json`, `sample_submission.csv` en la raíz). Las rutas en `configs/base.yaml` son relativas a `paths.data_root`; `configs/kaggle.yaml` solo cambia esa raíz y `device`.
- **Alternativa:** rutas sueltas por archivo. Se descarta para que el mismo código corra igual local y en Kaggle.
- **Evidencia:** `README_COMPETITION_DATA.md` describe ese layout. El slug exacto de `/kaggle/input/...` hay que confirmarlo en la primera corrida en Kaggle.

### D-02 · Python local 3.12 y `lvis` opcional
- **Decisión:** venv local con Python 3.12 (el sistema trae 3.14, sin garantía de wheels para torch/SAM2); `requires-python >= 3.10` por Kaggle. `lvis` pasa a extra `.[lvis]`.
- **Evidencia:** `lvis` solo se usa como referencia de AP (PLAN §4.7) y no aporta a las métricas principales.

### D-03 · El tracer relanza los errores
- **Decisión:** `Tracer.stage` registra el error y lo relanza; `Tracer.image_guard` lo atrapa por imagen para que el run siga.
- **Alternativa:** que `stage` trague la excepción. Se descarta porque el código posterior seguiría con variables sin definir.

## Hallazgos en los archivos normativos (a resolver en E1-03)

### D-04 · **pendiente (E1-03 / E1-15)** · `near` y `on_top_of` no están en el enum del schema
- `scene_schema.json` restringe `relations[].predicate` a `left_of, right_of, above, below, overlaps`. PLAN §4.3 y E1-15 incluyen `near` y `on_top_of`: si se guardan en `relations`, el JSON **no valida**.
- **Propuesta:** guardar en `relations` solo los cinco predicados del schema y, si se implementan, poner `near` / `on_top_of` en un campo propio (por ejemplo `extra_relations`), que el schema admite (`additionalProperties: true`).

### D-05 · **pendiente (E1-20)** · Benchmark de 60 consignas, sin preguntas de atributo
- `questions_dev.jsonl` tiene 60 consignas sobre 20 imágenes: `description` 20, `count` 10, `existence` 10, `relation` 10, `position` 5, `size` 5. TASKS hablaba de 40 preguntas (se corrigió en E1-23).
- No hay preguntas de color/atributo: la op `attribute` del DSL no se evalúa en el benchmark (se puede mantener, pero no es prioritaria).
- Las preguntas están en **español** (`class_es`) y traen el campo estructurado `class_en` con el nombre del vocabulario. En modo LLM solo se recibe el texto: el category matcher (E1-21) tiene que mapear nombres en español (necesita un léxico es→en del vocabulario).
- Las relaciones del benchmark se formulan con instancias en singular ("¿La instancia de X está por encima de la instancia de Y?").

### D-06 · **pendiente (E1-06 / E1-13)** · El schema no fija umbrales de tamaño
- `size` es `small | medium | large` sin umbrales; se proponen con el EDA. Tampoco define la convención exacta de la grilla 3×3 ni del centroide (se asume `[x, y]` en píxeles, igual que el ejemplo).

### D-07 · **pendiente (E1-08)** · `categories.csv` no trae sinónimos
- El vocabulario ya viene limpio (sin `_` ni paréntesis). Para el experimento de sinónimos hace falta una fuente pública (metadata de categorías de LVIS v1) descargada en tiempo de ejecución.

### D-08 · Métrica de Kaggle
- `PipelineScore` = 40 % F1 de cajas + 35 % F1 de máscaras + 15 % consistencia estructural + 10 % QA (promedio de `questions` y `questions_llm`, solo en las 20 imágenes del benchmark). La segmentación pesa casi tanto como la detección: priorizar la calidad de SAM2 y la consistencia de los campos derivados.
