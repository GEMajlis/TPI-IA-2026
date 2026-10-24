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

### D-04 · **adoptada (E1-03); se implementa en E1-15** · `near` y `on_top_of` no están en el enum del schema
- `scene_schema.json` restringe `relations[].predicate` a `left_of, right_of, above, below, overlaps`. PLAN §4.3 y E1-15 incluyen `near` y `on_top_of`: si se guardan en `relations`, el JSON **no valida**.
- **Propuesta:** guardar en `relations` solo los cinco predicados del schema y, si se implementan, poner `near` / `on_top_of` en un campo propio (por ejemplo `extra_relations`), que el schema admite (`additionalProperties: true`).

### D-05 · **pendiente (E1-20)** · Benchmark de 60 consignas, sin preguntas de atributo
- `questions_dev.jsonl` tiene 60 consignas sobre 20 imágenes: `description` 20, `count` 10, `existence` 10, `relation` 10, `position` 5, `size` 5. TASKS hablaba de 40 preguntas (se corrigió en E1-23).
- No hay preguntas de color/atributo: la op `attribute` del DSL no se evalúa en el benchmark (se puede mantener, pero no es prioritaria).
- Las preguntas están en **español** (`class_es`) y traen el campo estructurado `class_en` con el nombre del vocabulario. En modo LLM solo se recibe el texto: el category matcher (E1-21) tiene que mapear nombres en español (necesita un léxico es→en del vocabulario).
- Las relaciones del benchmark se formulan con instancias en singular ("¿La instancia de X está por encima de la instancia de Y?").

### D-06 · **propuesta (E1-06)** · El schema no fija umbrales de tamaño
- `size` es `small | medium | large` sin umbrales. Propuesta del EDA (H2): `area_ratio` < 0,333 % → small, < 3 % → medium, si no large (COCO 32²/96² px sobre 640×480). Queda en `scene.size_thresholds`; si el enunciado define otros, se cambian ahí.
- Las convenciones de grilla 3×3, centroide y bbox quedan en `docs/schema_notes.md` §6.

### D-07 · **pendiente (E1-08)** · `categories.csv` no trae sinónimos
- El vocabulario ya viene limpio (sin `_` ni paréntesis). Para el experimento de sinónimos hace falta una fuente pública (metadata de categorías de LVIS v1) descargada en tiempo de ejecución.

### D-08 · Métrica de Kaggle
- `PipelineScore` = 40 % F1 de cajas + 35 % F1 de máscaras + 15 % consistencia estructural + 10 % QA (promedio de `questions` y `questions_llm`, solo en las 20 imágenes del benchmark). La segmentación pesa casi tanto como la detección: priorizar la calidad de SAM2 y la consistencia de los campos derivados.

## 2026-10-24 · E1-03, E1-06, E1-07 y E1-08

### D-09 · Sin NMS entre clases en la línea base (cambia PLAN §4.1)
- **Decisión:** la línea base usa solo NMS por clase. El NMS entre clases pasa a ser un experimento (E1-30).
- **Alternativa:** PLAN §4.1 lo ponía en la base con IoU ≈ 0,8.
- **Evidencia:** EDA H6. El GT tiene 25 pares de clases distintas con IoU de caja ≥ 0,8, y varios son el mismo objeto con dos etiquetas (cup/drinking glass, hat/baseball cap, television set/computer monitor). El NMS entre clases eliminaría TP reales.

### D-10 · **pendiente (E1-22)** · Preguntas singulares con varias instancias detectadas
- PLAN §4.5 responde `ambiguous` con la evidencia de todas. Pero en el benchmark cada clase de una pregunta de posición, tamaño o relación tiene exactamente 1 instancia en el GT (EDA H8), y `answer` es obligatorio.
- **Propuesta:** responder con la instancia de mayor confianza, bajar la confianza de la respuesta y guardar el estado `ambiguous` en un campo propio, para medirlo en el análisis de errores.

### D-11 · Sinónimos LVIS desde detectron2
- **Decisión:** se descarga en tiempo de ejecución `detectron2/data/datasets/lvis_v1_categories.py` (público). Se parsea con `ast.literal_eval`, sin ejecutar código, y se cachea en `data/cache/lvis_synonyms.json` junto con el sha256 de la fuente.
- **Alternativa:** `lvis_v1_val.json` (cientos de MB, y además trae anotaciones). Se descarta.
- **Evidencia:** 117 de 120 nombres coinciden directamente; `laptop`, `cell phone` y `computer monitor` se mapean con `vocab.lvis_aliases`.

## 2026-10-24 · LLM (E1-25 a E1-27, en desarrollo)

### D-12 · **pendiente de corrida T4** · Selección del LLM
- **Implementación:** los pesos se descargan en tiempo de ejecución desde Hugging Face. Kaggle
  aporta la T4; los modelos públicos no requieren credenciales. Para un repositorio privado se usa
  `HF_TOKEN` como Secret de Kaggle, nunca dentro del código.
- **Candidatos:** Qwen2.5-3B-Instruct en fp16 y Qwen3-4B en 4-bit NF4. Qwen2.5 queda como
  valor provisional de desarrollo hasta medir recursos y validez del DSL.
- **Protocolo:** una pasada de calentamiento y tres generaciones medidas con los mismos prompts;
  semilla 42, greedy decoding y T4. Comando:
  `pip install -e ".[llm]" && python scripts/benchmark_llm.py --config configs/base.yaml configs/kaggle.yaml`.

| Candidato | Precisión | Memoria de carga (MiB) | Latencia media (ms) | p95 (ms) | Decisión |
|---|---:|---:|---:|---:|---|
| Qwen2.5-3B-Instruct | fp16 | pendiente T4 | pendiente T4 | pendiente T4 | provisional |
| Qwen3-4B | 4-bit NF4 | pendiente T4 | pendiente T4 | pendiente T4 | pendiente |

No se completan E1-25/E1-26/E1-27 hasta reemplazar estos pendientes por mediciones reales,
validar al menos 95 % del JSON sobre el benchmark e inspeccionar las 20 descripciones de E1-16.
