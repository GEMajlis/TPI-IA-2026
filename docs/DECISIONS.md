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

### D-05 · **resuelta (E1-20, E1-21)** · Benchmark de 60 consignas, sin preguntas de atributo
- `questions_dev.jsonl` tiene 60 consignas sobre 20 imágenes: `description` 20, `count` 10, `existence` 10, `relation` 10, `position` 5, `size` 5. TASKS hablaba de 40 preguntas (se corrigió en E1-23).
- No hay preguntas de color/atributo: la op `attribute` del DSL no se evalúa en el benchmark (se puede mantener, pero no es prioritaria).
- Las preguntas están en **español** (`class_es`) y traen el campo estructurado `class_en` con el nombre del vocabulario. En modo LLM solo se recibe el texto: el category matcher (E1-21) tiene que mapear nombres en español (necesita un léxico es→en del vocabulario).
- Las relaciones del benchmark se formulan con instancias en singular ("¿La instancia de X está por encima de la instancia de Y?").
- **Resolución:** el DSL conserva la op `attribute` (sin prioridad en el ejecutor). El matcher usa un léxico es→en manual (`configs/lexicon_es.yaml`) que resuelve las 50 clases en español del benchmark. El predicado de las relaciones lo extrae del texto el adapter (E1-23) o el LLM (E1-26).

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

### D-10 · **adoptada (E1-22)** · Preguntas singulares con varias instancias detectadas
- PLAN §4.5 responde `ambiguous` con la evidencia de todas. Pero en el benchmark cada clase de una pregunta de posición, tamaño o relación tiene exactamente 1 instancia en el GT (EDA H8), y `answer` es obligatorio.
- **Propuesta:** responder con la instancia de mayor confianza, bajar la confianza de la respuesta y guardar el estado `ambiguous` en un campo propio, para medirlo en el análisis de errores.
- **Implementación (E1-22):** `normalized_answer` sale de la instancia de mayor confianza; `evidence` lleva todas (la elegida primero); la confianza es `c_max × c_max / Σc` (la parte de la confianza total que corresponde a la elegida); `status = "ambiguous"` y `status_detail` dicen cuántas instancias había. En las relaciones se aplica a cada lado, y el objeto nunca puede ser la misma instancia que el sujeto.

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

## 2026-10-24 · E1-20 y E1-21

### D-13 · Léxico español manual y LaBSE como último recurso del category matcher
- **Decisión:** el orden es nombre → léxico español → sinónimos LVIS → embeddings. El léxico (`configs/lexicon_es.yaml`) es una traducción manual de las 120 categorías con variantes regionales (heladera/nevera, celular/móvil, etc.). Para el fallback se usa `sentence-transformers/LaBSE` con umbral de similitud coseno **0,90**. El modelo se carga recién cuando el léxico no alcanza.
- **Alternativas:** `paraphrase-multilingual-MiniLM-L12-v2` y `intfloat/multilingual-e5-small`. Se descartan porque mezclan mucho las distribuciones de positivos y negativos.
- **Evidencia** (`scripts/calibrate_matcher.py`: 18 términos dentro del vocabulario que el léxico no resuelve y 22 términos fuera del vocabulario):

  | Modelo | Similitud máx. de un negativo | Falsos positivos con 0,90 | Aciertos | Categoría errónea |
  |---|---|---|---|---|
  | LaBSE | 0,887 ("puerta" → doorknob) | 0 / 22 | 5 / 18 | 1 ("gorro" → baseball cap) |
  | MiniLM-L12 multilingüe | 0,959 ("puerta" → doorknob) | 3 / 22 | 3 / 18 | 1 |
  | multilingual-e5-small | 0,939 | 15 / 22 | 11 / 18 | 4 |

  El umbral prioriza la precisión: preferimos `unsupported` antes que contestar sobre otra categoría. El fallback recupera poco (5 de 18), así que la cobertura en español depende sobre todo del léxico.
- **Limitación:** el léxico incluye las traducciones que usa el benchmark (`class_es`), porque son las traducciones naturales. Por eso la cobertura del 100 % sobre el benchmark está sobreestimada para preguntas redactadas de otra forma. La muestra de calibración es chica (40 términos).

## 2026-10-24 · E1-12, E1-19 y E1-29 (corridas completas en Kaggle, 120 imágenes de dev)

### D-14 · Umbral de detección 0,18 (antes 0,1)
- **Decisión:** `detection.score_threshold = 0.18`, elegido como el máximo F1 de cajas en el split de tuning (80 imágenes, estratificado por cantidad de instancias, semilla 42) y reportado en holdout (40).
- **Evidencia (holdout):** con 0,10 → P 0,55 · R 0,71 · F1 **0,619**; con 0,18 → P 0,72 · R 0,62 · F1 **0,666** (+4,7 puntos). Predicciones por imagen: ~57 → ~28. En tuning el F1 con 0,18 es 0,646, así que el holdout no muestra sobreajuste.
- **NMS entre clases (IoU 0,8):** F1 holdout 0,6667 vs 0,6661 sin él. No aporta, así que sigue apagado (confirma D-09).
- **Efecto colateral:** la mitad de objetos por escena reduce las relaciones y el peso del submission.
- Salidas: `outputs/tuning/1bc16d79/` (curva, split y resumen) en el run de Kaggle.

### D-15 · Diagnóstico de la segmentación con oráculo (E1-12)
- SAM2.1 hiera-large con **cajas GT** (1790 instancias): IoU ≥ 0,5 en el 84 % (mIoU 0,74). Por tamaño, cruda → post-procesada: small 72 % → 75 %, medium 96 % → 94 %, large 97 % → 95 %.
- **Lectura:** el error de máscaras en objetos chicos es en buena parte de SAM2 (aun con cajas perfectas, 1 de cada 4 no llega a IoU 0,5). El post-proceso ayuda en small y perjudica en medium y large (probablemente "componente principal" corta partes de objetos ocluidos).
- **Propuesta para E1-30:** aplicar el post-proceso solo a máscaras chicas. Con cajas predichas, 176 de los 1264 TP de cajas se pierden en máscaras; el `mask_score` correlaciona 0,75 con el IoU real.

### D-16 · Consistencia de la representación (E1-19)
- 120 escenas, 6905 objetos, 924 198 relaciones: **0 inconsistencias**. Atributos en pares TP (1140): posición 97 %, tamaño 95 %, color 90 % (confusiones entre tonos vecinos: negro/gris, gris/blanco, marrón/negro).
- **Riesgo abierto:** en el 9 % de los objetos más del 10 % de la máscara cae fuera de la caja entregada (por el margen de recorte de 0,1). No viola el schema, pero no sabemos si la métrica de consistencia de Kaggle compara caja y máscara. Candidato a experimento: `box_margin: 0` o entregar como `bbox` la caja de la máscara.

## 2026-10-24 · E1-30 (experimentos controlados)

### D-17 · Post-proceso solo en máscaras chicas: **descartado**
- **Hipótesis (D-15):** el post-proceso ayuda en small y perjudica en medium/large; aplicarlo solo a máscaras crudas con área relativa < 0,00333 sube el F1 de máscaras.
- **Cambio (única variable):** `segmentation.postprocess.apply_below_area_ratio: null → 0.00333` (`configs/experiments/e1_30_pp_solo_chicas.yaml`). Línea base: run de dev con umbral 0,18.
- **Oráculo (cajas GT, 1790 instancias), IoU ≥ 0,5:** cruda 83,5 % · post-proceso en todas 84,1 % · **solo en chicas 85,2 %** (small 74,9 %, medium 96,1 %, large 97,3 %). Con cajas perfectas la hipótesis se cumple.
- **De punta a punta (cajas predichas), F1 de máscaras:** tuning 0,586 → 0,596 (+0,96); **holdout 0,614 → 0,609 (−0,54)**. F1 de cajas sin cambios (0,666 en holdout). Por tamaño en holdout: small −0,17, medium −0,93, large −1,32 puntos.
- **Decisión:** no se adopta (se decide con holdout; además, ±1 punto en 40 imágenes son 2–3 instancias, indistinguible del ruido).
- **Lectura:** las cajas de OWLv2 son más flojas que las GT; SAM2 se derrama fuera del objeto y el recorte a la caja del post-proceso es lo que lo corrige en medium/large. Con cajas GT ese recorte no hace falta, por eso el oráculo lo ve como perjuicio. Con el experimento sube la fracción de máscaras con > 10 % fuera de la caja (5,6 % → 6,0 %). **Una mejora medida con el oráculo no garantiza una mejora end-to-end: el error de detección se propaga a la segmentación.**
- Registro: `outputs/experiments.csv` en el run de Kaggle (`exp_run` 20261024-193053-56d96386).

### D-18 · Prompt del intérprete v2 sin contenido del benchmark (E1-26)
- **Problema:** `interpreter_v1.txt` usaba como ejemplos categorías del benchmark (reloj, anillo, collar, mochila) y una de sus preguntas casi literal ("¿El reloj está por encima del anillo?"). Así, la validez del JSON y la exactitud del modo LLM medidas sobre el benchmark salen infladas, y en la defensa se puede objetar como ajuste sobre el conjunto de evaluación. Además, la v1 no tenía ningún ejemplo de `describe`, que es el 33 % de las consignas.
- **Decisión:** `interpreter_v2.txt` (y `llm.interpreter.prompt_version` apunta a la v2). Los ejemplos usan categorías que no están en el benchmark (taza, libro, paraguas, gato, lámpara, silla, auto), cubren todas las ops (incluidos `describe`, `attribute`, filtro de color y `left_of`) y combinan la redacción coloquial con la de "la instancia de la categoría X", que es el formato de entrada esperado (formato, no contenido). La v1 se conserva para trazabilidad.
- **Control:** `tests/test_interpreter_prompt.py` falla si un ejemplo del prompt usa una categoría (en inglés o en español, con plurales simples) o una pregunta del benchmark, y si algún ejemplo no es un DSL válido. Sobre la v1 detecta anillo, collar, mochila y reloj.
- También se cambió el prompt de latencia de `scripts/benchmark_llm.py`, que repetía la misma pregunta.

