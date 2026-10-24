# Backlog de tareas

Cada tarea tiene ID, objetivo, dependencias, entregables y criterios de aceptación. Un agente toma una tarea cuando sus dependencias están en `[x]`.
Las **líneas** agrupan tareas que se pueden llevar en paralelo: **INF** infraestructura, **EVA** evaluación, **PER** percepción, **ESC** escena, **QA** preguntas, **LLM**, **ENT** entrega.

Las referencias a campos del SceneJSON se resuelven siempre contra `data/competition/scene_schema.json`.

---

# ETAPA 1 — Promoción

## Fundaciones

### [x] E1-00 · Bootstrap del repo — INF
- **Objetivo:** estructura de carpetas de `docs/PLAN.md §2`, `pyproject.toml` (deps: torch, transformers, pycocotools, lvis, jsonschema, pydantic, pyyaml, numpy, opencv-python, scikit-learn, pandas, matplotlib, pytest, ruff), `.gitignore` (data/dataset, outputs, pesos), README breve.
- **Depende de:** —
- **Aceptación:** `pip install -e ".[dev]"` funciona; `pytest` y `ruff check` corren (aunque sin tests todavía).
- **Nota:** `lvis` quedó como extra opcional (`.[lvis]`), solo se usa como referencia de AP. Datos locales en `data/dataset/` (misma estructura que el mount de Kaggle); override `configs/kaggle.yaml`. Venv local con Python 3.12.

### [x] E1-01 · Config, semillas y run_id — INF
- **Objetivo:** `scene/config.py` con carga de YAML y overrides (`base.yaml` + `experiments/x.yaml`), `set_seed`, `run_id` (timestamp + hash corto de la config), creación de `outputs/runs/<run_id>/` y `run_manifest.json` (config completa, versiones de librerías, dispositivo).
- **Depende de:** E1-00
- **Aceptación:** tests de merge de overrides; misma config → mismo `config_hash`.
- **Nota (interfaz):** `load_config(*paths, dotlist=[...])` combina YAML de izquierda a derecha y admite overrides `clave.sub=valor`; `start_run(cfg) -> RunContext` (run_id, run_dir, device, `trace_path`, `update_manifest`). `data_path(cfg, key)` resuelve rutas contra `paths.data_root`.

### [x] E1-02 · Carga de datos y frontera GT — INF
- **Objetivo:**
  - `scene/io/dataset.py`: lista imágenes (dev y test) y vocabulario de categorías (id, nombre, sinónimos) **sin leer anotaciones de instancias**.
  - `eval/annotations.py`: carga GT por imagen (instancias, `neg_category_ids`, `not_exhaustive_category_ids`).
  - `tests/test_import_boundary.py`: analiza con AST que ningún módulo de `src/scene/` importe `eval` ni `eda`, ni referencie archivos de anotaciones.
- **Depende de:** E1-00
- **Aceptación:** el test de frontera pasa y falla si se agrega un import prohibido (testeado).
- **Nota (interfaz):** `list_images(cfg, split)` con splits `dev` (120), `test` (60) y `benchmark` (20); las dimensiones se leen del archivo de imagen, no de la metadata de GT. `categories.csv` no trae sinónimos: `Category.synonyms` queda vacío hasta E1-08. El test de frontera además prohíbe las claves `eval`, `annotations`, `neg_category_ids` y `not_exhaustive_category_ids` en `src/scene/`, e importa todo `scene` en un subproceso para verificar que no se cargue `eval`/`eda`.

### [x] E1-03 · Lectura de archivos normativos — INF
- **Objetivo:** leer `scene_schema.json`, `sample_submission.csv` y el benchmark de preguntas. Escribir `docs/schema_notes.md` con campos obligatorios, enums (posiciones, tamaños, relaciones), umbrales definidos, formato de `questions` / `questions_llm` / `interpreted_query`, y los tipos de pregunta y campos estructurados del benchmark con ejemplos. Implementar `scene/scene/schema.py` con validación por `jsonschema`.
- **Depende de:** E1-00
- **Aceptación:** el SceneJSON de ejemplo de `sample_submission.csv` valida; un JSON con un campo obligatorio faltante falla. `schema_notes.md` lista explícitamente todo lo que el schema **no** define y que el grupo debe decidir.
- **Nota (interfaz):** `validate_scene(scene, schema_path, structural=True) -> list[str]` y `assert_valid_scene(...)`. Además del schema, `structural_errors` chequea ids únicos, `mask.size == [H, W]`, suma de counts y referencias de relaciones/evidencia. Constantes `POSITIONS`, `SIZES`, `RELATION_PREDICATES` y `QUESTION_TYPES` (testeadas contra el schema). Las 80 filas de `sample_submission.csv` validan.

### [x] E1-04 · RLE no comprimido column-major — INF
- **Objetivo:** `scene/masks/rle.py` con `encode(mask) -> {"size":[H,W],"counts":[...]}` y `decode(rle) -> mask`.
- **Depende de:** E1-00
- **Aceptación:** round-trip exacto en máscaras aleatorias, vacías, llenas, con bordes y de 1 píxel; decodificación equivalente a `pycocotools` (`frPyObjects` sobre RLE no comprimido); los counts empiezan por la corrida de ceros (puede ser 0).
- **Nota:** además `rle.area(rle)` sin decodificar; test contra el RLE de `sample_submission.csv`.

### [x] E1-05 · Tracer — INF
- **Objetivo:** `scene/tracing/tracer.py` con un context manager `with tracer.stage(image_id, "detection", params=...) as s:` que registre tiempos, estado, n_in/n_out, artifact_path y la excepción si la hay (sin abortar el run completo), escribiendo en `trace.jsonl`.
- **Depende de:** E1-01
- **Aceptación:** test con una etapa que falla: queda registrada con `status=error` y el traceback, y el pipeline sigue con la imagen siguiente.
- **Nota (interfaz):** `tracer.stage(...)` registra y **relanza** el error (para no seguir con datos incompletos); `tracer.image_guard(image_id)` lo atrapa por imagen y el run sigue. Campo `extra` para conteos por filtro y `gpu_mem_peak_mb` si hay CUDA.

### [x] E1-06 · EDA — EVA
- **Objetivo:** `src/eda/` más una sección de notebook. Cada gráfico responde una pregunta y termina en una decisión. Mínimo:
  - instancias por imagen;
  - distribución de área relativa de los objetos (porcentaje de objetos chicos);
  - frecuencia de categorías;
  - proporción de categorías positivas, negativas, no exhaustivas y no verificadas por imagen;
  - superposición entre instancias (oclusión);
  - distribución de los centroides en la grilla 3×3;
  - tipos y objetos del benchmark de preguntas.
- **Depende de:** E1-02
- **Aceptación:** `docs/eda_findings.md` con 5 a 8 hallazgos, cada uno con su implicancia (por ejemplo: "X % de los objetos tiene área relativa < 1 % → evaluar tiling"). Propone los umbrales de tamaño si el schema no los fija.
- **Nota:** `scripts/run_eda.py` genera tablas, figuras y `eda_summary.json` en `outputs/eda/`. La sección de notebook se arma en E1-32 a partir de ese script. Parámetros nuevos en `base.yaml`: `scene.size_thresholds`, `detection.max_detections`, `eval.frequency_bins`.

### [x] E1-07 · Evaluador federado LVIS — EVA
- **Objetivo:** `eval/lvis_federated.py`. Por imagen y categoría, matching greedy por score con IoU ≥ 0,5 (cajas o máscaras). TP/FN sobre GT anotado; FP solo en categorías negativas, o en positivas no matcheadas que **no** sean no exhaustivas; el resto se ignora. Reporta P, R, F1 y **cobertura evaluable** (fracción de predicciones que entraron en la cuenta de precisión), global, por tamaño y por frecuencia de clase.
- **Depende de:** E1-02, E1-04
- **Aceptación:** tests con casos sintéticos que reproducen el ejemplo horse/car/chair del enunciado; predicción perfecta → F1 = 1; salida en `metrics_detection.json` y un CSV por categoría.
- **Nota (interfaz):** `evaluate(preds, gt, image_ids, iou_type="bbox"|"segm", ...)` o `evaluate_from_config(preds, gt, cfg, iou_type)` → `FederatedResult` (summary, by_size, by_frequency, per_category, events). `predictions_from_scenes(scenes)` convierte SceneJSON. `save_metrics(result, run_dir, name)`. F1 global = micro; también se reporta `f1_macro`. El test horse/car/chair se armó con las reglas del README del dataset (falta el enunciado para confirmar el ejemplo exacto). El GT contra sí mismo da F1 = 1 en cajas y máscaras.

## Percepción

### [x] E1-08 · Vocabulario y prompts — PER
- **Objetivo:** `detection/vocabulary.py`: limpieza de nombres, sinónimos opcionales desde el metadata de categorías, y mapeo query → category_id. Configurable (`vocab.mode: clean | synonyms`).
- **Depende de:** E1-02
- **Aceptación:** tests de limpieza; las 120 categorías quedan mapeadas sin colisiones.
- **Nota (interfaz):** `vocabulary_from_config(cfg, categories) -> Vocabulary` (`queries`, `prompts()`, `query_to_category()`, `by_category()`, `dropped`). Los sinónimos vienen de `lvis_v1_categories.py` de detectron2 (público), se descargan una vez y se cachean en `data/cache/lvis_synonyms.json`. Hay 3 nombres que la competencia renombró y se mapean con `vocab.lvis_aliases`. Modo `clean`: 120 queries; modo `synonyms`: 215 queries, sin colisiones.

### [x] E1-09 · Interfaz Detector + OWLv2 (línea base) — PER
- **Objetivo:** `detection/base.py` (interfaz `detect(image) -> list[Detection]`) y `detection/owlv2.py`. Cajas en coordenadas de la imagen original (atención al padding cuadrado de OWLv2), batching de queries, `device` configurable.
- **Depende de:** E1-08, E1-05
- **Aceptación:** smoke test en CPU con 2 imágenes; las cajas dibujadas coinciden visualmente con los objetos (guardar una imagen de debug); se traza la etapa.
- **Nota (interfaz):** `Detection` y `Detector.detect(image) -> list[Detection]` en `detection/base.py` (más `save_detections`/`load_detections`). `Owlv2Detector` y `build_detector(cfg, vocab, device)` en `detection/owlv2.py`: cajas × lado mayor (padding cuadrado), por clase queda la query de mayor score (sinónimos), `labels_per_box` categorías por caja, piso `score_floor` y tope `raw_topk` para la caché cruda. fp16 en CUDA. `scripts/run_detection.py` corre detección + post-proceso con traza, caché (`detections_raw/`, `detections/`), `--debug` y `--eval` (F1 de cajas). Smoke test en CPU sobre 2 imágenes de dev: ~12 s/imagen, F1 = 0,65 (muestra chica, no representativa).

### [x] E1-10 · Post-proceso de detecciones — PER
- **Objetivo:** `detection/postprocess.py`: umbral global (y opcional por clase), NMS por clase, NMS entre clases configurable y tope de detecciones por imagen. Cada filtro registra en la traza cuántas detecciones elimina.
- **Depende de:** E1-09
- **Aceptación:** tests unitarios de NMS; los parámetros salen de la config.
- **Nota (interfaz):** `postprocess(dets, cfg["detection"]) -> PostprocessResult(detections, counts)`; `counts` = sobrevivientes por paso (`raw`, `threshold`, `nms_per_class`, [`nms_cross_class`], `max_detections`) y va a `extra.counts` de la traza. NMS en numpy (sin torchvision). Umbral inicial 0,1 (se ajusta en E1-29); NMS entre clases apagado (D-09).

### [x] E1-11 · SAM2 con cajas como prompt — PER
- **Objetivo:** `segmentation/sam2.py` (batch de cajas por imagen, multimask y selección por IoU predicho) y `segmentation/postprocess.py` (recorte a la caja con margen, componente principal, relleno de huecos).
- **Depende de:** E1-10
- **Aceptación:** smoke test; máscaras de debug superpuestas; `mask_score` guardado; caché de máscaras por imagen (RLE en disco).
- **Nota (interfaz):** `Sam2Segmenter.segment(image, dets) -> list[SegmentedObject]` y `build_segmenter(cfg, device)` en `segmentation/sam2.py`; `SegmentedObject` suma `raw_area` (píxeles antes del post-proceso). El embedding de imagen se calcula una vez y las cajas van en lotes de `box_batch_size`. Caché `masks/<image_id>.json` con RLE (`save_segmented`/`load_segmented`). `postprocess_mask` con pasos apagables desde `segmentation.postprocess`. `scripts/run_segmentation.py --from-run <run|latest>` reutiliza las detecciones de `run_detection.py` y con `--eval` da F1 de cajas y de máscaras. Requiere `torchvision` (lo pide el procesador de SAM2 en `transformers`). Los checkpoints `facebook/sam2.1-*` son de video y avisan al cargarse en `Sam2Model`: es lo esperado. Smoke test en CPU con `hiera-tiny`, 2 imágenes: ~9 s/imagen, F1 de máscaras 0,59.

### [x] E1-12 · Evaluación de segmentación con oráculo — EVA
- **Objetivo:** `eval/segmentation.py`: F1 de máscaras (usando E1-07 con IoU de máscaras), mIoU de pares matcheados, y una corrida de SAM2 con **cajas GT** como prompt (oráculo, solo en eval) para separar el error de detección del de segmentación.
- **Depende de:** E1-07, E1-11
- **Aceptación:** tabla comparativa entre cajas predichas y cajas GT en `metrics_segmentation.json`.
- **Nota (interfaz):** corrido en Kaggle (D-15). `oracle_rows` / `summarize_oracle` / `predicted_pairs_summary` en `eval/segmentation.py`: SAM2 con cajas GT y post-proceso apagado (una sola pasada evalúa la máscara cruda y la post-procesada), mIoU y fracción con IoU ≥ 0,5 por tamaño, correlación `mask_score`–IoU, y con cajas predichas cuántos TP de cajas se pierden en máscaras. `scripts/eval_segmentation.py --run <run de dev>` escribe `metrics_segmentation_oracle.json` (nombre propio, para no pisar `metrics_segmentation.json`) y `segmentation_oracle.csv`. Smoke test (hiera-tiny, 3 imágenes): mIoU oráculo 0,85; el post-proceso no mejora y baja medium de 0,94 a 0,90 → candidato a experimento (E1-30).

## Escena

### [x] E1-13 · Propiedades geométricas — ESC
- **Objetivo:** `scene/scene/properties.py`: área, área relativa, centroide, posición 3×3 (nombres oficiales) y tamaño `small|medium|large`, calculados desde la **máscara decodificada del RLE**.
- **Depende de:** E1-03, E1-04
- **Aceptación:** tests con máscaras sintéticas para cada una de las 9 regiones y los bordes; máscara vacía → fallback a la caja con la marca correspondiente.
- **Nota (interfaz):** `properties_from_mask(mask, small_max, medium_max, source)` / `properties_from_rle(...)` / `properties_from_config(mask, cfg)` → `GeometricProperties` (`area`, `area_ratio`, `centroid` [x, y], `position`, `size`, `mask_bbox` xyxy exclusivo, `source`). Para máscaras vacías, `ensure_nonempty(mask, box)` devuelve la caja rasterizada (al menos 1 px) con `source="box_fallback"`: el builder (E1-16) debe guardar **esa** máscara en el RLE, así `area_ratio > 0` y las propiedades coinciden con la máscara entregada. Convenciones de `docs/schema_notes.md` §6; `size_label` usa los mismos cortes que `eval.lvis_federated.size_label`.

### [x] E1-14 · Color dominante — ESC
- **Objetivo:** `scene/scene/color.py`: k-means en Lab sobre los píxeles de la máscara, mapeo a una paleta nombrada y una confianza (proporción del cluster dominante).
- **Depende de:** E1-13
- **Aceptación:** tests con parches de color sólido y con mezcla; paleta definida en la config.
- **Nota (interfaz):** `dominant_color(image, mask, palette, k, max_pixels, min_pixels, seed)` / `dominant_color_from_config(image, mask, cfg)` → `DominantColor(name, rgb, confidence, lab_distance)` o `None` si la máscara tiene menos de `min_pixels` (color desconocido). k-means de sklearn en Lab (OpenCV, float) con submuestreo y semilla fija; paleta de 11 colores en `scene.color.palette`. ~40 ms por objeto en CPU.

### [x] E1-15 · Relaciones espaciales — ESC
- **Objetivo:** `scene/scene/relations.py`: `left_of`, `right_of`, `above`, `below`, `near`, `overlaps`, `on_top_of`, con márgenes configurables, tope de pares y valores de relación del schema.
- **Depende de:** E1-13
- **Aceptación:** tests sintéticos por relación, incluidos casos límite (centroides alineados → sin relación horizontal).
- **Nota (interfaz):** `compute_relations(objs: list[RelObject], width, height, cfg["scene"]["relations"]) -> RelationsResult(relations, extra_relations, stats)`. `RelObject(id, centroid, box, mask, confidence)` usa la caja de la máscara (`mask_bbox`). `relations` solo con el enum del schema (direccionales por centroide con margen, en los dos sentidos; `overlaps` simétrica por IoU de máscaras); `near` y `on_top_of` van en `extra_relations` (D-04). Operaciones con máscaras solo en pares cuyas cajas se tocan. El tope `max_relations` prioriza pares de mayor confianza y nunca deja una relación sin su inversa. En una imagen real con 50 objetos: 4370 relaciones en 0,07 s. **Ojo (E1-18):** con ~100 objetos son ~17 000 relaciones (~1 MB de JSON por escena); si el submission queda muy pesado, bajar `max_relations` o limitar los objetos que entran a relaciones.

### [x] E1-16 · Builder del SceneJSON — ESC
- **Objetivo:** `scene/scene/builder.py`: arma el SceneJSON (imagen, objetos con ids únicos, clase, confianza, bbox, RLE, propiedades, atributos, relaciones y origen observado/derivado/desconocido) y lo valida con E1-03.
- **Depende de:** E1-11, E1-13, E1-14, E1-15
- **Aceptación:** el JSON valida contra el schema; imagen sin detecciones → JSON válido con lista vacía.
- **Nota (interfaz):** `build_scene(image_id, image, segmented, cfg) -> (scene, stats)` en `scene/scene/builder.py`; valida con `assert_valid_scene`. Ids `obj_<n>` ordenados por confianza. `bbox` = caja del detector (observada); además `category_name`, `area_px`, `mask_bbox`, `mask_score`, `attributes` (color, confianza y RGB), `provenance` (observed/derived/unknown por campo), `source` y `extra_relations`. Si la máscara queda vacía se guarda la caja rasterizada (`source.mask = "box_fallback"`). Peso: una escena de 74 objetos ocupa ~750 KB, >80 % en `relations`.

### [x] E1-17 · Orquestador end-to-end — INF
- **Objetivo:** `scene/pipeline.py` y `scripts/run_pipeline.py`: recorre imágenes, corre las etapas con caché, traza y guarda el SceneJSON por imagen. Flags `--split`, `--limit` y `--from-stage`.
- **Depende de:** E1-16, E1-05
- **Aceptación:** corrida completa de dev en Kaggle con GPU; una segunda corrida que solo cambia el umbral reutiliza la caché de detecciones crudas.
- **Nota (interfaz):** corrida completa de dev en Kaggle (120 imágenes, 0 errores, mismos F1 que las corridas por etapa). `Pipeline(cfg, ctx, tracer, vocab, make_detector, make_segmenter, from_stage, debug).run(images)` en `scene/pipeline.py`; los modelos se cargan solo si hay algo que no está en caché. Caché compartida entre runs en `paths.cache`, con clave = hash de la config de la etapa y las anteriores (`stage_keys`): cambiar el umbral reutiliza las detecciones crudas; cambiar relaciones o color reutiliza también las máscaras (testeado con modelos falsos). Traza con `status=cached`. `scripts/run_pipeline.py` admite `--split` con varios splits, `--images`, `--limit`, `--from-stage`, `--debug` y `--eval`. `start_run` ahora agrega un sufijo si el `run_id` ya existe. Probado en CPU sobre una imagen externa al dataset.

### [x] E1-18 · Submission writer — ENT
- **Objetivo:** `scene/io/submission.py` y `scripts/make_submission.py`: 80 filas (60 de test y 20 del benchmark), columnas `image_id,SceneJSON`, mismo orden e ids que `sample_submission.csv`, cada JSON validado.
- **Depende de:** E1-17
- **Aceptación:** validación completa sin errores; **hito H1** al subirlo a Kaggle y obtener un puntaje.
- **Nota (interfaz):** submission completo generado y aceptado por Kaggle el 2026-10-24
  (80 filas válidas, 15,4 MB, con `questions` y `questions_llm`), `PipelineScore = 0.6509`;
  **hito H1 cumplido**. `build_submission(scenes, cfg) -> (rows, errors)`, `write_submission`,
  `read_submission` en `scene/io/submission.py`.
  `scripts/make_submission.py --run <run|latest> [...]` junta escenas de varios runs, valida las
  80 filas y no escribe nada si alguna falla.
  Testeado con las 80 filas de `sample_submission.csv`.

### [x] E1-19 · Chequeo de consistencia de la representación — EVA
- **Objetivo:** `eval/scene_consistency.py`: verifica que el área, el centroide, la posición y el tamaño coincidan con la máscara decodificada; que las relaciones sean coherentes (simetría left/right, ids existentes); y que la evidencia apunte a ids válidos. Además mide la exactitud de posición, tamaño y color contra GT en pares matcheados.
- **Depende de:** E1-16, E1-07
- **Aceptación:** reporte `metrics_scene.json`; 0 inconsistencias en el run base.
- **Nota (interfaz):** `internal_consistency(scenes, cfg)` (sin GT: área, centroide, posición y tamaño recalculados desde el RLE con pycocotools y código propio; relaciones con ids válidos, enum, inversa presente y dirección coherente con los centroides; evidencias; fracción de máscara fuera de la caja) y `attribute_accuracy(scenes, gt, cfg, images)` (pares TP de máscaras: posición, tamaño y color contra los de la máscara GT) en `eval/scene_consistency.py`. Para emparejar, `Prediction` y `MatchEvent` ganan `pred_id`/`gt_id` (opcionales). `scripts/run_eval.py --run <run|latest>` calcula F1 de cajas y máscaras (con desglose por tamaño), consistencia y atributos, y escribe `metrics_scene.json`. Smoke test con 3 imágenes de dev: 0 inconsistencias; posición 94 %, tamaño 100 %, color 94 %.

## Preguntas

### [x] E1-20 · DSL de consultas — QA
- **Objetivo:** `qa/dsl.py` con los modelos pydantic del DSL (`docs/PLAN.md §3`) y el resultado (`answer`, `normalized_answer`, `evidence`, `confidence`, `status`).
- **Depende de:** E1-03
- **Aceptación:** cada pregunta del benchmark tiene una representación DSL escrita a mano en un fixture de tests (solo para tests, nunca para inferencia).
- **Nota (interfaz):** `Query(op, target, target2, relation, attribute, output_type, reason)` con `Target(category: str, filters: Filters(color, position, size))`; `extra="forbid"` y validación cruzada (`relation`/`target2` solo con `op="relation"`, `attribute` solo con `op="attribute"`, `output_type` se completa desde la op y tiene que coincidir). Se agregó la op `unsupported` (con `reason`) para que el intérprete LLM (E1-26) pueda rechazar preguntas que no entran en ninguna op. `QueryResult(answer, normalized_answer, evidence, confidence, status, detail)` y `to_question_answer(question_id, type, question, query=None)` arman el item de `questions` (o de `questions_llm`, con `interpreted_query`) más los campos propios `status` y `status_detail`. Mapeos `QUESTION_TYPE_TO_OP` y `ANSWER_FORMAT_TO_OUTPUT`. Fixture: `tests/fixtures/benchmark_dsl.json` (60 consignas; el test controla op, formato, categorías y que el predicado coincida con la frase del texto).

### [x] E1-21 · Category matcher — QA
- **Objetivo:** `qa/category_matcher.py`: texto → category_id mediante normalización, plurales y sinónimos, con similitud de embeddings como último recurso y umbral; devuelve `None` si no hay match.
- **Depende de:** E1-08
- **Aceptación:** tests con plurales, sinónimos y una categoría fuera de vocabulario (→ None).
- **Nota (interfaz):** `CategoryMatcher(categories, lexicon, synonyms, embedder, embedding_threshold).match(text) -> CategoryMatch(category_id, category_name, method, score, matched_text) | None` (y `match_id`); `matcher_from_config(cfg, categories, device)` lee `qa.matcher`. Para cada forma del texto (tal cual y después en singular, en inglés y en español, palabra por palabra) busca nombre → léxico español (`configs/lexicon_es.yaml`, traducción manual de las 120 categorías) → sinónimos LVIS; un texto que apunta a dos categorías dentro de un nivel se descarta (`dropped`). Último recurso: LaBSE con umbral 0,90 (D-13), cargado recién en el primer uso. Las 50 clases en español del benchmark se resuelven por léxico. `scripts/calibrate_matcher.py` deja la calibración en `outputs/matcher/`.

### [x] E1-22 · Ejecutor determinista — QA
- **Objetivo:** `qa/executor.py`: ejecuta el DSL sobre el SceneJSON para todas las ops, con filtros por color, posición y tamaño, evidencia, confianza (regla del PLAN §4.5) y estados `unsupported`, `insufficient_information` y `ambiguous`. Normaliza respuestas: entero, yes/no, región oficial o etiqueta de tamaño.
- **Depende de:** E1-20, E1-21, E1-16
- **Aceptación:** tests por op sobre SceneJSON sintéticos, incluidos los tres estados no-ok; la evidencia siempre apunta a ids existentes.
- **Nota (interfaz):** `Executor(resolve_category, category_names, palette, absent_confidence, color_aliases, display_names).execute(query, scene) -> QueryResult`; `executor_from_config(cfg, resolve_category, category_names, display_names)` (`resolve_category` = `CategoryMatcher.match_id`). Estados: categoría fuera de vocabulario, color fuera de la paleta u op `unsupported` → `unsupported`; pregunta singular sin instancias → `insufficient_information`; varias instancias → `ambiguous` (D-10). En los estados no-ok, `normalized_answer` es el nombre del estado y la confianza es 0. `near`/`on_top_of` se buscan en `extra_relations`. La confianza de "no hay" es 1 − máx. de las instancias descartadas por filtros, o 1 − `detection.score_threshold` si no hay ninguna. El texto de `answer` sale de `qa/verbalize.py` (E1-27) con el nombre en español; `describe` usa una plantilla determinista (conteos y objeto más confiable).

### [x] E1-23 · Modo determinista (adapter) e integración en `questions` — QA
- **Objetivo:** `qa/structured_adapter.py` traduce los campos estructurados del benchmark al DSL sin LLM. Integra `questions` en los SceneJSON de las 20 imágenes del benchmark, conservando `question_id`, `type` y `question`.
- **Depende de:** E1-22, E1-18
- **Aceptación:** las 60 consignas del benchmark (20 de descripción + 40 estructuradas, ver `docs/DECISIONS.md`) producen una respuesta o un estado explícito; el submission valida; **hito H2** si además F1 ≥ 0,40.
- **Nota:** corrido en Kaggle sobre las 20 imágenes del benchmark: las 60 consignas responden (51 `ok`, 5 `ambiguous`, 4 `insufficient_information`, 0 `unsupported`) y el submission valida (H2: F1 de cajas 0,653 ≥ 0,40). `query_from_benchmark(q, phrases)` usa `type` y `class_en`; el predicado se busca en el texto con `qa.adapter.relation_phrases` (gana la frase más larga). El test compara las 60 traducciones contra el fixture escrito a mano de E1-20. `answer_questions(...)` arma los items (con `interpreted_query` y `status`). `scripts/run_qa.py --run <run|latest>` agrega `questions` a `scenes/<image_id>.json` del run, valida cada escena, traza la etapa `qa_deterministic` y guarda `qa_deterministic.csv`; después se corre `make_submission.py`. Probado con un run armado a partir de las 80 filas de `sample_submission.csv`: las 60 consignas responden y el submission valida.

### [x] E1-24 · Evaluación de QA — EVA
- **Objetivo:** `eval/qa_eval.py`:
  - **Fidelidad:** recomputa cada respuesta desde el SceneJSON predicho y verifica que la evidencia sostenga la respuesta.
  - **Exactitud contra GT:** construye un SceneJSON desde GT (`eval/oracle.py`, solo en eval) y corre el mismo ejecutor.
  - Exactitud por tipo y por modo, y tasa de unsupported.
- **Depende de:** E1-23, E1-07
- **Aceptación:** tabla con exactitud sobre el JSON predicho y sobre el JSON oráculo por tipo de pregunta, que permite atribuir cada error a percepción o a razonamiento.
- **Nota (interfaz):** `eval/oracle.py::oracle_scene(img_gt, image, cfg, names)` arma el SceneJSON desde el GT con el mismo builder (confianza 1). `eval/qa_eval.py`: `evidence_supports` (la evidencia sostiene la respuesta: conteo = largo de la evidencia, existencia ⇔ evidencia no vacía, posición/tamaño del objeto de evidencia, tripla de relación presente o ausente), `reexecutes` (volver a ejecutar el `interpreted_query` da lo mismo), `evaluate_mode(mode, ...)` (referencia = mismo ejecutor con la consulta estructurada sobre el oráculo; atribución `perception` / `interpretation` (DSL del LLM ≠ estructurado) / `oracle_not_ok`) y `summarize`. `scripts/eval_qa.py --run <run>` escribe `metrics_qa.json` y `qa_eval.csv`. **Ojo:** la referencia usa nuestras reglas de posición, tamaño y relaciones, no las respuestas de la cátedra (el benchmark no las trae). Run local sobre las 20 imágenes del benchmark (OWLv2 + SAM2 tiny), modo determinista: **72,5 %** de coincidencia con el oráculo en las 40 cerradas (count 40 %, existence 90 %, position 100 %, relation 80 %, size 60 %), evidencia y re-ejecución 100 %, los 11 errores son de percepción.

## LLM

### [x] E1-25 · Carga del LLM y benchmark de recursos — LLM
- **Objetivo:** `llm/model.py`: carga configurable (Qwen2.5-3B-Instruct o Qwen3-4B; fp16 o 4-bit), generación con semilla fija y temperatura 0. Medir memoria y latencia en T4.
- **Depende de:** E1-01
- **Aceptación:** tabla de memoria y latencia por candidato en `docs/DECISIONS.md` con la elección justificada.
- **En desarrollo (2026-10-24):** wrapper lazy para fp16/4-bit y `scripts/benchmark_llm.py`
  implementados. **Medido en T4 (D-12):** Qwen2.5-3B fp16 6,0 GB y 5,3 s/consulta (elegido);
  Qwen3-4B 4-bit 2,6 GB y 10,9 s.

### [x] E1-26 · Intérprete pregunta → DSL — LLM
- **Objetivo:** `llm/interpreter.py` con prompt few-shot versionado. Recibe **solo** el texto de la pregunta, produce JSON, valida contra el DSL, reintenta una vez ante error y, si vuelve a fallar, devuelve `unsupported`. El mapeo de la categoría lo hace E1-21, no el LLM.
- **Depende de:** E1-25, E1-20, E1-21
- **Aceptación:** validez del JSON ≥ 95 % y coincidencia del DSL con el del modo determinista medida sobre el benchmark; casos no soportados y ambiguos incluidos en los few-shot.
- **En desarrollo (2026-10-24):** intérprete, prompt versionado, validación, resolución de
  categorías por inyección y fallback implementados con tests. **Prompt v2 (D-18):** sin
  categorías ni preguntas del benchmark y con ejemplos de todas las ops (incluido `describe`).
  **Medido en T4:** JSON válido 1,00 y DSL igual al estructurado 1,00 en las 60 consignas
  (con la v1: 0,65 y 0,65).

### [x] E1-27 · Verbalización y descripción — LLM
- **Objetivo:** `qa/verbalize.py` (respuesta en lenguaje natural a partir del resultado del ejecutor) y `llm/describer.py` (descripción a partir de un resumen textual compacto del SceneJSON). Verificador de fidelidad: cada categoría y número mencionado debe existir en el JSON.
- **Depende de:** E1-25, E1-16
- **Aceptación:** las 20 descripciones generadas, con la métrica de fidelidad calculada; se registran los casos con afirmaciones no sustentadas.
- **En desarrollo (2026-10-24):** verbalización, resumen compacto, descriptor y verificador de
  categorías/números implementados con tests. La fidelidad de las 20 descripciones la mide
  `scripts/eval_qa.py` (`llm.descriptions`: proporción fiel, respaldos con la plantilla y consignas
  que el intérprete no tomó como `describe`). **Medido en T4:** 18/20 descripciones fieles; las 2
  restantes usan la plantilla determinista.

### [x] E1-28 · Integración de `questions_llm` — LLM
- **Objetivo:** genera `questions_llm` (con `interpreted_query`) y las descripciones para las 20 imágenes del benchmark mediante intérprete → mismo ejecutor → verbalización, e integra el resultado en el submission.
- **Depende de:** E1-26, E1-27, E1-23
- **Aceptación:** el submission valida con `questions_llm` completo; E1-24 se extiende al modo LLM.
- **Nota (interfaz):** corrido en la T4 sobre las 20 imágenes del benchmark: modo LLM con exactitud contra el oráculo 0,725 (igual al determinista), evidencia y re-ejecución 100 %, submission válido con `questions_llm`. `scene/qa/llm_mode.py::answer_questions_llm(questions, scene, interpreter, executor, describer, names)`: el intérprete (E1-26) recibe solo `question`; su DSL lo ejecuta el mismo `Executor` de E1-22; si el intérprete falla o marca la pregunta como no soportada, se ejecuta `Query(op="unsupported", reason=...)`, así `interpreted_query` siempre es un DSL válido. `op="describe"` → `SceneDescriber` (E1-27); si la descripción no es fiel tras el reintento se usa la plantilla determinista (`status_detail` lo indica). Cada item lleva `interpreter` (estado, intentos, errores, salida cruda y estado del descriptor). `scripts/run_qa_llm.py --run <run>` agrega `questions_llm` a las escenas del benchmark (etapa `qa_llm` en la traza, `qa_llm.csv`); `eval_qa.py` evalúa los dos modos (validez del JSON, coincidencia del DSL con el estructurado y exactitud).

## Experimentación, análisis y entrega

### [x] E1-29 · Ajuste de umbral con tuning/holdout — EVA
- **Objetivo:** `scripts/tune_thresholds.py`: split fijo de 80/40 (estratificado, con semilla), barrido de umbral (y NMS entre clases) sobre la caché de detecciones crudas, selección en tuning y reporte en holdout.
- **Depende de:** E1-07, E1-17
- **Aceptación:** curva F1 vs. umbral guardada y valor elegido escrito en `base.yaml`, con su justificación.
- **Nota (interfaz):** corrido en Kaggle: umbral **0,18** en `base.yaml` (D-14). `split_tuning_holdout` (estratificado por cantidad de instancias: se ordena y de cada grupo consecutivo va una imagen al holdout, con la semilla global), `sweep` y `choose` (máximo F1 en tuning; ante empate, el umbral más alto) en `eval/tuning.py`. `scripts/tune_thresholds.py` barre 0,02–0,50 con y sin NMS entre clases sobre la caché cruda (no corre modelos) y escribe `curve.csv`, `curve.png`, `split.json` y `summary.json` en `outputs/tuning/<hash>/`, con el F1 de holdout del umbral elegido y del actual.

### [x] E1-30 · Experimento controlado de mejora — EVA
- **Objetivo:** al menos una comparación contra la línea base cambiando una sola variable (candidatos: sinónimos en el vocabulario, detector alternativo, NMS entre clases, umbral por clase). Registro en `outputs/experiments.csv` con hipótesis, cambio, métricas de tuning y holdout, costo y conclusión. Si hace falta, implementar el detector alternativo (`detection/yolo_world.py` o `detection/gdino.py` con chunks).
- **Depende de:** E1-29
- **Aceptación:** mejora sobre holdout justificada (o, si no mejora, documentada); **F1 ≥ 0,50 en dev → hito H3** junto con E1-28 y E1-31.
- **Nota:** dos comparaciones controladas contra la línea base, una variable cada una, decididas con el holdout de E1-29: (1) **umbral 0,10 → 0,18**, F1 de cajas en holdout 0,619 → 0,666, **adoptada** (D-14); (2) **post-proceso solo en máscaras chicas**, mejora con cajas GT (85,2 % vs 84,1 % con IoU ≥ 0,5) pero no end-to-end (F1 de máscaras en holdout 0,614 → 0,609), **descartada** (D-17). Herramientas: `configs/experiments/*.yaml`, `scripts/compare_runs.py` (tuning/holdout, por tamaño, consistencia y costo; fila en `outputs/experiments.csv`) y la variante `post_small` del oráculo. F1 de detección en dev 0,653 ≥ 0,50 (parte de H3).

### [x] E1-31 · Análisis de errores y propagación — EVA
- **Objetivo:** `eval/error_analysis.py`:
  - Embudo de instancias a partir de las trazas (raw → umbral → NMS → SAM2 → JSON) contra el GT.
  - Errores por tamaño y por clase.
  - Galería de casos correctos, ambiguos y fallidos para detección, segmentación y QA.
  - Atribución de errores de QA a percepción o a razonamiento usando el oráculo.
- **Depende de:** E1-12, E1-24, E1-28
- **Aceptación:** sección de notebook y `docs/error_analysis.md` con conclusiones que motivan la extensión de la Etapa 2.
- **Nota:** percepción corrida en Kaggle sobre dev y conclusiones en `docs/error_analysis.md` (55,5 % de las instancias llega a TP de máscara; se pierden sobre todo en detección: 11,6 % sin caja, 8,4 % con otra clase, 16,6 % bajo el umbral; small 40 % vs ~72 % medium/large → motiva tiling tipo SAHI para la Etapa 2). La atribución de QA la da E1-24: en los dos modos los 11 errores sobre 40 cerradas son de percepción (6 de conteo, 1 existencia alucinada, 4 objetos no detectados). Falta solo la sección de notebook (E1-32). `eval/error_analysis.py`: `funnel_rows`/`funnel_summary` (primera etapa en la que se pierde cada instancia GT: `no_box` → `wrong_class` → `below_threshold` → `nms` → `max_detections` → `box_match` → `mask` → `ok`; las etapas de detección se recalculan desde la caché cruda con el mismo post-proceso y piden cobertura IoU ≥ 0,5, las dos últimas usan el matching federado), `fp_types` (duplicado, localización, confusión de clase, categoría negativa, fondo), `per_image_scores`, `pick_gallery` (correctas, ambiguas y fallidas, sin superposición), `draw_case` (GT verde, FN amarillo, TP azul, FP rojo, ignoradas gris) y `worst_classes`. `scripts/analyze_errors.py --run <run>` usa la config de inferencia del run y escribe `error_analysis/` (summary.json, funnel.csv, fp_types.csv, per_image.csv, worst_classes.csv, gallery/).

### [x] E1-32 · Generador de notebook y corrida en Kaggle — ENT
- **Objetivo:** `scripts/build_notebook.py` genera `TPI_IA_2026P_<grupo>.ipynb`: instalación, `%%writefile` de los módulos y secciones en Markdown (EDA, metodología, resultados, experimentos, análisis, submission). Separa claramente GT, predicciones y datos derivados.
- **Depende de:** E1-31
- **Aceptación:** corre de punta a punta en Kaggle con GPU y sin archivos externos; queda guardada con todas sus salidas; genera `submission.csv` y los CSV y JSON de métricas.
- **Cierre (2026-10-24):** `TPI_IA_2026P_15.ipynb` se ejecutó completa en Kaggle sobre una
  Tesla T4. La corrida de desarrollo `20261024-223716-6f583413` procesó 120 imágenes y la de
  entrega `20261024-225121-6f583413` generó las 80 filas. La última celda validó la entrega y
  produjo `grupo15_entrega_artifacts.zip` con métricas, trazas, manifiestos, análisis y submission.

### [x] E1-33 · Reporte técnico — ENT
- **Objetivo:** `report/REPORTE.md` con la estructura sugerida (portada, dataset, metodología, resultados, discusión y errores, conclusiones). Las tablas se generan desde los CSV del run final para garantizar correspondencia con la notebook. Declara modelos, recursos y código reutilizado.
- **Depende de:** E1-32
- **Aceptación:** cada número del reporte es trazable a un archivo del run final.
- **Nota:** `scripts/build_report.py` valida el submission y genera `report/REPORTE.md` y
  `report/report_data.json` desde `grupo15_entrega_artifacts.zip`. El snapshot conserva las cifras
  usadas y el SHA-256 de cada JSON, CSV, manifiesto y traza de origen.

### [ ] E1-34 · Verificación final de validez — ENT
- **Objetivo:** checklist: el test de frontera pasa; la pipeline procesa una imagen externa al dataset sin errores; las semillas y el manifest están completos; el reporte coincide con la notebook; el submission es válido; el modelo LLM es ejecutable en Kaggle; el checklist de promoción del enunciado está cubierto punto por punto.
- **Depende de:** E1-33
- **Aceptación:** `docs/checklist_entrega.md` completado.

---

# ETAPA 2 — Sobresaliente

> Arranca solo con el **hito H3** cumplido.

### [ ] E2-01 · Selección de la extensión e hipótesis — EVA
- **Objetivo:** a partir de `docs/error_analysis.md`, elegir **una** extensión (PLAN §4.9) y escribir la hipótesis cuantificable (por ejemplo: "el re-scoring con CLIP reduce los FP en categorías negativas sin bajar el recall más de 2 pts → F1 +X"), la métrica, el diseño experimental y el costo esperado.
- **Depende de:** E1-31
- **Aceptación:** `docs/extension.md` aprobado por el grupo antes de implementar.

### [ ] E2-02a · Extensión A: re-scoring con CLIP sobre el recorte enmascarado — PER
- **Objetivo:** módulo `detection/rescoring.py` que calcula la similitud CLIP del recorte enmascarado contra las queries del vocabulario y la combina con el score del detector (peso configurable, ajustado en tuning).
- **Depende de:** E2-01
- **Aceptación:** activable por config; trazado; tests básicos.

### [ ] E2-02b · Extensión B: tiling tipo SAHI — PER
- **Objetivo:** inferencia por tiles con solapamiento más la imagen completa, fusión con NMS o WBF, activable por config.
- **Depende de:** E2-01
- **Aceptación:** mismo formato de salida que el detector base; trazado; costo de tiempo registrado.

### [ ] E2-02c · Extensión C: ensamble con WBF — PER
- **Objetivo:** fusión de dos detectores con Weighted Boxes Fusion.
- **Depende de:** E2-01, E1-30 (detector alternativo implementado)
- **Aceptación:** activable por config; trazado.

> Se implementa **solo** la extensión elegida en E2-01.

### [ ] E2-03 · Evaluación cuantitativa del aporte — EVA
- **Objetivo:** comparar la línea base de promoción contra la extensión sobre holdout: F1 de cajas y máscaras, desglose por tamaño y clase, efecto en la exactitud de QA (propagación), costo en tiempo y memoria. Opcional: intervalos por bootstrap sobre imágenes.
- **Depende de:** E2-02x
- **Aceptación:** tabla y gráficos en el notebook; fila en `experiments.csv`.

### [ ] E2-04 · Análisis crítico e integración en la entrega — ENT
- **Objetivo:** documentar qué mejora, qué limitaciones conserva y qué costo introduce; decidir con datos si la extensión entra al submission final; actualizar la notebook, el reporte y el checklist (**hito H4**).
- **Depende de:** E2-03
- **Aceptación:** sección "Extensión" del reporte consistente con el notebook; E1-32 a E1-34 se vuelven a correr con la versión final.

---

## Grafo de dependencias (resumen)

```
E1-00 → E1-01 → E1-05 ─────────────┐
      → E1-02 → E1-06 (EDA)        │
             → E1-07 (eval) ───────┼──────────────┐
             → E1-08 → E1-09 → E1-10 → E1-11 → E1-12
      → E1-03 → E1-13 → E1-14/15 → E1-16 → E1-17 → E1-18 (H1)
      → E1-04 ┘                       └→ E1-19
E1-03 → E1-20 ┐
E1-08 → E1-21 ┴→ E1-22 → E1-23 (H2) → E1-24
E1-01 → E1-25 → E1-26/27 → E1-28
E1-17 → E1-29 → E1-30 ┐
E1-12/24/28 → E1-31 ──┴→ E1-32 → E1-33 → E1-34 (H3)
E1-31 → E2-01 → E2-02x → E2-03 → E2-04 (H4)
```

Desde el arranque hay cuatro frentes que se pueden trabajar en paralelo: **INF/EVA** (E1-01..07), **PER** (E1-08..11), **ESC** (E1-13..15, con máscaras sintéticas) y **QA/LLM** (E1-20..22 y E1-25, con SceneJSON sintéticos).
