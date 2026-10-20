# Plan técnico — TPI IA 2026: Comprensión visual de escenas

## 1. Objetivo y etapas

- **Etapa 1: Promoción.** Pipeline completa (detección, SAM2, SceneJSON, QA determinista y QA con LLM), tracing completo, evaluación por etapa con análisis de propagación de errores, y una mejora justificada con experimento controlado contra una línea base. Meta cuantitativa: **F1 de detección ≥ 0,50** (IoU ≥ 0,5, política LVIS federada) sobre desarrollo.
- **Etapa 2: Sobresaliente.** Una extensión no trivial, motivada por el análisis de errores de la Etapa 1, con hipótesis explícita, evaluación cuantitativa de su aporte y análisis de costo y limitaciones.

La Etapa 2 solo arranca cuando el **hito H3** (promoción completa y submission válido) está cumplido.

### Hitos

| Hito | Condición |
|---|---|
| H1 | Primer `submission.csv` válido de 80 filas (aunque las métricas sean bajas) |
| H2 | Requisitos de **aprobado** cubiertos: F1 ≥ 0,40, `questions` con existencia y conteo, traza básica, evaluación y comparación básicas |
| H3 | Requisitos de **promoción** cubiertos: F1 ≥ 0,50, `questions_llm`, tracing completo, manejo de ambiguas y no soportadas, mejora controlada |
| H4 | Extensión de sobresaliente evaluada y documentada |

## 2. Arquitectura y repo

```
tpi-ia-2026/
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── base.yaml                 # config de la línea base
│   └── experiments/*.yaml        # cada experimento = override de base
├── data/                         # gitignore salvo competition/
│   ├── competition/              # scene_schema.json, sample_submission.csv, benchmark
│   └── dataset/                  # copia local del dataset de Kaggle
├── src/
│   ├── scene/                    # INFERENCIA (no ve GT)
│   │   ├── config.py             # carga YAML, seeds, run_id, hash de config
│   │   ├── io/                   # carga de imágenes y vocabulario, escritura de submission
│   │   ├── detection/            # interfaz Detector, owlv2, yolo_world, gdino, postprocess, vocabulary
│   │   ├── segmentation/         # sam2, postprocess de máscaras
│   │   ├── masks/rle.py          # RLE no comprimido column-major
│   │   ├── scene/                # properties, color, relations, builder, schema (validación)
│   │   ├── qa/                   # dsl, executor, structured_adapter, category_matcher, verbalize
│   │   ├── llm/                  # model, interpreter, describer, prompts/
│   │   ├── tracing/tracer.py
│   │   └── pipeline.py           # orquestador con caché por etapa
│   ├── eval/                     # ÚNICO lugar que lee GT
│   │   ├── annotations.py, lvis_federated.py, segmentation.py
│   │   ├── scene_consistency.py, qa_eval.py, oracle.py, error_analysis.py
│   └── eda/
├── scripts/                      # run_pipeline, run_eval, make_submission, build_notebook, tune_thresholds
├── notebooks/                    # notebook final GENERADA (no se edita a mano)
├── report/REPORTE.md
└── tests/
```

**Notebook final.** Se genera con `scripts/build_notebook.py`, que escribe los módulos de `src/` mediante celdas `%%writefile` y luego agrega las secciones de ejecución y análisis en Markdown. De esa forma corre en Kaggle sin archivos externos más allá del dataset oficial y los modelos públicos que la propia notebook descarga. Nombre: `TPI_IA_2026P_[nro_grupo]`.

**Caché por etapa.** Detecciones, máscaras y SceneJSON se guardan por imagen en `outputs/runs/<run_id>/`. Así un experimento que solo cambia el umbral o las relaciones no necesita recorrer de nuevo los modelos.

## 3. Contratos internos

```python
@dataclass
class Detection:
    box_xyxy: tuple[float, float, float, float]  # píxeles, imagen original
    score: float
    category_id: int
    category_name: str
    source: str  # "owlv2", "yolo_world", ...
    raw_query: str  # texto/sinónimo que la disparó


@dataclass
class SegmentedObject:
    detection: Detection
    mask: np.ndarray  # bool HxW, solo en memoria
    mask_score: float  # IoU predicho por SAM2
```

El **SceneJSON** sigue `scene_schema.json`, que es la referencia normativa (ver `docs/schema_notes.md`). Se puede extender con campos propios, siempre que el JSON siga validando.

**DSL de consultas** (pydantic). Lo usan los dos modos:

```json
{
  "op": "count | exists | position | size | relation | attribute | describe",
  "target":  {"category": "cup", "filters": {"color": null, "position": null, "size": null}},
  "target2": {"category": "table", "filters": {}},
  "relation": "left_of | right_of | above | below | near | overlaps | on_top_of | null",
  "attribute": "color | null",
  "output_type": "int | yes_no | region | size_label | text"
}
```

El ejecutor devuelve `answer`, `normalized_answer`, `evidence` (ids de objetos del mismo SceneJSON), `confidence` y un `status` que puede ser `ok`, `unsupported`, `insufficient_information` o `ambiguous`.

## 4. Decisiones técnicas

### 4.1 Detección open-vocabulary
| Aspecto | Decisión | Justificación / experimento |
|---|---|---|
| Detector base | **OWLv2** (`google/owlv2-base-patch16-ensemble`, a validar) | Admite las 120 queries en una sola pasada. Grounding DINO tiene límite de tokens de texto y requeriría ~5 pasadas en chunks |
| Alternativas | YOLO-World v2 (rápido) y Grounding DINO en chunks | Candidatas para la comparación controlada (E1-30) |
| Vocabulario | 120 nombres LVIS limpios (`_`→espacio, sin desambiguación entre paréntesis) | Experimento: sumar sinónimos LVIS del metadata de categorías y quedarse con el máximo por clase |
| Umbral | Global, ajustado para maximizar F1 en un split de tuning | Opcional: por clase con regularización. Se reporta sobre holdout |
| Duplicados | NMS por clase (IoU 0,5) + NMS entre clases (IoU ≈ 0,8) | Resuelve confusiones de sinónimos (cup/mug) |
| Resolución | Nativa del modelo (960 en OWLv2) | Tiling como posible extensión (Etapa 2) |
| Máx. detecciones | Tope por imagen configurable | Se ajusta según la distribución de instancias del EDA |

**Ajustar el umbral con GT es legítimo** porque es selección de hiperparámetros, no asistencia a la inferencia. Para reportar sin sobreajuste: split fijo **80 imágenes de tuning / 40 de holdout** (estratificado por cantidad de instancias, con semilla fija), o CV de 5 folds.

### 4.2 Segmentación
- SAM 2.1 hiera-large (vía `transformers` o repo oficial, a validar en Kaggle), con las cajas en batch por imagen como prompt.
- `multimask_output=True` y selección de la máscara por IoU predicho.
- Post-proceso: recorte a la caja (con margen), componente conexa principal y relleno de huecos chicos.
- **RLE no comprimido, column-major:** `mask.flatten(order="F")`, corridas alternadas empezando por ceros, con `size=[H, W]`. Se testea con round-trip contra `pycocotools`.

### 4.3 Propiedades derivadas (siempre desde la máscara codificada)
- Área en píxeles, área relativa y centroide (de la máscara; si está vacía, se cae al de la caja y se marca).
- **Posición 3×3** por centroide: `top_left … bottom_right`, con los nombres oficiales.
- **Tamaño** `small | medium | large`: umbrales del schema si los define; si no, umbrales de área relativa justificados con el EDA.
- **Color dominante:** k-means (k=3) en espacio Lab sobre los píxeles de la máscara; el cluster mayor se mapea a una paleta de ~11 colores con nombre.
- **Relaciones:** `left_of`, `right_of`, `above`, `below` (centroides con margen mínimo), `overlaps` (IoU de máscaras > ε), `near` (distancia entre bordes normalizada por la diagonal de la imagen) y `on_top_of` (contacto de máscaras más orden vertical). Para evitar una explosión O(n²), las relaciones se limitan a pares relevantes con un tope configurable.
- Cada campo registra su origen: observado, derivado o desconocido (`null`).

### 4.4 Trazabilidad
Se escribe una línea JSONL por (imagen, etapa) en `outputs/runs/<run_id>/trace.jsonl`:
`run_id, config_hash, image_id, stage, model_id, params, t_start, t_end, duration_ms, status, n_in, n_out, artifact_path, error`.
Además, por run se guarda `run_manifest.json` con la config completa, las versiones de librerías y modelos, la semilla y el hardware.
Las trazas se usan para medir dónde se pierden instancias (raw → umbral → NMS → SAM2 → JSON) y el tiempo por etapa.

### 4.5 QA
- **Modo determinista:** `structured_adapter` traduce los campos estructurados de la cátedra al DSL con código, sin LLM.
- **Modo LLM:** el LLM recibe **solo el texto** de la pregunta y produce el DSL. El resultado se valida con pydantic, se reintenta una vez ante error y, si vuelve a fallar, la respuesta es `unsupported`.
- **Mismo ejecutor** para los dos modos. Conteos, filtros y comparaciones se resuelven en código.
- **Category matcher en código:** normalización (minúsculas, singular), sinónimos LVIS y, como último recurso, similitud de embeddings con umbral. Si no supera el umbral, la respuesta es `unsupported`.
- **Manejo explícito de casos:** categoría fuera de vocabulario, atributo no implementado, referencia singular con varias instancias (`ambiguous`, con la evidencia de todas) y objeto no detectado en una pregunta de relación (`insufficient_information`).
- **Confianza:** para *exists*, la máxima confianza de las instancias o 1 − máx. si no hay ninguna. Para *count*, el promedio de las instancias contadas. Para relaciones, el producto de las confianzas de los dos objetos. La regla queda documentada.

### 4.6 LLM liviano
- Candidatos: **Qwen2.5-3B-Instruct** o **Qwen3-4B**, en fp16 o 4-bit sobre T4. Se elige por validez del JSON en el benchmark y por latencia.
- Prompts versionados en `src/scene/llm/prompts/`, con few-shot que cubre todos los tipos de pregunta y casos no soportados.
- **Descripción:** el LLM recibe un resumen textual compacto del SceneJSON (objetos, conteos, posiciones, colores, relaciones principales) con la instrucción de no agregar nada. Después se verifica automáticamente: cada sustantivo y número mencionado debe existir en el JSON (métrica de fidelidad).

### 4.7 Evaluación
| Etapa | Métricas |
|---|---|
| Detección | P, R y F1 federados LVIS (IoU ≥ 0,5) + **cobertura evaluable**, global, por tamaño y por frecuencia de clase. AP con `lvis-api` solo como referencia |
| Segmentación | F1 de máscaras, mIoU de pares matcheados, y el mismo cálculo con **cajas GT (oráculo)** para aislar el error propio de SAM2 |
| Representación | Validez contra el schema, consistencia de los campos derivados con la máscara decodificada, y exactitud de posición, tamaño y color contra GT en los pares matcheados |
| QA | Exactitud por tipo y por modo, tasa de unsupported, validez del JSON intermedio, fidelidad de las respuestas al SceneJSON predicho, y exactitud sobre un **SceneJSON construido desde GT (oráculo)** para separar el error de percepción del de razonamiento |
| Descripción | Fidelidad (tasa de afirmaciones verificables y correctas) y cobertura de los objetos principales |
| Costo | Tiempo por etapa y por imagen, y memoria GPU pico |

Los oráculos son **solo para evaluación** y nunca se usan en el submission.

### 4.8 Experimentación
- La **línea base** es `configs/base.yaml`: OWLv2, vocabulario limpio, umbral global y NMS por clase.
- Cada experimento es un override YAML que cambia **una sola variable**. Se registra en `outputs/experiments.csv` (run_id, hipótesis, cambio, métricas en tuning y holdout, costo, conclusión).
- Candidatos para la mejora controlada de promoción: sinónimos en el vocabulario, detector alternativo, NMS entre clases y umbral por clase.
- También se documentan los intentos que no mejoraron.

### 4.9 Extensión (Etapa 2)
Se elige con los resultados del análisis de errores de la Etapa 1:
- **Si dominan los falsos positivos o las confusiones entre clases:** re-scoring de cada detección con CLIP sobre el recorte enmascarado, combinado con el score del detector.
- **Si domina el recall bajo en objetos chicos:** tiling tipo SAHI con fusión de detecciones.
- **Si los detectores tienen errores complementarios:** ensamble con Weighted Boxes Fusion.

## 5. Riesgos principales
| Riesgo | Mitigación |
|---|---|
| Submission inválido por RLE o schema | Validación en cada fila antes de escribir y tests de round-trip (E1-04, E1-17) |
| Uso accidental de GT en inferencia | Test de frontera de imports y prueba con una imagen arbitraria (E1-02, E1-34) |
| Cuota o tiempo de GPU en Kaggle | Caché por etapa, smoke tests locales en CPU y corridas completas planificadas |
| LLM que genera JSON inválido | Validación, reintento, fallback a `unsupported` y métrica de validez |
| Sobreajuste del umbral a las 120 imágenes | Split tuning/holdout fijo y reporte sobre holdout |
| Reporte desalineado con la notebook | Las tablas del reporte se generan desde los CSV del run final (E1-33) |
