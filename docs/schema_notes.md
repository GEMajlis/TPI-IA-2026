# Notas sobre los archivos normativos (E1-03)

Fuente: `data/competition/scene_schema.json` (JSON Schema 2020-12), `sample_submission.csv`, `questions_dev.jsonl` y los README de la competencia. **Ante cualquier duda, gana el schema.**
Validación en código: `scene.scene.schema.validate_scene(scene, schema_path)` (jsonschema + chequeos estructurales).

## 1. SceneJSON: campos obligatorios

`additionalProperties: true` en todos los niveles: se pueden agregar campos propios (por ejemplo, el origen observado/derivado de cada valor) siempre que los mínimos estén presentes.

| Ruta | Tipo / restricción |
|---|---|
| `image.id` | string no vacío (= `image_id` de la fila: `test_0001`, `tpi_0001`, ...) |
| `image.width`, `image.height` | enteros ≥ 1 |
| `objects` | array, **máx. 300** |
| `objects[].id` | string no vacío |
| `objects[].category_id` | entero ≥ 1 (ids de `categories.csv`, 1..120) |
| `objects[].confidence` | número en [0, 1] |
| `objects[].bbox` | 4 números, **xyxy** en píxeles (según el README de la competencia) |
| `objects[].mask` | `{"size": [H, W], "counts": [enteros ≥ 0, mín. 1]}`: RLE **no comprimido, column-major** |
| `objects[].centroid` | 2 números |
| `objects[].area_ratio` | número en (0, 1] (**estrictamente > 0**: una máscara vacía no valida) |
| `objects[].position` | enum de 9 celdas (ver §2) |
| `objects[].size` | `small` \| `medium` \| `large` |
| `relations` | array, **máx. 20000**; obligatorio aunque esté vacío |
| `relations[]` | `{subject, predicate, object}`; subject/object son strings no vacíos |
| `questions` | opcional, máx. 60, items `questionAnswer` |
| `questions_llm` | opcional, máx. 60, items `questionAnswer` + `interpreted_query` |

## 2. Enums

- **position:** `top_left, top_center, top_right, middle_left, center, middle_right, bottom_left, bottom_center, bottom_right`.
- **size:** `small, medium, large`.
- **predicate:** `left_of, right_of, above, below, overlaps`. **No incluye `near` ni `on_top_of`** (ver D-04).
- **question type:** `description, count, existence, position, size, relation`.

Las constantes están en `scene/scene/schema.py` y un test verifica que coincidan con el schema.

## 3. Respuestas (`questions` / `questions_llm`)

Campos obligatorios de cada item: `question_id`, `type`, `question` (los tres se copian tal cual del benchmark), `answer` (string no vacío, en lenguaje natural), `normalized_answer` (**entero ≥ 0 o string no vacío**), `evidence` (array de strings **únicos**) y `confidence` ([0, 1]).
`questions_llm` exige además `interpreted_query`, un objeto libre: ahí va nuestro DSL (PLAN §3).

Formato de `normalized_answer` según `answer_format` del benchmark:

| type | answer_format | normalized_answer |
|---|---|---|
| `count` | `integer` | entero ≥ 0 |
| `existence` | `boolean` | `"yes"` / `"no"` |
| `relation` | `boolean` | `"yes"` / `"no"` |
| `position` | `position_3x3` | una de las 9 celdas |
| `size` | `size_class` | `small` / `medium` / `large` |
| `description` | `free_text` | string no vacío (la misma descripción u otra versión normalizada) |

En las filas de **test**, `questions` y `questions_llm` se pueden omitir o dejar vacías. En las 20 filas del benchmark son obligatorias en la práctica (la métrica de QA las usa).

## 4. Benchmark de preguntas (`questions_dev.jsonl`)

60 consignas sobre 20 imágenes de desarrollo (3 por imagen: una descripción y dos preguntas estructuradas).
Campos: `schema_version`, `question_id` (`DEV-<n>-DESC`, `DEV-<n>-Q1`, `DEV-<n>-Q2`), `image_id` (entero), `image_file`, `type`, `question` (en **español**), `answer_format`, `allowed_answers` (solo en las cerradas), `class_en` (nombres del vocabulario) y `class_es` (traducción usada en el texto).

| type | n | Ejemplo |
|---|---|---|
| description | 20 | "Describí brevemente la escena usando sólo información sustentada por el SceneJSON." |
| count | 10 | `¿Cuántas instancias de la categoría "pomo de puerta" aparecen en la imagen?` → `class_en: ["doorknob"]` |
| existence | 10 | `¿Hay al menos una instancia de la categoría "tomacorriente" en la imagen?` |
| relation | 10 | `¿La instancia de "reloj" está por encima de la instancia de "anillo"?` → `class_en: ["clock", "ring"]` |
| position | 5 | `¿En qué zona de la imagen está la instancia de "teléfono celular"?` |
| size | 5 | `¿Qué tamaño relativo tiene la instancia de "collar"?` |

Observaciones para QA:
- **El predicado de las relaciones no viene estructurado:** hay que extraerlo del texto con código ("por encima de" → `above`, "por debajo de" → `below`, "a la derecha de" → `right_of`, "a la izquierda de" → `left_of`). El orden de `class_en` es [sujeto, objeto].
- No hay preguntas de color/atributo ni de `overlaps`.
- Las preguntas de posición, tamaño y relación se refieren a **"la instancia"** (singular).

## 5. Submission

`submission.csv` con columnas exactas `image_id,SceneJSON`, 80 filas: `test_0001..test_0060` y las 20 imágenes del benchmark (`tpi_XXXX`), en el orden de `sample_submission.csv`. El `SceneJSON` va serializado como string JSON en la celda.
La fila de ejemplo usa un píxel en (0, 0): `bbox [0,0,1,1]`, `counts [0,1,H*W-1]`, `centroid [0,0]`, `area_ratio 1/(H*W)`.

**Métrica (`PipelineScore`):** 40 % F1 de cajas + 35 % F1 de máscaras + 15 % consistencia estructural + 10 % QA (promedio de `questions` y `questions_llm`, solo en las imágenes del benchmark).

## 6. Lo que el schema NO define (decisiones del grupo)

| Tema | Decisión / propuesta | Dónde |
|---|---|---|
| Umbrales de `size` | `area_ratio` < 0,333 % → small; < 3 % → medium; si no, large (COCO 32²/96² px sobre una imagen de 640×480) | `scene.size_thresholds`, `docs/eda_findings.md` |
| Grilla 3×3 | Tercios iguales de ancho y alto; la celda la decide el centroide; un punto sobre el borde va a la celda de la derecha/abajo | E1-13 |
| Convención de `centroid` | `[x, y]` en píxeles, media de los índices de los píxeles de la máscara (el ejemplo da `[0,0]` para el píxel (0, 0)) | E1-13 |
| Convención de `bbox` desde la máscara | `[x_min, y_min, x_max + 1, y_max + 1]` (borde exclusivo, como el ejemplo `[0,0,1,1]`) | E1-13 / E1-16 |
| Qué cuenta como "consistencia estructural" en la métrica | Desconocido. Asumimos: campos derivados coherentes con la máscara, ids únicos y referencias válidas (`structural_errors`) | E1-19 |
| Semántica de `confidence` | Score del detector (objetos); regla del PLAN §4.5 (respuestas) | E1-16 / E1-22 |
| `near`, `on_top_of` | Fuera de `relations`; si se implementan, van en un campo propio `extra_relations` | D-04 |
| Máscara vacía | No valida (`area_ratio > 0`): el objeto se descarta o se usa la caja como máscara, con la marca correspondiente | E1-13 |
| Ids de objetos | Libres; usamos `obj_<n>` únicos por imagen | E1-16 |
| Respuesta cuando hay varias instancias en una pregunta singular | Pendiente (ver D-10) | E1-22 |
| Texto de `answer` y `normalized_answer` de `description` | Libre; definimos la plantilla en E1-27 | E1-27 |
