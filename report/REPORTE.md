# TPI IA 2026P - Comprensión visual de escenas

## 1. Portada

**Grupo 15 - Curso 5K1**

**Equipo Kaggle:** Grupo 15 - 5K1

**Etapa:** 1 - Promoción

**Notebook:** `TPI_IA_2026P_15.ipynb`

### Integrantes

- Gonzalo Ezequiel Majlis (95047)
- Tomas Morinigo (95639)
- Ramiro Figueroa Manzi (96306)
- Franco Manuel Lucero (96311)

## 2. Análisis del dataset

El conjunto de desarrollo contiene 120 imágenes, 1790 instancias y
120 categorías
evaluables. La mediana es 10,5 instancias por imagen y el
máximo es 66. El 71,1% de los
objetos ocupa menos del 1% de la imagen, y el 56,1% se
superpone con otra instancia. Esto anticipa dificultades de recall, oclusión y segmentación fina.

| Tamaño por área relativa | Proporción |
|---|---:|
| Small (< 0,333%) | 52,3% |
| Medium (< 3%) | 33,2% |
| Large | 14,5% |

Las categorías aparecen en entre 3 y
11 imágenes. Debido a la política federada de LVIS, una imagen
tiene en promedio 6,37 categorías positivas,
0,98 negativas verificadas y
112,65 no verificadas. Por eso la precisión se
reporta junto con la cobertura evaluable y no se consideran falsos positivos las categorías que
LVIS no permite juzgar. El benchmark lingüístico tiene 60 consignas: 20 descripciones y 40
preguntas cerradas.

**Decisiones derivadas.** Se usan umbrales de tamaño 0,00333 y 0,03, NMS por clase para no borrar
instancias legítimas de clases superpuestas, tope de 100 detecciones y análisis separado por tamaño.

## 3. Metodología

La pipeline es `imagen -> OWLv2 -> post-proceso -> SAM2 -> SceneJSON -> QA`. El detector
`google/owlv2-base-patch16-ensemble` consulta las 120 categorías con el prompt
`a photo of a {}`, conserva detecciones crudas desde score 0,01 y
aplica umbral 0,18, NMS por clase con IoU 0,5 y un máximo de 100 objetos. Las cajas resultantes se
usan como prompts de `facebook/sam2.1-hiera-large`; por caja se elige la mejor de
tres máscaras y se aplican recorte con margen, componente principal y relleno de huecos.

El SceneJSON guarda ids de instancia, clase, score, caja, máscara RLE no comprimida column-major,
centroide, área relativa, posición 3x3, tamaño, color dominante y relaciones espaciales. Cada campo
distingue datos observados, derivados y desconocidos. La validación usa el `scene_schema.json`
normativo y chequeos de referencias, ids y tamaños de máscara.

El modo determinista transforma los campos estructurados del benchmark en un DSL validado. El
modo LLM usa `Qwen/Qwen2.5-3B-Instruct` en fp16: recibe
solamente el texto, genera el mismo DSL, y el ejecutor determinista calcula la respuesta. El LLM
verbaliza y describe, pero nunca recibe la imagen, el GT ni calcula conteos o relaciones. Los
estados `ambiguous`, `unsupported` e `insufficient_information` evitan inventar datos.

Las anotaciones se leen exclusivamente desde `src/eval/` y `src/eda/`. La inferencia acepta una
imagen arbitraria y no importa esos paquetes. Cada etapa registra parámetros, modelo, tiempos,
estado, conteos, memoria y artefacto en `trace.jsonl`. La semilla global y del LLM es
42.

### Modelos, recursos y código reutilizado

- OWLv2: `google/owlv2-base-patch16-ensemble`, descargado de Hugging Face.
- Segmentación: `facebook/sam2.1-hiera-large`, descargado de Hugging Face.
- LLM: `Qwen/Qwen2.5-3B-Instruct`; candidato comparado: `Qwen/Qwen3-4B`.
- Matching semántico de respaldo: `sentence-transformers/LaBSE`.
- Sinónimos LVIS: metadata pública de Detectron2, descargada y parseada sin ejecutar código.
- Librerías: PyTorch, Transformers, NumPy, OpenCV, Pillow, pandas, scikit-learn, pycocotools,
  pydantic y jsonschema. No se incorporó código ad hoc de terceros al repositorio.
- Ejecución final: Python 3.13.15, PyTorch
  2.11.0+cu128 y GPU Tesla T4 de
  14,6 GiB en Kaggle.

## 4. Resultados

### 4.1 Percepción sobre desarrollo

Se evaluaron las 120 imágenes con IoU >= 0,5 y política federada LVIS.

| Etapa | Precisión | Recall | F1 | F1 macro | Cobertura evaluable |
|---|---:|---:|---:|---:|---:|
| Cajas | 0,700 | 0,612 | **0,653** | 0,674 | 0,439 |
| Máscaras | 0,643 | 0,555 | **0,596** | 0,619 | 0,434 |

El F1 de cajas supera la meta de promoción de 0,50.

| Tamaño | F1 cajas | F1 máscaras |
|---|---:|---:|
| Small | 0,567 | 0,457 |
| Medium | 0,727 | 0,710 |
| Large | 0,745 | 0,781 |

Con cajas GT, SAM2 alcanza IoU >= 0,5 en
84,1% de 1790
instancias. Con cajas predichas, 137 TP de caja se
pierden en la etapa de máscara, lo que confirma la propagación del error de localización.

### 4.2 Representación estructurada

Se validaron 120 escenas, 3564 objetos y
265934 relaciones con **0 inconsistencias**. Sobre
993 pares TP, la exactitud fue 97,2% en posición,
95,1% en tamaño y 89,9% en color. El
5,6% de los objetos tiene más del 10%
de su máscara fuera de la caja, principal alerta de consistencia geométrica.

### 4.3 Preguntas y descripciones

| Métrica | Determinista | LLM |
|---|---:|---:|
| Exactitud vs. oráculo (40 cerradas) | 72,5% | 72,5% |
| Evidencia válida | 100,0% | 100,0% |
| Reejecución coincide | 100,0% | 100,0% |
| JSON intermedio válido | - | 100,0% |
| DSL igual al modo estructurado | - | 100,0% |

Por tipo, ambos modos logran count 40,0%,
existence 90,0%, position
100,0%, relation
80,0% y size
60,0%. De 20 descripciones LLM,
90,0% fueron fieles y
2 usaron el fallback determinista. Los
11 errores cerrados se atribuyeron a percepción, no a la
interpretación del LLM.

### 4.4 Experimentos controlados

Cada comparación cambia una sola variable respecto de la línea base y se decide con el holdout de
40 imágenes (split estratificado con semilla 42), que no participa de la elección.

**Mejora adoptada: umbral de detección.** Sobre la caché de detecciones crudas se barrieron
los umbrales 0,02 a 0,50; el de máximo F1 de cajas en tuning fue 0,18 (F1 0,646 en tuning). En holdout:

| Configuración | Precisión | Recall | F1 de cajas | Predicciones por imagen |
|---|---:|---:|---:|---:|
| Línea base, umbral 0,10 | 0,550 | 0,708 | 0,619 | 57,5 (dev completo) |
| **Umbral elegido, 0,18** | 0,719 | 0,620 | **0,666** | 28,2 |
| Umbral elegido + NMS entre clases (IoU 0,8) | 0,721 | 0,620 | 0,667 | 28,1 |

El umbral elegido **mejora el F1 de holdout en 4,7 puntos** frente a la línea base: cambia recall por
precisión y reduce a la mitad las predicciones por imagen. El NMS entre clases no aporta y queda
apagado. Medición registrada en `docs/DECISIONS.md` (D-14), corrida en Kaggle con
`scripts/tune_thresholds.py`; se reproduce con ese script sobre el run de desarrollo.

**Variante descartada: post-proceso de máscaras solo en objetos chicos.** Se cambió una sola variable: `apply_below_area_ratio: null -> 0.00333`. El F1 de máscaras en holdout pasó de
0,614 a 0,609
(-0.0054); aunque tuning mejoró
+0.0096, la variante se **descartó** porque empeoró el
holdout. El costo de segmentación fue 125,4 s en la base y
123,9 s en la variante. El oráculo sugería una mejora, pero el
experimento end-to-end mostró que las cajas predichas cambian el efecto del post-proceso.

**Intérprete LLM: prompt v1 → v2.** Con la percepción fija, el prompt v1 daba JSON válido en el
65% de las consignas (las 20 descripciones quedaban `unsupported` por falta de un ejemplo de
`describe`) y usaba categorías del benchmark como ejemplos. El v2, sin contenido del benchmark,
llega al 100% y es el que se usa en este run. Es una medición previa a la corrida final,
registrada en `docs/DECISIONS.md` (D-18).

### 4.5 Recursos, traza y entrega

| Etapa del run dev | Registros | Tiempo acumulado (s) | Pico GPU trazado (MiB) | Errores |
|---|---:|---:|---:|---:|
| detection | 120 | 16,7 | 1299 | 0 |
| detection_postprocess | 120 | 0,1 | 1209 | 0 |
| segmentation | 120 | 125,4 | 2671 | 0 |
| scene | 120 | 54,1 | 1209 | 0 |

La traza de desarrollo contiene 480 registros sobre 120 imágenes y
0 errores. En el benchmark de tres prompts, Qwen2.5-3B usó
5994 MiB y 5214 ms promedio; Qwen3-4B
4-bit usó 2601 MiB pero 10742 ms, por
lo que se eligió el primero.

El submission final contiene 80 filas, 2271 objetos,
164796 relaciones, 20 escenas de benchmark,
60 respuestas deterministas y 60 respuestas
LLM; no presenta errores de schema y pesa 15,44 MiB. La referencia
registrada del leaderboard es `PipelineScore = 0,6509`.

## 5. Discusión y análisis de errores

De las 1790 instancias GT, 55,5% llega a TP de máscara. Las pérdidas
principales aparecen antes de SAM2: 11,6% sin caja candidata,
8,4% con clase incorrecta y 16,6%
bajo el umbral. Sólo 0,4% se pierde por NMS y ninguna por el tope de
detecciones. En objetos small llega a máscara correcta el
40,1%, contra
72,1% en medium y
73,0% en large.

Los falsos positivos se concentran en mala localización (233) y
fondo (213); duplicados, confusiones y categorías negativas son
minoritarios. Las clases más problemáticas combinan objetos chicos o delgados (`strap`,
`doorknob`, `streetlight`) y bajo recall (`book`, `curtain`, `person`). La galería del run conserva
cuatro casos correctos, cuatro ambiguos y cuatro fallidos.

Las limitaciones principales son: recall bajo en objetos pequeños; dependencia de SAM2 respecto de
la caja; cobertura evaluable LVIS de aproximadamente 43,9%; conteo QA
de sólo 40,0%; y un benchmark de LLM pequeño.
La gran cantidad de relaciones también aumenta el tamaño del envío, aunque no produjo
inconsistencias.

## 6. Conclusiones y trabajo futuro

La Etapa 1 cumple el nivel promoción: F1 de cajas superior a 0,50, SceneJSON válido con atributos y
relaciones, QA determinista y con LLM sobre un único ejecutor, manejo explícito de incertidumbre,
tracing completo y experimento controlado. El resultado muestra que el cuello de botella está en la
percepción, especialmente en objetos pequeños, y no en la interpretación lingüística.

La extensión prioritaria para una Etapa 2 es tiling tipo SAHI con fusión de detecciones, evaluada
sobre el holdout y con medición de costo. También se propone reducir relaciones redundantes y
calibrar conteos sin usar GT durante inferencia.

## Trazabilidad

El reporte fue generado por `scripts/build_report.py`. El snapshot
`report/report_data.json` conserva los datos usados y el SHA-256 completo de cada fuente del ZIP.

| Clave | Archivo dentro de `grupo15_entrega_artifacts.zip` | SHA-256 |
|---|---|---|
| `final_summary` | `outputs/final_summary.json` | `340c96fff2943426da96630206180cba05436d81faca014e12488547a9227497` |
| `eda` | `outputs/eda/eda_summary.json` | `dfd312d675421ece741b9d37a3fefc2a3faec1f7360222a0260725d6ca564e80` |
| `detection` | `outputs/runs/20261025-183446-6f583413/metrics_detection.json` | `3f05719345b185b95203b9bc7620ab09b5b2d81ea2c205a36d5edef5e66d1974` |
| `segmentation` | `outputs/runs/20261025-183446-6f583413/metrics_segmentation.json` | `80db72b22cf1ded681bc27d44bbc984f6565b4d49dce0158bcd68bf40c144465` |
| `segmentation_oracle` | `outputs/runs/20261025-183446-6f583413/metrics_segmentation_oracle.json` | `9c00351d3e790fe50d069bd12f71d36b0a9f7d1327a743a7d881ae2eb69aa332` |
| `scene` | `outputs/runs/20261025-183446-6f583413/metrics_scene.json` | `f3fa33a27a4a42fd2b199787d920b959a0d642d4d7ebea3c8a0bb70322b6e820` |
| `qa` | `outputs/runs/20261025-184746-6f583413/metrics_qa.json` | `73e6cbac415e97400d1a4b28c35077aaa524f8b76bf48d378198bd8645f403f6` |
| `error_analysis` | `outputs/runs/20261025-183446-6f583413/error_analysis/summary.json` | `e13e8d300d5b75407184138d2de3f160b31f0aa0372b27b66944fdce8287ef1b` |
| `experiment` | `outputs/experiments.csv` | `b4f5b8f6a16b9be5573832497dcc6261cfc430be5c4f5055e97adf85d3366adb` |
| `llm_benchmark` | `outputs/llm_benchmark.json` | `c32b0508523665ce2d18d909eed0f97da9a5438d9e91a1fa270766209cb7e494` |
| `dev_manifest` | `outputs/runs/20261025-183446-6f583413/run_manifest.json` | `30dc56ee9a9773edceb66f6dd795fbd9d725a98b09bc6afac91d97acf3f65374` |
| `submission_manifest` | `outputs/runs/20261025-184746-6f583413/run_manifest.json` | `0399fcf600953fd07426113f075fc32e3afacb0b5ef74b02c777e24e9362f9e9` |
| `dev_trace` | `outputs/runs/20261025-183446-6f583413/trace.jsonl` | `da8934aa92060667ac3ad329a1fca565a3b4e1bded23d4cb615b6f0e987be9d4` |
| `submission_trace` | `outputs/runs/20261025-184746-6f583413/trace.jsonl` | `3d430696d4d0e1d137521d9b164f7c1fd74adacb953934daf238b1ff67e5e3a1` |
| `submission` | `submission.csv` | `b004322f70b2abc43008646932369948ff5036c7cca00b08157bdf6cc4a7696d` |
