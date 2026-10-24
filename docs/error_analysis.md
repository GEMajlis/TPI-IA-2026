# Análisis de errores y propagación (E1-31)

Run analizado: dev completo (120 imágenes, 1790 instancias GT), configuración base con umbral 0,18 (`20261024-190854-e834f3e7` en Kaggle).
Se regenera con `python scripts/analyze_errors.py --run <run>` (salidas en `<run>/error_analysis/`).
Las etapas de detección del embudo se recalculan desde la caché de detecciones crudas con el mismo post-proceso que la pipeline y cuentan una instancia como "cubierta" si alguna caja de su clase tiene IoU ≥ 0,5. Las dos últimas etapas usan el matching uno a uno del evaluador federado.

## 1. Embudo: en qué etapa se pierde cada instancia

| Etapa donde se pierde | Total | small (937) | medium (594) | large (259) |
|---|---|---|---|---|
| Ninguna caja cruda la cubre (`no_box`) | 208 (11,6 %) | 190 (20,3 %) | 18 (3,0 %) | 0 |
| Solo la cubren cajas de otra clase (`wrong_class`) | 150 (8,4 %) | 79 (8,4 %) | 50 (8,4 %) | 21 (8,1 %) |
| Clase correcta bajo el umbral (`below_threshold`) | 298 (16,6 %) | 190 (20,3 %) | 73 (12,3 %) | 35 (13,5 %) |
| NMS | 8 (0,4 %) | 4 | 2 | 2 |
| Tope de detecciones | 0 | 0 | 0 | 0 |
| Matching uno a uno (`box_match`) | 10 (0,6 %) | 4 | 5 | 1 |
| Caja TP, máscara con IoU < 0,5 (`mask`) | 123 (6,9 %) | 94 (10,0 %) | 18 (3,0 %) | 11 (4,2 %) |
| **TP de máscara (`ok`)** | **993 (55,5 %)** | **376 (40,1 %)** | **428 (72,1 %)** | **189 (73,0 %)** |

Recall acumulado después de cada etapa: 0,884 (alguna caja) → 0,800 (de la clase correcta) → 0,634 (sobre el umbral) → 0,629 (tras NMS y tope) → 0,623 (TP de caja) → **0,555 (TP de máscara)**.

**Conclusiones**

1. **El error se origina en la detección y casi no se propaga por el post-proceso.** NMS, tope y matching juntos pierden el 1 % de las instancias. Lo que se pierde, se pierde en OWLv2: o no propone ninguna caja (11,6 %), o la propone con otra clase (8,4 %), o con score bajo (16,6 %).
2. **Los objetos chicos concentran el problema.** Son el 52 % del GT, pero solo el 40 % llega a TP contra ~72 % en medium y large. De las 937 instancias chicas, 190 no tienen ninguna caja cerca y otras 190 tienen una caja de la clase correcta con score bajo el umbral.
3. **El umbral no es la palanca.** Las 298 instancias bajo el umbral se recuperarían bajándolo, pero el barrido de E1-29 muestra que eso agrega más FP que TP (con 0,10 el F1 de holdout baja de 0,666 a 0,619). Hace falta una señal de detección más fuerte, no un umbral más permisivo.
4. **La segmentación agrega un error moderado y también concentrado en small.** El 10 % de las chicas pierde el TP en la máscara (3–4 % en medium/large). Con cajas GT, 1 de cada 4 chicas no llega a IoU 0,5 (D-15): es un límite de SAM2 a esta resolución.
5. **La confusión de clase es transversal al tamaño** (~8 % en los tres): son objetos que OWLv2 ve pero nombra mal, típicamente partes o vecinos semánticos.

## 2. Falsos positivos (469 FP de cajas)

| Tipo | Cantidad | Definición |
|---|---|---|
| Mala localización | 233 (49,7 %) | IoU en [0,1; 0,5) con un GT de su clase |
| Fondo | 213 (45,4 %) | sin GT de su clase debajo, en una categoría positiva y exhaustiva |
| Duplicado | 11 (2,3 %) | IoU ≥ 0,5 con un GT ya tomado por una predicción de mayor score |
| Confusión de clase | 8 (1,7 %) | IoU ≥ 0,5 con un GT de otra clase |
| Categoría negativa | 4 (0,9 %) | clase verificada como ausente |

- **El NMS ya hace su trabajo:** los duplicados son 11. Un NMS más agresivo o entre clases no tiene de dónde ganar (coincide con D-09 y D-14).
- **Los FP son de ubicación, no de nombre:** el 95 % son cajas mal ubicadas o sobre fondo. Las de mala localización suelen ser cajas que agrupan varias instancias (una sola caja para varios libros) o que cubren solo una parte del objeto.
- **Los FP se concentran justo sobre el umbral:** 349 de 469 (74 %) tienen score entre 0,18 y 0,3. Con score ≥ 0,4 quedan 40 FP.
- Las confusiones de clase son escasas y entre vecinos semánticos (watch ↔ bracelet, necktie ← shirt, teddy bear ← elephant).

## 3. Clases que más instancias pierden

book (44), car (26), wheel (25), person (22), curtain (22), strap (20), chair (19), shoe (18), broccoli (17), pillow (16). Son objetos **repetidos y chicos** (libros en un estante, ruedas, brócoli en un plato) y **partes** de otros objetos (strap, wheel). En las repeticiones OWLv2 tiende a dar una caja para el grupo, que cuenta como FP de localización y deja las instancias como FN. La tabla completa está en `worst_classes.csv`.

## 4. Galería

En `error_analysis/gallery/`, con el GT en verde (FN en amarillo) y las predicciones en azul (TP), rojo (FP) o gris (ignoradas por la política federada):

- **Correctas** (mayor F1 por imagen): tpi_0109, tpi_0036, tpi_0017, tpi_0086.
- **Fallidas** (menor F1): tpi_0004, tpi_0038, tpi_0064, tpi_0093.
- **Ambiguas** (confusiones de clase o muchas predicciones ignoradas): tpi_0016, tpi_0029, tpi_0104, tpi_0100.

## 5. Preguntas (modo determinista, E1-23)

Las 60 consignas del benchmark se responden. 9 quedan con un estado distinto de `ok`: 5 `ambiguous` (1 de posición, 3 de relación y 1 de tamaño: había varias instancias y se usó la de mayor confianza, D-10) y 4 `insufficient_information` (2 de relación y 2 de tamaño: no se detectó el objeto). Ninguna queda `unsupported`. La atribución de cada error de QA a percepción o a razonamiento queda para E1-24, con el SceneJSON oráculo.

## 6. Qué motiva para la Etapa 2

| Candidata | Error que ataca | Techo (instancias) |
|---|---|---|
| **Tiling tipo SAHI** (E2-02b) | `no_box` y `below_threshold` en objetos chicos: con recortes el objeto se ve más grande y OWLv2 lo detecta con más score | hasta 380 (21 % del GT) |
| Re-scoring con CLIP sobre el recorte enmascarado (E2-02a) | `wrong_class` (150) y parte de `below_threshold` (298) | hasta ~450, pero con el riesgo de subir FP de fondo |
| Ensamble con WBF (E2-02c) | Localización (233 FP) | indirecto |

**Recomendación:** tiling tipo SAHI. Ataca la mayor fuente de pérdida medida, que además está concentrada en un subgrupo identificable (small), y permite una hipótesis falsable: sube el recall en small sin bajar la precisión en medium/large. Su costo es de cómputo (varias pasadas de OWLv2 por imagen) y hay que medirlo.
