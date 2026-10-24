# Hallazgos del EDA (E1-06)

Datos: 120 imágenes de desarrollo, 1790 instancias, 120 categorías. Se regenera con
`python scripts/run_eda.py` (tablas en `outputs/eda/*.csv`, figuras en `outputs/eda/figures/`, resumen en `outputs/eda/eda_summary.json`).
Código: `src/eda/stats.py` (tablas) y `src/eda/plots.py` (una figura por pregunta).

## H1 · Los objetos son muy chicos → el recall en objetos chicos es el cuello de botella
- Mediana del área relativa: **0,29 %** de la imagen. El **29 %** de los objetos ocupa menos del 0,1 % y el **71 %** menos del 1 %.
- Con los umbrales COCO absolutos (32² y 96² px), el 54 % de los objetos es *small*.
- **Implicancia:** el F1 de detección va a depender sobre todo de los objetos chicos. Hay que reportar las métricas por tamaño desde la línea base, y el tiling tipo SAHI (PLAN §4.9) es el primer candidato para la extensión si el recall en *small* es bajo. Las imágenes tienen 640 px de lado mayor y OWLv2 trabaja a 960, así que ya hay un reescalado ×1,5 implícito.
- Figura: `area_ratio.png`.

## H2 · Umbrales de tamaño propuestos (el schema no los fija)
- Se propone **small < 0,333 %**, **medium < 3 %**, **large ≥ 3 %** del área de la imagen: equivalen a 32² y 96² px (convención COCO) sobre una imagen de referencia de 640×480, y aplicados como fracción respetan la pregunta "¿Qué tamaño **relativo** tiene…?".
- Reparto resultante en el GT: 52 % small, 33 % medium y 15 % large.
- Los 5 objetos de las preguntas de tamaño del benchmark quedan así: necklace, watch y spectacles → small; elephant (0,57 %) → medium; tablecloth (7,5 %) → large.
- **Implicancia:** `scene.size_thresholds` en `configs/base.yaml`. Si el enunciado o la cátedra definen otros umbrales, se cambian ahí y nada más.

## H3 · Pocas instancias por imagen, con cola larga → tope de detecciones holgado
- Mediana de 10,5 instancias por imagen y media de 14,9. El máximo es 66 y solo el 3 % de las imágenes tiene más de 50.
- **Implicancia:** `detection.max_detections = 100` (el schema admite 300). El tope no debería recortar recall; se valida con el embudo de trazas (E1-31).
- Figura: `instances_per_image.png`.

## H4 · La evaluación federada ignora la mayoría de las predicciones → conviene un umbral bajo
- Por imagen hay en promedio **6,4 categorías positivas**, **1,0 negativa** y 0,4 no exhaustivas. Las otras **112,7 de 120 no están verificadas**: sus predicciones se ignoran.
- Los FP solo salen de (a) predicciones de categorías positivas que no matchean (duplicados o mala localización) y (b) las ~1 categoría negativa por imagen.
- **Implicancia:** el umbral que maximiza F1 probablemente sea bajo (el recall pesa más que las alucinaciones de categorías ausentes). Hay que medir siempre la **cobertura evaluable** junto al F1 y ajustar el umbral con el split de tuning (E1-29). Ojo: un umbral muy bajo puede no generalizar si el GT oculto del test tiene más negativas.
- Figura: `federated_status.png`.

## H5 · Categorías balanceadas, sin cola de raras extrema
- Cada categoría es positiva en **3 a 11 imágenes** (el dataset está curado). Las 10 más frecuentes suman solo el 20 % de las instancias.
- Bins para el desglose por frecuencia: **rare ≤ 4 imágenes (25 categorías)**, **common 5–7 (61)**, **frequent ≥ 8 (34)** (`eval.frequency_bins`).
- **Implicancia:** con 3–11 imágenes por clase, un umbral **por clase** se sobreajustaría; la línea base usa un umbral global, y el umbral por clase solo entra como experimento con regularización.
- Figura: `category_frequency.png`.

## H6 · Mucha superposición y objetos anotados con dos categorías → sin NMS entre clases en la base
- El **56 %** de las instancias comparte píxeles con otra y el **16 %** tiene más de la mitad de su máscara compartida (partes dentro de objetos: wheel/car, button/shirt, knob/drawer).
- Solo el **1,7 %** tiene otra instancia **de su misma clase** con IoU de caja ≥ 0,5 → el NMS por clase con IoU 0,5 casi no elimina objetos reales.
- Hay **25 pares de instancias de clases distintas con IoU de caja ≥ 0,8**, y varios son **el mismo objeto con dos etiquetas** (cup/drinking glass, hat/baseball cap, television set/computer monitor, bed/blanket, handbag/suitcase).
- **Implicancia:** el NMS entre clases (PLAN §4.1) eliminaría TP reales. En la línea base queda **desactivado** y se evalúa como experimento (E1-30). El post-proceso de SAM2 no debe forzar máscaras disjuntas.
- Figura: `occlusion.png`.

## H7 · Los centroides se concentran en el centro y la fila inferior
- `center` 19 %, `middle_right` y `bottom_center` 13 % cada una; la fila superior tiene entre 8 y 9 % por celda.
- **Implicancia:** la grilla 3×3 por tercios produce las 9 clases con frecuencia razonable. No hace falta ajustarla. Es un buen chequeo de sanidad para la consistencia de `position` (E1-19).
- Figura: `position_grid.png`.

## H8 · El benchmark de QA es "amable" con la percepción, salvo en existencia
- 60 consignas: 20 descripciones, 10 count, 10 existence, 10 relation, 5 position y 5 size.
- En **count** la categoría siempre es positiva, con 2 a 4 instancias. En **position, size y relation** cada clase mencionada tiene **exactamente 1 instancia** en el GT.
- En **existence**: 5 positivas ("yes") y **5 negativas** verificadas ("no": belt, sofa, plate, dog, napkin).
- **Implicancia:** si se detectan varias instancias en una pregunta singular, es más probable que sobren detecciones que que la referencia sea ambigua. Se propone responder con la instancia de mayor confianza, bajar la confianza y registrar el caso (D-10). Las preguntas de existencia negativa son sensibles a los FP de esa clase: conviene un umbral de existencia propio, más alto que el de detección.
- Figura: `benchmark_types.png`.
