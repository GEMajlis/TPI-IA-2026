# Checklist final de entrega - Etapa 1

_Grupo 15 · Curso 5K1 · Verificación: 24 de octubre de 2026_

## Resultado

La implementación de la Etapa 1 cumple los requisitos técnicos de **promoción**. La validación
automatizada está en `docs/delivery_validation.json`; se regenera con:

```bash
python scripts/validate_delivery.py --artifacts outputs/kaggle/final
pytest -q
ruff check .
ruff format --check .
```

Runs finales:

- Desarrollo: `20261025-183446-6f583413` (120 imágenes).
- Submission y QA: `20261025-184746-6f583413` (80 imágenes).
- Hardware: Tesla T4 en Kaggle.

## Condiciones de validez

| Condición del enunciado | Estado | Evidencia |
|---|---|---|
| Notebook reproducible, completa y auditable | Cumplida | `TPI_IA_2026P_15.ipynb` ejecutada de punta a punta; ZIP final con métricas, trazas y manifests |
| Inferencia sin consultar anotaciones | Cumplida | `tests/test_import_boundary.py`; GT restringido a `src/eval/` y `src/eda/` |
| Reporte consistente con la implementación | Cumplida | `scripts/build_report.py`, `report/report_data.json` y SHA-256 de cada fuente |
| Defensa con comprensión de todos los integrantes | Acción manual | Preparar exposición de hasta 10 minutos y preguntas para los cuatro integrantes |

## Nivel aprobado

| Requisito | Estado | Evidencia |
|---|---|---|
| EDA relevante que fundamenta decisiones | Cumplido | 120 imágenes, 1790 instancias, tamaño, oclusión, frecuencia, política LVIS y benchmark; `docs/eda_findings.md` |
| Pipeline visual completa | Cumplido | OWLv2 → post-proceso → SAM2 → SceneJSON en `src/scene/pipeline.py` |
| F1 de detección >= 0,40 | Cumplido | F1 de cajas 0,653 en desarrollo, IoU >= 0,5 y evaluación LVIS federada |
| Uso efectivo de máscaras y JSON válido | Cumplido | F1 de máscaras 0,596; propiedades desde máscara; 0 errores de schema en 80 filas |
| `questions` determinista con existencia y conteo | Cumplido | 60 respuestas sobre 20 escenas; existencia 90% y conteo 40% contra oráculo |
| Ejecución modular, reproducible y trazada | Cumplido | 480 registros de desarrollo, 360 de submission y 0 errores |
| Evaluación cuantitativa y fidelidad de QA | Cumplido | Métricas de cajas, máscaras, atributos, consistencia, QA y evidencia reejecutable |
| Comparación experimental y casos correctos/fallidos | Cumplido | `outputs/experiments.csv` y galería de 4 casos correctos, 4 ambiguos y 4 fallidos |

## Nivel promoción

| Requisito | Estado | Evidencia |
|---|---|---|
| F1 de detección >= 0,50 | Cumplido | 0,653 en las 120 imágenes de desarrollo |
| Atributos y relaciones consultables | Cumplido | Posición 97,2%, tamaño 95,1%, color 89,9% y relaciones espaciales en SceneJSON |
| `questions_llm` con LLM liviano en Kaggle | Cumplido | Qwen2.5-3B fp16 en T4; 60 salidas, JSON y DSL válidos al 100% |
| El LLM recibe sólo texto y genera el JSON intermedio | Cumplido | `scene.llm.interpreter`; la imagen y los campos estructurados no llegan al modelo |
| Ambos modos resuelven ubicación, comparación, atributos y relaciones | Cumplido | DSL compartido con `position`, `size`, `attribute` y `relation`; mismo ejecutor determinista |
| Ambigüedad, falta de información y fuera de alcance explícitos | Cumplido | Estados `ambiguous`, `insufficient_information` y `unsupported`, con pruebas unitarias |
| Tracing completo y análisis de propagación | Cumplido | Parámetros, tiempos, estados, memoria y artefactos; embudo de 1790 instancias |
| Mejora justificada contra línea base | Cumplido | Umbral 0,10 → 0,18 adoptado por holdout; variante de máscaras descartada al bajar 0,614 → 0,609 |

## Requisitos de la notebook

- [x] Corre completamente en Kaggle usando sólo el dataset oficial y modelos públicos.
- [x] Fija semilla 42 y registra configuración, versiones, hardware y modelos.
- [x] Está organizada en secciones Markdown.
- [x] Conserva resultados cuantitativos auditables en JSON y CSV.
- [x] Distingue celdas `ground-truth`, `inference` y documentación.
- [x] Genera `submission.csv` y `grupo15_entrega_artifacts.zip`.
- [x] El nombre es `TPI_IA_2026P_15`.
- [x] El reporte se generó desde los mismos artefactos de la notebook.

## Verificaciones adicionales

- [x] Las 80 filas conservan ids y orden de `sample_submission.csv`.
- [x] Los 80 SceneJSON validan contra `scene_schema.json` y sus referencias internas existen.
- [x] Las 20 escenas del benchmark incluyen 60 `questions` y 60 `questions_llm`.
- [x] La pipeline procesa imágenes externas mediante `image_record_from_path` sin GT; lo cubre
  `tests/test_pipeline_submission.py::test_pipeline_genera_escenas_validas_y_traza`.
- [x] Los manifests de desarrollo y submission comparten el mismo hash de configuración.
- [x] El benchmark LLM registra CUDA, Tesla T4, memoria y latencia.
- [x] El submission fue aceptado por Kaggle con `PipelineScore = 0.6509`.

## Acciones manuales antes de entregar

- [ ] Crear y aprobar el PR de `desarrollo` a `main`.
- [ ] Confirmar que la versión guardada de Kaggle sea accesible para la cátedra.
- [ ] Ensayar una defensa de hasta 10 minutos; todos deben explicar cualquier componente.
- [ ] No modificar la entrega después del 1 de noviembre de 2026 a las 23:59.
