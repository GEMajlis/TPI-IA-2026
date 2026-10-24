# Cómo ejecutar la notebook final en Kaggle

Notebook: `notebooks/TPI_IA_2026P_15.ipynb`

Equipo: **Grupo 15 - 5K1**

## 1. Abrirla en la competencia

1. Entrá a la competencia **TPI IA 2026 Primavera - Comprensión Visual de Escenas**.
2. Abrí la pestaña **Code**.
3. Creá una notebook con **New Notebook**. Hacerlo desde la competencia asegura que los datos
   oficiales queden asociados.
4. En el editor elegí **File → Import Notebook** y seleccioná
   `TPI_IA_2026P_15.ipynb` desde tu computadora.

La notebook es autocontenida: escribe el código y la configuración con `%%writefile`. No hace
falta clonar GitHub ni subir el repositorio.

## 2. Configurar la sesión

En el panel **Settings / Session options**:

- **Language:** Python.
- **Accelerator:** GPU T4 (si aparece `GPU T4 x2`, también sirve; el proyecto usa la primera GPU).
- **Internet:** On, porque OWLv2, SAM2, Qwen y las dependencias se descargan de fuentes públicas.

Después de cambiar el acelerador, Kaggle reinicia la sesión. Esperá a que figure conectada antes de
ejecutar. La [documentación oficial de Kaggle](https://www.kaggle.com/docs/notebooks) ubica Internet
y Accelerator dentro de Session options y advierte que activar GPU reinicia la sesión.

## 3. Verificar los datos

En **Input** debe aparecer la competencia. La ruta esperada es:

```text
/kaggle/input/competitions/tpi-ia-2026-primavera-comprension-visual-de-escenas
```

Si no aparece, usá **Add Input**, buscá la competencia y agregala. Tenés que haber aceptado sus
reglas con la misma cuenta.

## 4. Ejecutar

1. Elegí **Run All**.
2. No cambies el orden de las celdas ni edites los SceneJSON.
3. La primera ejecución demora porque descarga modelos y procesa desarrollo, el experimento y las
   80 imágenes de la submission.
4. La última celda debe imprimir **“Entrega validada”**. Si una celda falla, no continúes a mano:
   copiá el traceback completo y el título de la sección para corregir la causa.

Los cuatro controles costosos están activos por defecto en la celda de parámetros:

```python
RUN_ORACLE_SEGMENTATION = True
RUN_THRESHOLD_TUNING = True
RUN_CONTROLLED_EXPERIMENT = True
RUN_LLM_BENCHMARK = True
```

Para un smoke test se pueden desactivar, pero la versión final guardada debe ejecutarlos.

## 5. Guardar la versión final

Cuando termine sin errores:

1. Guardá una versión de la notebook con todas las salidas (`Save Version`).
2. Confirmá que `/kaggle/working/submission.csv` tiene 80 filas.
3. En el panel **Output**, descargá:
   - `submission.csv`;
   - `grupo15_entrega_artifacts.zip`;
   - la notebook ejecutada (`.ipynb`) con sus salidas.

El ZIP contiene métricas, CSV, manifests, trazas, figuras y el resumen final; excluye pesos, cachés y
SceneJSON pesados. Es el insumo para generar el reporte E1-33 y verificar E1-34.

## 6. Qué devolver para cerrar la entrega

Compartí estos dos archivos:

1. `TPI_IA_2026P_15.ipynb` ejecutada.
2. `grupo15_entrega_artifacts.zip`.

Si se vuelve a subir la nueva `submission.csv`, compartí también una captura del puntaje. El score
de referencia ya aceptado es `PipelineScore = 0.6509`.
