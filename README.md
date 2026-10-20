# TPI IA 2026 — Comprensión visual de escenas (UTN FRC)

Pipeline: imagen → detección open-vocabulary → SAM2 → SceneJSON → preguntas (modo determinista y modo LLM).

- Plan técnico: [docs/PLAN.md](docs/PLAN.md)
- Backlog: [docs/TASKS.md](docs/TASKS.md)
- Decisiones y desvíos del plan: [docs/DECISIONS.md](docs/DECISIONS.md)
- Reglas para agentes: [CLAUDE.md](CLAUDE.md)

## Instalación

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

## Datos

- `data/competition/`: archivos normativos (schema, `sample_submission.csv`, benchmark de preguntas, vocabulario). Se versionan.
- `data/dataset/`: copia local de la fuente de datos de la competencia, con la misma estructura que monta Kaggle
  (`development/`, `tpi_ia_2026_competition_public/`, `tpi_2026_qa_public/`). No se versiona.
- En Kaggle se usa `configs/kaggle.yaml` como override para apuntar a `/kaggle/input/...`.

**Regla de oro:** las anotaciones (GT) solo se leen desde `src/eval/` y `src/eda/`. Lo verifica `tests/test_import_boundary.py`.
