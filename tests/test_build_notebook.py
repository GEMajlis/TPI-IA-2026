import ast
import json

import nbformat

from scripts.build_notebook import ROOT, build_notebook, load_delivery, runtime_files


def test_delivery_tiene_datos_del_grupo() -> None:
    cfg = load_delivery(ROOT / "configs" / "delivery.yaml")
    assert cfg.group_number == "15"
    assert cfg.course == "5K1"
    assert cfg.kaggle_team == "Grupo 15 - 5K1"
    assert len(cfg.members) == 4
    assert cfg.output.name == "TPI_IA_2026P_15.ipynb"


def test_notebook_es_autocontenida_y_separa_gt() -> None:
    cfg = load_delivery(ROOT / "configs" / "delivery.yaml")
    notebook = build_notebook(ROOT, cfg)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["tpi_ia"]["group_number"] == "15"

    sources = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "Gonzalo Ezequiel Majlis (95047)" in sources
    assert "Franco Manuel Lucero (96311)" in sources
    assert 'run_pipeline("--split", "dev", "--eval")' in sources
    assert 'run_pipeline("--split", "test", "benchmark")' in sources
    assert "scripts/run_qa_llm.py" in sources
    assert "scripts/analyze_errors.py" in sources
    assert "PipelineScore = 0.6509" in sources
    assert "grupo15_entrega_artifacts.zip" in sources

    write_cells = [cell for cell in notebook["cells"] if "source" in cell["metadata"]["tags"]]
    written = {cell["source"].splitlines()[0].removeprefix("%%writefile ") for cell in write_cells}
    expected = {path.relative_to(ROOT).as_posix() for path in runtime_files(ROOT)}
    assert written == expected
    for cell in write_cells:
        first = cell["source"].splitlines()[0]
        if first.startswith("%%writefile src/scene/"):
            assert "inference" in cell["metadata"]["tags"]
            assert "ground-truth" not in cell["metadata"]["tags"]
        if first.startswith(("%%writefile src/eval/", "%%writefile src/eda/")):
            assert "ground-truth" in cell["metadata"]["tags"]


def test_notebook_generada_es_valida_y_sin_salidas() -> None:
    cfg = load_delivery(ROOT / "configs" / "delivery.yaml")
    serialized = json.dumps(build_notebook(ROOT, cfg), ensure_ascii=False)
    loaded = json.loads(serialized)
    nbformat.validate(nbformat.from_dict(loaded))
    ids = [cell["id"] for cell in loaded["cells"]]
    assert len(ids) == len(set(ids))
    for cell in loaded["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            if not cell["source"].lstrip().startswith("%"):
                ast.parse(cell["source"])
