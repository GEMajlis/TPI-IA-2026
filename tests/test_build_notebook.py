import ast
import json

import nbformat

from scripts.build_notebook import (
    ROOT,
    build_notebook,
    load_delivery,
    runtime_files,
    writefile_cell,
)


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
    assert "sys.path.insert(0, SRC_ROOT)" in sources
    assert "PipelineScore = 0.6509" in sources
    assert "grupo15_entrega_artifacts.zip" in sources

    write_cells = [cell for cell in notebook["cells"] if "source" in cell["metadata"]["tags"]]
    written = {cell["metadata"]["source_path"] for cell in write_cells}
    expected = {path.relative_to(ROOT).as_posix() for path in runtime_files(ROOT)}
    assert written == expected
    for cell in write_cells:
        if cell["source"].startswith("%%writefile"):
            assert cell["source"].partition("\n")[2].strip()
        source_path = cell["metadata"]["source_path"]
        if source_path.startswith("src/scene/"):
            assert "inference" in cell["metadata"]["tags"]
            assert "ground-truth" not in cell["metadata"]["tags"]
        if source_path.startswith(("src/eval/", "src/eda/")):
            assert "ground-truth" in cell["metadata"]["tags"]


def test_archivos_vacios_no_usan_writefile_magic() -> None:
    cfg = load_delivery(ROOT / "configs" / "delivery.yaml")
    notebook = build_notebook(ROOT, cfg)
    empty_paths = {
        path.relative_to(ROOT).as_posix()
        for path in runtime_files(ROOT)
        if path.stat().st_size == 0
    }
    empty_cells = [
        cell for cell in notebook["cells"] if cell["metadata"].get("source_path") in empty_paths
    ]

    assert {cell["metadata"]["source_path"] for cell in empty_cells} == empty_paths
    assert all(not cell["source"].startswith("%%writefile") for cell in empty_cells)
    assert all(".write_text('', encoding='utf-8')" in cell["source"] for cell in empty_cells)


def test_archivo_con_solo_espacios_no_usa_writefile_magic(monkeypatch) -> None:
    source = ROOT / "src" / "eda" / "__init__.py"
    monkeypatch.setattr(type(source), "read_text", lambda *_args, **_kwargs: "\n")

    cell = writefile_cell(source, ROOT)

    assert not cell["source"].startswith("%%writefile")
    assert ".write_text('\\n', encoding='utf-8')" in cell["source"]


def test_runtime_no_incluye_metadatos_de_instalacion() -> None:
    paths = runtime_files(ROOT)

    assert all(not any(part.endswith(".egg-info") for part in path.parts) for path in paths)


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
