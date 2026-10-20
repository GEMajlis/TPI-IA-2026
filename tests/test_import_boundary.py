"""Frontera GT: la inferencia (`src/scene/`) no puede importar `eval`/`eda` ni tocar anotaciones.

NO desactivar este test (CLAUDE.md, regla de oro).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT

SCENE_DIR = ROOT / "src" / "scene"
FORBIDDEN_PACKAGES = {"eval", "eda"}
# Nombres de archivos de anotaciones (COCO/LVIS) que no pueden aparecer en la inferencia.
ANNOTATION_FILE_RE = re.compile(r"annotations?\.json|instances_\w*\.json|lvis_v1_\w*\.json", re.I)
# Claves que solo existen en el GT o en la sección de config exclusiva de evaluación.
FORBIDDEN_KEYS = {"eval", "annotations", "neg_category_ids", "not_exhaustive_category_ids"}


def _forbidden_module(name: str | None) -> bool:
    if not name:
        return False
    parts = name.split(".")
    if parts[0] == "src":
        parts = parts[1:]
    return bool(parts) and parts[0] in FORBIDDEN_PACKAGES


def find_violations(source: str, filename: str = "<src>") -> list[str]:
    """Devuelve las violaciones de la frontera GT encontradas en un archivo fuente."""
    tree = ast.parse(source, filename=filename)
    found: list[str] = []

    def report(node: ast.AST, msg: str) -> None:
        found.append(f"{filename}:{getattr(node, 'lineno', '?')}: {msg}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _forbidden_module(alias.name):
                    report(node, f"import prohibido {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if _forbidden_module(node.module):
                report(node, f"import prohibido desde {node.module!r}")
            elif node.module in (None, "src"):
                for alias in node.names:
                    if alias.name in FORBIDDEN_PACKAGES:
                        report(node, f"import prohibido de {alias.name!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            first = node.args[0] if node.args else None
            arg = first.value if isinstance(first, ast.Constant) else None
            if not isinstance(arg, str):
                continue
            if fname in {"import_module", "__import__"} and _forbidden_module(arg):
                report(node, f"import dinámico prohibido {arg!r}")
            if fname == "get" and arg in FORBIDDEN_KEYS:
                report(node, f"acceso a clave de GT/eval {arg!r}")
        elif isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value in FORBIDDEN_KEYS:
                report(node, f"acceso a clave de GT/eval {key.value!r}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if ANNOTATION_FILE_RE.search(node.value):
                report(node, f"referencia a archivo de anotaciones {node.value!r}")
    return found


def scan_package(directory: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        violations += find_violations(source, str(path.relative_to(directory.parent)))
    return violations


def test_scene_no_cruza_la_frontera() -> None:
    assert SCENE_DIR.is_dir()
    assert scan_package(SCENE_DIR) == []


def test_importar_scene_no_carga_eval_ni_eda() -> None:
    """Chequeo dinámico: importar todos los módulos de `scene` no carga `eval` ni `eda`."""
    code = (
        "import importlib, pkgutil, sys, scene\n"
        "for m in pkgutil.walk_packages(scene.__path__, 'scene.'):\n"
        "    importlib.import_module(m.name)\n"
        "bad = sorted(n for n in sys.modules if n.split('.')[0] in ('eval', 'eda'))\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


@pytest.mark.parametrize(
    "snippet",
    [
        "import eval",
        "import eval.annotations",
        "import eda.plots as p",
        "from eval import annotations",
        "from eval.lvis_federated import evaluate",
        "from src.eval.annotations import load_annotations",
        "from src import eda",
        "import importlib\nimportlib.import_module('eval.oracle')",
        "__import__('eda')",
        "path = 'development/annotations.json'",
        "open('lvis_v1_val.json')",
        "cfg['eval']['annotations']",
        "neg = img['neg_category_ids']",
        "x = meta.get('not_exhaustive_category_ids')",
    ],
)
def test_detecta_violaciones(snippet: str) -> None:
    assert find_violations(snippet), f"no se detectó: {snippet!r}"


@pytest.mark.parametrize(
    "snippet",
    [
        "import numpy as np",
        "from scene.masks import rle",
        "from scene.config import data_path",
        "x = 'evaluation results'",
        "model.eval()",
        "cfg['paths']['categories']",
    ],
)
def test_no_marca_codigo_legitimo(snippet: str) -> None:
    assert find_violations(snippet) == []


def test_falla_si_se_agrega_un_import_prohibido(tmp_path: Path) -> None:
    """Simula un paquete `scene` con un módulo que importa GT: el escaneo lo tiene que marcar."""
    pkg = tmp_path / "scene"
    (pkg / "qa").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "qa" / "executor.py").write_text("from eval.annotations import load_annotations\n")
    violations = scan_package(pkg)
    assert len(violations) == 1 and "executor.py" in violations[0]
