"""Fixtures compartidas por los tests."""

from pathlib import Path

import pytest

from scene.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "base.yaml"
DATA_ROOT = ROOT / "data" / "dataset"

requires_data = pytest.mark.skipif(
    not (DATA_ROOT / "development").is_dir(), reason="falta la copia local del dataset"
)


@pytest.fixture
def base_cfg(tmp_path: Path) -> dict:
    """Config base con rutas absolutas y salidas en un directorio temporal."""
    return load_config(
        BASE_CONFIG,
        dotlist=[f"paths.data_root={DATA_ROOT}", f"paths.outputs={tmp_path / 'runs'}"],
    )
