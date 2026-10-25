import json
import random
from datetime import datetime

import numpy as np
import pytest
import yaml

from scene.config import (
    apply_dotlist,
    config_hash,
    deep_merge,
    load_config,
    make_run_id,
    set_seed,
    start_run,
)
from tests.conftest import BASE_CONFIG


def test_deep_merge_mezcla_dicts_y_reemplaza_listas() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2]}
    override = {"nested": {"y": 20, "z": 30}, "lst": [9]}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 20, "z": 30}, "lst": [9]}
    assert base["nested"] == {"x": 1, "y": 2}  # no muta la base


def test_load_config_con_override_de_experimento(tmp_path) -> None:
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"seed": 7, "paths": {"outputs": "otra/carpeta"}}))
    cfg = load_config(BASE_CONFIG, exp)
    assert cfg["seed"] == 7
    assert cfg["paths"]["outputs"] == "otra/carpeta"
    assert cfg["paths"]["data_root"] == "data/dataset"  # el resto se conserva


def test_dotlist_parsea_tipos() -> None:
    cfg = apply_dotlist({"det": {"thr": 0.1}}, ["det.thr=0.35", "det.nms=true", "new.key=[1, 2]"])
    assert cfg == {"det": {"thr": 0.35, "nms": True}, "new": {"key": [1, 2]}}
    with pytest.raises(ValueError):
        apply_dotlist({}, ["sin_igual"])


def test_misma_config_mismo_hash() -> None:
    a = load_config(BASE_CONFIG)
    b = load_config(BASE_CONFIG)
    assert config_hash(a) == config_hash(b)
    # El orden de las claves no cambia el hash; un valor distinto sí.
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1})
    assert config_hash(a) != config_hash(apply_dotlist(a, ["seed=1"]))


def test_run_id_timestamp_y_hash() -> None:
    cfg = {"seed": 1}
    run_id = make_run_id(cfg, now=datetime(2026, 10, 18, 12, 30, 5))
    assert run_id == f"20261018-123005-{config_hash(cfg)[:8]}"


def test_set_seed_reproducible() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand())
    set_seed(123)
    assert (random.random(), np.random.rand()) == first


def test_start_run_crea_carpeta_y_manifest(base_cfg) -> None:
    ctx = start_run(base_cfg, run_id="test-run")
    assert ctx.run_dir.is_dir()
    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "test-run"
    assert manifest["config_hash"] == config_hash(base_cfg)
    assert manifest["config"] == base_cfg
    assert manifest["seed"] == base_cfg["seed"]
    assert manifest["packages"]["numpy"] == np.__version__
    assert manifest["hardware"]["device"] in {"cpu", "cuda"}
    ctx.update_manifest("models", {"detector": "google/owlv2-base-patch16-ensemble"})
    assert json.loads(ctx.manifest_path.read_text(encoding="utf-8"))["models"][
        "detector"
    ].startswith("google/")
