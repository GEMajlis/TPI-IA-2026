"""Carga de configuración, semillas, run_id y manifest de cada corrida."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Librerías cuya versión se registra en el manifest (si están instaladas).
TRACKED_PACKAGES = (
    "numpy",
    "torch",
    "transformers",
    "pycocotools",
    "opencv-python",
    "pydantic",
    "jsonschema",
    "scikit-learn",
    "pandas",
    "pillow",
)


def deep_merge(base: dict, override: dict) -> dict:
    """Combina dos dicts: los anidados se mezclan y el resto (listas incluidas) se reemplaza."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_dotlist(cfg: dict, dotlist: list[str]) -> dict:
    """Aplica overrides tipo `detection.threshold=0.3`. El valor se parsea como YAML."""
    out = copy.deepcopy(cfg)
    for item in dotlist:
        if "=" not in item:
            raise ValueError(f"Override inválido (se espera clave=valor): {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = out
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"No se puede asignar dentro de un valor no-dict: {dotted!r}")
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def load_config(*paths: str | Path, dotlist: list[str] | None = None) -> dict:
    """Carga uno o más YAML y los combina de izquierda a derecha (base primero)."""
    if not paths:
        raise ValueError("Se necesita al menos un archivo de configuración")
    cfg: dict = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: la raíz del YAML debe ser un mapeo")
        cfg = deep_merge(cfg, data)
    return apply_dotlist(cfg, dotlist or [])


def config_hash(cfg: dict) -> str:
    """Hash SHA-256 de la config canónica (claves ordenadas): misma config → mismo hash."""
    canonical = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_run_id(cfg: dict, now: datetime | None = None) -> str:
    """run_id = timestamp + hash corto de la config."""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{config_hash(cfg)[:8]}"


def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch (si está instalado)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> str:
    """Traduce `auto` a `cuda` si hay GPU disponible; si no, `cpu`."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def data_path(cfg: dict, key: str) -> Path:
    """Ruta absoluta de `paths.<key>` resuelta contra `paths.data_root`."""
    paths = cfg["paths"]
    return Path(paths["data_root"]) / paths[key]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "gpu": None,
    }
    if device.startswith("cuda"):
        import torch

        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": props.name, "total_memory_mb": props.total_memory // 2**20}
    return info


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class RunContext:
    """Datos de una corrida: id, config y carpeta de salida."""

    run_id: str
    config: dict
    config_hash: str
    run_dir: Path
    device: str

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / self.config["tracing"]["manifest_file"]

    @property
    def trace_path(self) -> Path:
        return self.run_dir / self.config["tracing"]["trace_file"]

    def update_manifest(self, key: str, value: Any) -> None:
        """Agrega o reemplaza una clave del manifest (por ejemplo, los modelos usados)."""
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest[key] = value
        _write_json(self.manifest_path, manifest)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def start_run(cfg: dict, run_id: str | None = None) -> RunContext:
    """Fija la semilla, crea `outputs/runs/<run_id>/` y escribe `run_manifest.json`."""
    set_seed(cfg["seed"])
    if run_id is None:
        # Dos runs con la misma config en el mismo segundo: se agrega un sufijo.
        base_id = run_id = make_run_id(cfg)
        n = 1
        while (Path(cfg["paths"]["outputs"]) / run_id).exists():
            run_id, n = f"{base_id}-{n}", n + 1
    run_dir = Path(cfg["paths"]["outputs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(cfg.get("device", "auto"))
    ctx = RunContext(
        run_id=run_id, config=cfg, config_hash=config_hash(cfg), run_dir=run_dir, device=device
    )
    manifest = {
        "run_id": run_id,
        "config_hash": ctx.config_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "config": cfg,
        "python": sys.version,
        "packages": _package_versions(),
        "hardware": _hardware_info(device),
        "git_commit": _git_commit(),
        "models": {},
    }
    _write_json(ctx.manifest_path, manifest)
    return ctx
