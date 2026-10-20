"""Trazas por (imagen, etapa) en `trace.jsonl`.

Uso:
    tracer = Tracer(ctx.trace_path, run_id=ctx.run_id, config_hash=ctx.config_hash)
    for img in images:
        with tracer.image_guard(img.image_id):          # un error no corta el run
            with tracer.stage(img.image_id, "detection", params=p, n_in=1) as s:
                dets = detector.detect(img)
                s.n_out = len(dets)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageRecord:
    """Una línea de la traza. La etapa completa `n_out`, `artifact_path` y `extra`."""

    run_id: str
    config_hash: str
    image_id: str
    stage: str
    model_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    t_start: float = 0.0  # epoch en segundos
    t_end: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # ok | cached | error
    n_in: int | None = None
    n_out: int | None = None
    artifact_path: str | None = None
    error: str | None = None
    traceback: str | None = None
    gpu_mem_peak_mb: float | None = None
    # Datos propios de la etapa (por ejemplo, cuántas detecciones elimina cada filtro).
    extra: dict[str, Any] = field(default_factory=dict)


def _cuda_torch():
    """Devuelve torch solo si ya está importado y hay CUDA (el tracer no importa torch)."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return torch
    return None


class Tracer:
    """Escribe una línea JSONL por etapa ejecutada."""

    def __init__(self, path: str | Path, run_id: str, config_hash: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config_hash = config_hash
        self.failed_images: list[str] = []

    def _write(self, record: StageRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @contextmanager
    def stage(
        self,
        image_id: str,
        stage: str,
        params: dict[str, Any] | None = None,
        model_id: str | None = None,
        n_in: int | None = None,
    ) -> Iterator[StageRecord]:
        """Mide y registra una etapa. Si falla, la registra con `status=error` y relanza."""
        rec = StageRecord(
            run_id=self.run_id,
            config_hash=self.config_hash,
            image_id=image_id,
            stage=stage,
            model_id=model_id,
            params=dict(params or {}),
            n_in=n_in,
        )
        torch = _cuda_torch()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        rec.t_start = time.time()
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec.status = "error"
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.traceback = traceback.format_exc()
            exc._traced = True  # para que image_guard no la registre dos veces
            raise
        finally:
            rec.duration_ms = (time.perf_counter() - t0) * 1000
            rec.t_end = time.time()
            if torch is not None:
                rec.gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / 2**20
            self._write(rec)

    @contextmanager
    def image_guard(self, image_id: str) -> Iterator[None]:
        """Atrapa cualquier error de una imagen para que el run siga con la siguiente."""
        try:
            yield
        except Exception as exc:
            self.failed_images.append(image_id)
            if not getattr(exc, "_traced", False):
                # Error fuera de una etapa: se registra igual para no perderlo.
                self._write(
                    StageRecord(
                        run_id=self.run_id,
                        config_hash=self.config_hash,
                        image_id=image_id,
                        stage="image",
                        status="error",
                        t_start=time.time(),
                        t_end=time.time(),
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                )


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Lee `trace.jsonl` como lista de dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
