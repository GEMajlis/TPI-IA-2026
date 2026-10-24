"""Benchmark reproducible de los candidatos LLM. Ejecutar en una GPU T4."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from datetime import datetime
from pathlib import Path

from scene.config import load_config, resolve_device
from scene.llm.model import benchmark_llm, build_llm

PROMPTS = [
    "Convertí a JSON DSL: ¿Cuántas tazas aparecen?",
    "Convertí a JSON DSL: ¿La lámpara está por encima de la silla?",
    "Describí brevemente: Total de objetos: 2; cup: 1; table: 1.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--candidate", action="append", help="Candidato; por defecto, todos")
    parser.add_argument("--output", type=Path, default=Path("outputs/llm_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    device = resolve_device(cfg.get("device", "auto"))
    if not device.startswith("cuda"):
        raise SystemExit("Este benchmark debe ejecutarse en una GPU CUDA (objetivo: T4)")

    candidates = args.candidate or list(cfg["llm"]["candidates"])
    benchmark_cfg = cfg["llm"]["benchmark"]
    results = []
    for candidate in candidates:
        model = build_llm(cfg, device=device, candidate=candidate)
        try:
            result = benchmark_llm(
                model,
                PROMPTS,
                candidate=candidate,
                warmup=benchmark_cfg["warmup"],
                repeats=benchmark_cfg["repeats"],
            )
            results.append(result.to_dict())
        finally:
            model.close()
            gc.collect()

    import torch

    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "prompts": len(PROMPTS),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
