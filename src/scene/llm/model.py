"""Carga y medición reproducible de modelos de lenguaje de Hugging Face."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from scene.config import resolve_device, set_seed


class TextGenerator(Protocol):
    """Contrato mínimo usado por el intérprete y el descriptor."""

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Genera texto de manera determinista para un prompt."""


@dataclass(frozen=True)
class GenerationMetrics:
    """Mediciones de una generación individual."""

    latency_ms: float
    input_tokens: int
    output_tokens: int
    peak_memory_mb: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    """Resumen serializable del benchmark de un candidato."""

    candidate: str
    model_id: str
    precision: str
    device: str
    load_peak_memory_mb: float | None
    latency_mean_ms: float
    latency_p95_ms: float
    output_tokens_mean: float
    generation_peak_memory_mb: float | None
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransformersLLM:
    """Wrapper lazy de un modelo causal con chat template."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        precision: str,
        seed: int,
        max_new_tokens: int,
        trust_remote_code: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if precision not in {"fp16", "4bit"}:
            raise ValueError("llm.precision debe ser 'fp16' o '4bit'")
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.last_metrics: GenerationMetrics | None = None
        self.load_peak_memory_mb: float | None = None
        self._torch: Any | None = None

    def load(self) -> TransformersLLM:
        """Descarga/carga el modelo. En 4 bits requiere CUDA y `bitsandbytes`."""
        if self.model is not None:
            return self

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.precision == "4bit" and not self.device.startswith("cuda"):
            raise RuntimeError("La cuantización 4-bit requiere un dispositivo CUDA")

        set_seed(self.seed)
        self._torch = torch
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.precision == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - depende del entorno Kaggle
                raise RuntimeError("4-bit requiere transformers con BitsAndBytesConfig") from exc
            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": {"": self.device},
                }
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        if self.precision == "fp16":
            self.model.to(self.device)
        self.model.eval()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            self.load_peak_memory_mb = torch.cuda.max_memory_allocated() / 2**20
        return self

    def _formatted_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Genera con greedy decoding (`do_sample=False`) y registra costo."""
        self.load()
        torch = self._torch
        set_seed(self.seed)
        formatted = self._formatted_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        input_tokens = int(inputs["input_ids"].shape[-1])
        generated = output[0, input_tokens:]
        peak = torch.cuda.max_memory_allocated() / 2**20 if self.device.startswith("cuda") else None
        self.last_metrics = GenerationMetrics(
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=int(generated.shape[-1]),
            peak_memory_mb=peak,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        """Libera referencias y la caché CUDA entre candidatos del benchmark."""
        self.model = None
        self.tokenizer = None
        if self._torch is not None and self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def build_llm(
    cfg: dict, *, device: str | None = None, candidate: str | None = None
) -> TransformersLLM:
    """Construye el candidato configurado sin cargar todavía sus pesos."""
    llm_cfg = cfg["llm"]
    candidate = candidate or llm_cfg["selected"]
    if candidate not in llm_cfg["candidates"]:
        raise ValueError(f"Candidato LLM desconocido: {candidate!r}")
    candidate_cfg = llm_cfg["candidates"][candidate]
    return TransformersLLM(
        candidate_cfg["model_id"],
        device=device or resolve_device(cfg.get("device", "auto")),
        precision=candidate_cfg["precision"],
        seed=llm_cfg.get("seed", cfg["seed"]),
        max_new_tokens=llm_cfg["max_new_tokens"],
        trust_remote_code=llm_cfg["trust_remote_code"],
        chat_template_kwargs=candidate_cfg.get("chat_template_kwargs"),
    )


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def benchmark_llm(
    model: TransformersLLM,
    prompts: list[str],
    *,
    candidate: str,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    """Mide latencia y memoria. El llamador debe ejecutarlo en una T4."""
    if not prompts:
        raise ValueError("El benchmark necesita al menos un prompt")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup debe ser >= 0 y repeats >= 1")
    model.load()
    for index in range(warmup):
        model.generate(prompts[index % len(prompts)])

    samples: list[GenerationMetrics] = []
    for index in range(repeats):
        model.generate(prompts[index % len(prompts)])
        if model.last_metrics is None:  # pragma: no cover - protege el contrato interno
            raise RuntimeError("El generador no registró las métricas")
        samples.append(model.last_metrics)

    peaks = [sample.peak_memory_mb for sample in samples if sample.peak_memory_mb is not None]
    latencies = [sample.latency_ms for sample in samples]
    return BenchmarkResult(
        candidate=candidate,
        model_id=model.model_id,
        precision=model.precision,
        device=model.device,
        load_peak_memory_mb=model.load_peak_memory_mb,
        latency_mean_ms=statistics.fmean(latencies),
        latency_p95_ms=_percentile_95(latencies),
        output_tokens_mean=statistics.fmean(sample.output_tokens for sample in samples),
        generation_peak_memory_mb=max(peaks) if peaks else None,
        samples=len(samples),
    )
