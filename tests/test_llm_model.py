from scene.llm.model import GenerationMetrics, TransformersLLM, benchmark_llm, build_llm


class FakeBenchModel:
    model_id = "fake/model"
    precision = "fp16"
    device = "cuda"
    load_peak_memory_mb = 1000.0

    def __init__(self) -> None:
        self.calls = 0
        self.last_metrics = None

    def load(self):
        return self

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.last_metrics = GenerationMetrics(
            latency_ms=float(self.calls * 10),
            input_tokens=5,
            output_tokens=self.calls,
            peak_memory_mb=1100.0 + self.calls,
        )
        return "ok"


def test_build_llm_es_lazy_y_respeta_candidato(base_cfg) -> None:
    llm = build_llm(base_cfg, device="cpu", candidate="qwen25_3b")
    assert isinstance(llm, TransformersLLM)
    assert llm.model is None
    assert llm.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert llm.precision == "fp16"
    assert llm.seed == 42


def test_build_llm_rechaza_candidato_desconocido(base_cfg) -> None:
    try:
        build_llm(base_cfg, device="cpu", candidate="inexistente")
    except ValueError as exc:
        assert "desconocido" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("debió rechazar el candidato")


def test_precision_invalida_falla_antes_de_descargar() -> None:
    try:
        TransformersLLM(
            "modelo",
            device="cpu",
            precision="int8",
            seed=1,
            max_new_tokens=10,
        )
    except ValueError as exc:
        assert "precision" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("debió rechazar la precisión")


def test_benchmark_descarta_warmup_y_resume_muestras() -> None:
    model = FakeBenchModel()
    result = benchmark_llm(model, ["a", "b"], candidate="fake", warmup=1, repeats=3)
    assert result.samples == 3
    assert result.latency_mean_ms == 30.0
    assert result.latency_p95_ms == 40.0
    assert result.output_tokens_mean == 3.0
    assert result.generation_peak_memory_mb == 1104.0
