import pytest

from scene.tracing.tracer import Tracer, read_trace


def _tracer(tmp_path) -> Tracer:
    return Tracer(tmp_path / "trace.jsonl", run_id="r1", config_hash="abc")


def test_etapa_ok_registra_campos(tmp_path) -> None:
    tracer = _tracer(tmp_path)
    with tracer.stage("img1", "detection", params={"thr": 0.2}, model_id="owlv2", n_in=1) as s:
        s.n_out = 5
        s.artifact_path = "detections/img1.json"
        s.extra["removed_by_nms"] = 3
    (rec,) = read_trace(tracer.path)
    assert rec["run_id"] == "r1" and rec["config_hash"] == "abc"
    assert rec["image_id"] == "img1" and rec["stage"] == "detection"
    assert rec["status"] == "ok" and rec["error"] is None
    assert rec["params"] == {"thr": 0.2} and rec["model_id"] == "owlv2"
    assert (rec["n_in"], rec["n_out"]) == (1, 5)
    assert rec["artifact_path"] == "detections/img1.json"
    assert rec["extra"] == {"removed_by_nms": 3}
    assert rec["t_end"] >= rec["t_start"] and rec["duration_ms"] >= 0


def test_etapa_que_falla_se_registra_y_relanza(tmp_path) -> None:
    tracer = _tracer(tmp_path)
    with pytest.raises(RuntimeError):
        with tracer.stage("img1", "segmentation"):
            raise RuntimeError("sin memoria")
    (rec,) = read_trace(tracer.path)
    assert rec["status"] == "error"
    assert rec["error"] == "RuntimeError: sin memoria"
    assert "Traceback" in rec["traceback"]


def test_el_run_sigue_con_la_imagen_siguiente(tmp_path) -> None:
    tracer = _tracer(tmp_path)
    processed = []
    for image_id in ["a", "b", "c"]:
        with tracer.image_guard(image_id):
            with tracer.stage(image_id, "detection") as s:
                if image_id == "b":
                    raise ValueError("imagen corrupta")
                s.n_out = 1
            processed.append(image_id)
    assert processed == ["a", "c"]
    assert tracer.failed_images == ["b"]
    records = read_trace(tracer.path)
    # Una línea por etapa: el error no se duplica con un registro extra de image_guard.
    assert [(r["image_id"], r["status"]) for r in records] == [
        ("a", "ok"),
        ("b", "error"),
        ("c", "ok"),
    ]


def test_error_fuera_de_etapa_tambien_queda_trazado(tmp_path) -> None:
    tracer = _tracer(tmp_path)
    with tracer.image_guard("x"):
        raise KeyError("falta algo")
    (rec,) = read_trace(tracer.path)
    assert rec["stage"] == "image" and rec["status"] == "error"
    assert tracer.failed_images == ["x"]
