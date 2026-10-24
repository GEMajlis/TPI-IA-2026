import pandas as pd
import pytest

from eval.annotations import GTInstance, ImageGT
from eval.tuning import choose, evaluate_postprocess, split_tuning_holdout, sweep
from scene.detection.base import Detection


def gt_image(iid: str, n_instances: int) -> ImageGT:
    insts = [
        GTInstance(k, 1, (10.0 * k, 0.0, 10.0 * k + 8, 8.0), [[0, 0, 1, 0, 1, 1]], 64.0)
        for k in range(n_instances)
    ]
    return ImageGT(iid, f"{iid}.jpg", 200, 100, instances=insts, neg_category_ids={2})


@pytest.fixture
def gt() -> dict[str, ImageGT]:
    return {f"img_{i:03d}": gt_image(f"img_{i:03d}", i % 7 + 1) for i in range(120)}


def test_split_80_40_disjunto_y_determinista(gt) -> None:
    ids = sorted(gt)
    tuning, holdout = split_tuning_holdout(gt, ids, 40, seed=42)
    assert len(tuning) == 80 and len(holdout) == 40
    assert not set(tuning) & set(holdout) and set(tuning) | set(holdout) == set(ids)
    assert split_tuning_holdout(gt, ids, 40, seed=42) == (tuning, holdout)
    assert split_tuning_holdout(gt, ids, 40, seed=1) != (tuning, holdout)


def test_split_estratificado_por_cantidad_de_instancias(gt) -> None:
    tuning, holdout = split_tuning_holdout(gt, sorted(gt), 40, seed=0)
    mean = lambda ids: sum(len(gt[i].instances) for i in ids) / len(ids)  # noqa: E731
    assert abs(mean(tuning) - mean(holdout)) < 0.5


def det(box, score, cid=1) -> Detection:
    return Detection(tuple(map(float, box)), score, cid, "x", "fake", "x")


def test_el_umbral_cambia_precision_y_recall(base_cfg) -> None:
    gt = {"a": gt_image("a", 2)}
    raw = {
        "a": [
            det((0, 0, 8, 8), 0.9),  # TP
            det((10, 0, 18, 8), 0.2),  # TP con score bajo
            det((100, 50, 120, 70), 0.3, cid=2),  # FP (categoría negativa)
        ]
    }
    low = evaluate_postprocess(raw, gt, ["a"], base_cfg, {"score_threshold": 0.1})
    high = evaluate_postprocess(raw, gt, ["a"], base_cfg, {"score_threshold": 0.5})
    assert (low["recall"], low["precision"]) == (1.0, pytest.approx(2 / 3))
    assert (high["recall"], high["precision"]) == (0.5, 1.0)


def test_sweep_y_eleccion_con_empate_toma_el_umbral_mas_alto(base_cfg) -> None:
    gt = {"a": gt_image("a", 1)}
    raw = {"a": [det((0, 0, 8, 8), 0.9)]}
    curve = sweep(raw, gt, {"tuning": ["a"]}, base_cfg, [0.1, 0.3, 0.5], {"base": {}})
    assert len(curve) == 3 and (curve["f1"] == 1.0).all()
    assert choose(curve, "base")["threshold"] == 0.5


def test_choose_maximo() -> None:
    curve = pd.DataFrame(
        {"variant": "v", "split": "tuning", "threshold": [0.1, 0.2, 0.3], "f1": [0.5, 0.7, 0.6]}
    )
    assert choose(curve, "v")["threshold"] == 0.2
