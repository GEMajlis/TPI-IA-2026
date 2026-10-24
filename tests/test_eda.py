import pytest

from eda.stats import (
    benchmark_table,
    categories_table,
    grid_position,
    images_table,
    instances_table,
    overlap_table,
)
from eval.annotations import GTInstance, ImageGT


def _inst(i, cid, box) -> GTInstance:
    x0, y0, x1, y1 = box
    poly = [[x0, y0, x1, y0, x1, y1, x0, y1]]
    return GTInstance(i, cid, box, poly, float((x1 - x0) * (y1 - y0)))


@pytest.fixture
def gt() -> dict:
    img = ImageGT(
        "tpi_0001",
        "tpi_0001.jpg",
        90,
        90,
        instances=[
            _inst(1, 1, (0, 0, 30, 30)),
            _inst(2, 1, (1, 1, 30, 30)),  # duplicado casi exacto de la 1
            _inst(3, 2, (60, 60, 90, 90)),
        ],
        neg_category_ids={3},
        not_exhaustive_category_ids={2},
    )
    return {img.image_id: img}


@pytest.mark.parametrize(
    ("x", "y", "name"),
    [(0, 0, "top_left"), (45, 45, "center"), (89.9, 0, "top_right"), (90, 90, "bottom_right"),
     (29.9, 30, "middle_left"), (30, 60, "bottom_center")],
)  # fmt: skip
def test_grid_position(x, y, name) -> None:
    assert grid_position(x, y, 90, 90) == name


def test_tablas(gt) -> None:
    inst = instances_table(gt)
    assert list(inst["position"]) == ["top_left", "top_left", "bottom_right"]
    assert inst["area_ratio"].iloc[0] == pytest.approx(900 / 8100)
    img = images_table(gt, n_categories=5).iloc[0]
    assert (img.n_instances, img.n_positive, img.n_negative, img.n_unverified) == (3, 2, 1, 2)
    cats = categories_table(gt, {1: "a", 2: "b", 3: "c"}).set_index("category_id")
    assert cats.loc[1, "n_instances"] == 2 and cats.loc[3, "n_images_negative"] == 1
    assert cats.loc[2, "n_images_not_exh"] == 1


def test_overlap_detecta_duplicados(gt) -> None:
    ov = overlap_table(gt, iou_thr=0.5).set_index("instance_id")
    assert ov.loc[1, "same_class_dup"] and ov.loc[2, "same_class_dup"]
    assert not ov.loc[3, "same_class_dup"] and ov.loc[3, "max_covered"] == 0
    assert ov.loc[2, "max_covered"] == pytest.approx(1.0)  # la 2 queda dentro de la 1


def test_benchmark_estado_federado(gt) -> None:
    qs = [
        {"question_id": "Q1", "image_file": "tpi_0001.jpg", "type": "count", "class_en": ["a"]},
        {"question_id": "Q2", "image_file": "tpi_0001.jpg", "type": "existence", "class_en": ["c"]},
        {"question_id": "Q3", "image_file": "tpi_0001.jpg", "type": "existence", "class_en": ["d"]},
    ]
    bt = benchmark_table(qs, gt, {"a": 1, "b": 2, "c": 3, "d": 4}).set_index("question_id")
    assert (bt.loc["Q1", "status_0"], bt.loc["Q1", "n_gt_0"]) == ("positive", 2)
    assert bt.loc["Q2", "status_0"] == "negative" and bt.loc["Q3", "status_0"] == "unverified"
