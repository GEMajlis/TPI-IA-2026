import numpy as np
import pytest
from pycocotools import mask as mask_utils

from scene.masks import rle


def _coco_decode(r: dict) -> np.ndarray:
    h, w = r["size"]
    return mask_utils.decode(mask_utils.frPyObjects(r, h, w)).astype(bool)


def _cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    one_px = np.zeros((7, 5), bool)
    one_px[3, 2] = True
    border = np.zeros((6, 9), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    corner_last = np.zeros((4, 4), bool)
    corner_last[-1, -1] = True
    return {
        "random": rng.random((31, 17)) > 0.5,
        "random_sparse": rng.random((40, 60)) > 0.95,
        "empty": np.zeros((10, 12), bool),
        "full": np.ones((10, 12), bool),
        "one_pixel": one_px,
        "border": border,
        "first_pixel": np.pad(np.ones((1, 1), bool), ((0, 4), (0, 3))),
        "last_pixel": corner_last,
        "single_cell": np.ones((1, 1), bool),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_round_trip_exacto(name: str) -> None:
    mask = _cases()[name]
    encoded = rle.encode(mask)
    assert encoded["size"] == list(mask.shape)
    assert sum(encoded["counts"]) == mask.size
    assert np.array_equal(rle.decode(encoded), mask)
    assert rle.area(encoded) == int(mask.sum())


@pytest.mark.parametrize("name", list(_cases()))
def test_equivalente_a_pycocotools(name: str) -> None:
    mask = _cases()[name]
    assert np.array_equal(_coco_decode(rle.encode(mask)), mask)


def test_counts_empiezan_por_ceros() -> None:
    mask = np.zeros((3, 3), bool)
    mask[0, 0] = True
    assert rle.encode(mask)["counts"] == [0, 1, 8]
    assert rle.encode(np.ones((2, 2), bool))["counts"] == [0, 4]
    assert rle.encode(np.zeros((2, 2), bool))["counts"] == [4]


def test_orden_column_major() -> None:
    # Primera fila en 1: column-major la recorre como corridas alternadas de a una.
    mask = np.zeros((2, 3), bool)
    mask[0, :] = True
    assert rle.encode(mask)["counts"] == [0, 1, 1, 1, 1, 1, 1]


def test_ejemplo_de_sample_submission() -> None:
    # Fila test_0001 de sample_submission.csv: un píxel en (0, 0) de una imagen 640x425.
    decoded = rle.decode({"size": [640, 425], "counts": [0, 1, 271999]})
    assert decoded.shape == (640, 425)
    assert decoded.sum() == 1 and decoded[0, 0]


def test_decode_rechaza_counts_inconsistentes() -> None:
    with pytest.raises(ValueError):
        rle.decode({"size": [2, 2], "counts": [1, 2]})
    with pytest.raises(ValueError):
        rle.decode({"size": [2, 2], "counts": [-1, 5]})


def test_encode_rechaza_no_2d() -> None:
    with pytest.raises(ValueError):
        rle.encode(np.zeros((2, 2, 2), bool))
