import numpy as np
import pytest

from scene.scene.color import dominant_color, dominant_color_from_config

PALETTE = {
    "black": [20, 20, 20],
    "white": [245, 245, 245],
    "gray": [128, 128, 128],
    "red": [200, 30, 30],
    "orange": [240, 140, 30],
    "yellow": [240, 220, 40],
    "green": [40, 150, 50],
    "blue": [40, 80, 200],
    "purple": [130, 50, 160],
    "pink": [240, 150, 180],
    "brown": [120, 75, 40],
}


def solid(rgb, shape=(40, 40)) -> np.ndarray:
    return np.broadcast_to(np.array(rgb, dtype=np.uint8), (*shape, 3)).copy()


@pytest.mark.parametrize(
    ("rgb", "name"),
    [
        ((210, 25, 25), "red"),
        ((30, 160, 40), "green"),
        ((30, 70, 210), "blue"),
        ((250, 250, 250), "white"),
        ((10, 10, 10), "black"),
        ((125, 125, 125), "gray"),
        ((245, 215, 50), "yellow"),
    ],
)
def test_parche_solido(rgb, name: str) -> None:
    c = dominant_color(solid(rgb), np.ones((40, 40), bool), PALETTE)
    assert c.name == name
    assert c.confidence == 1.0
    assert c.rgb == tuple(rgb)


def test_mezcla_gana_el_color_mayoritario() -> None:
    img = solid((30, 70, 210))
    img[:10] = (210, 25, 25)  # 25 % rojo, 75 % azul
    c = dominant_color(img, np.ones((40, 40), bool), PALETTE)
    assert c.name == "blue"
    assert c.confidence == pytest.approx(0.75)


def test_solo_cuentan_los_pixeles_de_la_mascara() -> None:
    img = solid((30, 70, 210))
    img[10:30, 10:30] = (210, 25, 25)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    assert dominant_color(img, mask, PALETTE).name == "red"


def test_mascara_chica_da_color_desconocido() -> None:
    mask = np.zeros((40, 40), bool)
    mask[0, :5] = True
    assert dominant_color(solid((210, 25, 25)), mask, PALETTE, min_pixels=20) is None


def test_determinista_con_submuestreo() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    mask = np.ones((200, 200), bool)
    a = dominant_color(img, mask, PALETTE, max_pixels=500, seed=3)
    b = dominant_color(img, mask, PALETTE, max_pixels=500, seed=3)
    assert a == b


def test_desde_config(base_cfg: dict) -> None:
    c = dominant_color_from_config(solid((30, 70, 210)), np.ones((40, 40), bool), base_cfg)
    assert c.name == "blue"
    assert set(c.to_dict()) == {"name", "rgb", "confidence", "lab_distance"}
