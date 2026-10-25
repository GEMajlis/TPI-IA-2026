"""Color dominante dentro de la máscara: k-means en Lab y mapeo a una paleta con nombre.

La confianza es la proporción de píxeles del cluster dominante. Si la máscara tiene pocos píxeles
el color queda desconocido (`None`) en lugar de inventarlo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class DominantColor:
    name: str  # nombre de la paleta más cercano al centro del cluster dominante
    rgb: tuple[int, int, int]  # centro del cluster dominante
    confidence: float  # proporción de píxeles del cluster dominante
    lab_distance: float  # distancia (CIE76) entre el cluster y el color de la paleta

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rgb"] = list(self.rgb)
        return d


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (N×3) → Lab float (L en [0, 100])."""
    arr = (rgb.reshape(-1, 1, 3).astype(np.float32)) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def nearest_palette_color(lab: np.ndarray, palette: dict[str, list[int]]) -> tuple[str, float]:
    names = list(palette)
    ref = rgb_to_lab(np.array([palette[n] for n in names], dtype=np.uint8))
    dist = np.linalg.norm(ref - lab[None, :], axis=1)
    i = int(dist.argmin())
    return names[i], float(dist[i])


def dominant_color(
    image: np.ndarray,
    mask: np.ndarray,
    palette: dict[str, list[int]],
    k: int = 3,
    max_pixels: int = 5000,
    min_pixels: int = 20,
    seed: int = 0,
) -> DominantColor | None:
    """Color dominante de los píxeles de `mask` en `image` (RGB uint8 HxWx3)."""
    pixels = image[mask]
    if len(pixels) < min_pixels:
        return None
    rng = np.random.default_rng(seed)
    if len(pixels) > max_pixels:
        pixels = pixels[rng.choice(len(pixels), max_pixels, replace=False)]
    lab = rgb_to_lab(pixels)
    n_clusters = min(k, len(np.unique(pixels, axis=0)))
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed).fit(lab)
    counts = np.bincount(km.labels_, minlength=n_clusters)
    top = int(counts.argmax())
    members = pixels[km.labels_ == top]
    name, dist = nearest_palette_color(km.cluster_centers_[top], palette)
    return DominantColor(
        name=name,
        rgb=tuple(int(v) for v in members.mean(axis=0).round()),
        confidence=float(counts[top] / counts.sum()),
        lab_distance=dist,
    )


def dominant_color_from_config(
    image: np.ndarray, mask: np.ndarray, cfg: dict
) -> DominantColor | None:
    c = cfg["scene"]["color"]
    return dominant_color(
        image,
        mask,
        c["palette"],
        k=c["k"],
        max_pixels=c["max_pixels"],
        min_pixels=c["min_pixels"],
        seed=cfg["seed"],
    )
