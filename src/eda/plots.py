"""Gráficos del EDA. Cada función responde una pregunta y devuelve la figura."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from scene.scene.schema import POSITIONS  # noqa: E402

# Paleta: un solo tono para magnitudes, tinta de texto neutra y grilla recesiva.
SERIES = "#2a78d6"
ACCENT = "#eb6834"  # solo para marcar umbrales/decisiones
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eaf2fc", "#2a78d6", "#0d3a73"])


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2)


def _fig(width: float = 7, height: float = 3.6) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    return fig, ax


def _vline(ax: plt.Axes, x: float, label: str) -> None:
    ax.axvline(x, color=ACCENT, linewidth=1.5, linestyle="--")
    ax.annotate(
        label, (x, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
        textcoords="offset points", color=TEXT_2, fontsize=8,
    )  # fmt: skip


def instances_per_image(images: pd.DataFrame, max_detections: int) -> plt.Figure:
    """¿Cuántos objetos hay por imagen? → tope de detecciones."""
    fig, ax = _fig()
    ax.hist(images["n_instances"], bins=30, color=SERIES, edgecolor=SURFACE, linewidth=2)
    _vline(ax, float(images["n_instances"].median()), "mediana")
    _style(ax, "Instancias anotadas por imagen", "instancias", "imágenes")
    ax.text(0.98, 0.9, f"máx = {images['n_instances'].max()} · tope propuesto = {max_detections}",
            transform=ax.transAxes, ha="right", color=TEXT_2, fontsize=9)  # fmt: skip
    return fig


def area_ratio(instances: pd.DataFrame, small_max: float, medium_max: float) -> plt.Figure:
    """¿Qué tan chicos son los objetos? → umbrales de tamaño y necesidad de tiling."""
    fig, ax = _fig()
    vals = np.log10(instances["area_ratio"].clip(lower=1e-6))
    ax.hist(vals, bins=40, color=SERIES, edgecolor=SURFACE, linewidth=2)
    _vline(ax, np.log10(small_max), f"small < {small_max:.2%}")
    _vline(ax, np.log10(medium_max), f"medium < {medium_max:.0%}")
    ax.set_xticks([-5, -4, -3, -2, -1, 0], ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"])
    _style(ax, "Área relativa de los objetos (escala log)", "área / área de la imagen", "objetos")
    return fig


def category_frequency(categories: pd.DataFrame, rare_max: int, common_max: int) -> plt.Figure:
    """¿Hay categorías raras? → bins de frecuencia para el desglose de métricas."""
    fig, ax = _fig()
    counts = categories["n_images_positive"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=SERIES, width=0.8)
    for x, y in counts.items():
        ax.text(x, y + 0.4, str(y), ha="center", color=TEXT_2, fontsize=8)
    ax.axvline(rare_max + 0.5, color=ACCENT, linestyle="--", linewidth=1.5)
    ax.axvline(common_max + 0.5, color=ACCENT, linestyle="--", linewidth=1.5)
    _style(ax, "Imágenes en las que aparece cada categoría", "imágenes", "categorías")
    return fig


def federated_status(images: pd.DataFrame) -> plt.Figure:
    """¿Cuántas categorías por imagen son evaluables? → cuánto pesan los FP."""
    fig, ax = _fig()
    cols = {
        "n_positive": "positivas",
        "n_negative": "negativas",
        "n_not_exhaustive": "no exhaustivas\n(subconj. de positivas)",
        "n_unverified": "no verificadas",
    }
    means = images[list(cols)].mean()
    ax.bar(list(cols.values()), means.values, color=SERIES, width=0.6)
    for i, v in enumerate(means.values):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", color=TEXT, fontsize=9)
    _style(ax, "Estado de las 120 categorías por imagen (promedio)", "", "categorías por imagen")
    return fig


def occlusion(overlaps: pd.DataFrame) -> plt.Figure:
    """¿Cuánto se superponen las instancias? → post-proceso de máscaras y NMS."""
    fig, ax = _fig()
    ax.hist(overlaps["max_covered"], bins=20, range=(0, 1), color=SERIES, edgecolor=SURFACE,
            linewidth=2)  # fmt: skip
    _style(ax, "Fracción máxima de la máscara compartida con otra instancia",
           "fracción cubierta", "instancias")  # fmt: skip
    return fig


def position_grid(instances: pd.DataFrame) -> plt.Figure:
    """¿Dónde caen los centroides? → sanidad de la grilla 3×3."""
    share = instances["position"].value_counts(normalize=True).reindex(POSITIONS, fill_value=0)
    grid = share.to_numpy().reshape(3, 3)
    fig, ax = plt.subplots(figsize=(4.2, 3.8), facecolor=SURFACE)
    ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=grid.max())
    for k, name in enumerate(POSITIONS):
        r, c = divmod(k, 3)
        color = SURFACE if grid[r, c] > 0.6 * grid.max() else TEXT
        ax.text(c, r, f"{name}\n{grid[r, c]:.1%}", ha="center", va="center", color=color,
                fontsize=8)  # fmt: skip
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Centroides por celda 3×3", loc="left", color=TEXT, fontsize=11)
    return fig


def benchmark_types(bench: pd.DataFrame) -> plt.Figure:
    """¿Qué se pregunta en el benchmark? → prioridades de QA."""
    fig, ax = _fig()
    counts = bench["type"].value_counts()
    ax.bar(counts.index, counts.values, color=SERIES, width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", color=TEXT, fontsize=9)
    _style(ax, "Consignas del benchmark por tipo", "", "consignas")
    return fig
