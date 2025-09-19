"""Plotting helpers shared across simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .io import ensure_directory


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 300, close: bool = True) -> str:
    """Persist *fig* to *path* with consistent defaults."""
    dest = Path(path)
    if dest.parent:
        ensure_directory(dest.parent)
    fig.tight_layout()
    fig.savefig(dest, dpi=dpi)
    if close:
        plt.close(fig)
    return str(dest)


def heatmap(
    ax: Axes,
    matrix: np.ndarray,
    *,
    xticklabels: Sequence[str],
    yticklabels: Sequence[str],
    title: str,
    cbar_label: str,
) -> None:
    """Render a heatmap with consistent labelling."""
    im = ax.imshow(matrix, origin='lower', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)


def multi_heatmap(
    matrices: Iterable[tuple[str, np.ndarray]],
    *,
    xticklabels: Sequence[str],
    yticklabels: Sequence[str],
) -> Figure:
    """Build a figure holding one column per matrix."""
    matrices = list(matrices)
    n_cols = len(matrices)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]  # type: ignore[list-item]
    for ax, (title, matrix) in zip(axes, matrices):
        heatmap(ax, matrix, xticklabels=xticklabels, yticklabels=yticklabels, title=title, cbar_label=title)
    return fig
