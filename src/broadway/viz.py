"""Shared plotting style: palette, Q-Q style, and a despine helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns

from broadway.config.viz import load_viz_config

LABEL_FONTSIZE: int = 8
TITLE_FONTSIZE: int = 9
TICK_FONTSIZE: int = 7
SUPTITLE_FONTSIZE: int = 13
GRID_ALPHA: float = 0.3

QQ_SCATTER_SIZE: float = 6.0
QQ_SCATTER_ALPHA: float = 0.55
QQ_SCATTER_EDGE_COLOR: str = "none"
QQ_REF_LINE_COLOR: str = "red"
QQ_REF_LINE_STYLE: str = "--"
QQ_REF_LINE_WIDTH: float = 0.8
QQ_XLABEL: str = "Theoretical quantiles"
QQ_YLABEL: str = "Sample quantiles (z)"


def default_palette() -> str:
    return load_viz_config().palette


def palette_colors(n: int, palette: str | None = None) -> list:
    if palette is None:
        palette = default_palette()
    return sns.color_palette(palette, n + 3)[:n]


def despine(ax: Axes) -> None:
    sns.despine(ax=ax)


def draw_coef_forest(
    ax: Axes,
    coef: pd.Series,
    ci_low: pd.Series,
    ci_high: pd.Series,
    labels: dict | None = None,
    annotate: bool = True,
) -> None:
    """Horizontal coefficient forest plot: points with CI error bars."""
    y = np.arange(len(coef))
    ax.errorbar(
        coef.to_numpy(),
        y,
        xerr=[
            coef.to_numpy() - ci_low.to_numpy(),
            ci_high.to_numpy() - coef.to_numpy(),
        ],
        fmt="o",
        capsize=2,
    )
    ax.axvline(0, color="red", linewidth=1)
    ax.set_yticks(y)
    if labels is None:
        ax.set_yticklabels(coef.index)
    else:
        ax.set_yticklabels([labels.get(name, name) for name in coef.index])
    ax.set_ylim(-0.5, len(coef) - 0.5)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    if annotate:
        for yi, (value, lo, hi) in enumerate(zip(coef, ci_low, ci_high)):
            ax.annotate(
                f"{value:.3f} [{lo:.2f}, {hi:.2f}]",
                xy=(value, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=TICK_FONTSIZE,
            )
