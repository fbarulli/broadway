"""Shared plotting style: palette, Q-Q style, and a despine helper."""

from __future__ import annotations

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
    return sns.color_palette(palette, n)


def despine(ax: Axes) -> None:
    sns.despine(ax=ax)
