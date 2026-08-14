"""Shared plotting style: one palette and a despine helper."""

from __future__ import annotations

import seaborn as sns

PALETTE = "BuPu_r"


def palette_colors(n: int) -> list:
    return sns.color_palette(PALETTE, n)


def despine(ax) -> None:
    sns.despine(ax=ax)
