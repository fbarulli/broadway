from __future__ import annotations

import inspect

from broadway import viz


def test_palette_is_bupu_r() -> None:
    assert viz.PALETTE == "BuPu_r"


def test_palette_colors_returns_requested_count() -> None:
    assert len(viz.palette_colors(3)) == 3


def test_despine_is_callable() -> None:
    assert callable(viz.despine)
    assert "ax" in inspect.signature(viz.despine).parameters
