from __future__ import annotations

import inspect

from broadway import viz
from broadway.config.viz import VizConfig, load_viz_config


def test_palette_is_bupu_r() -> None:
    assert viz.default_palette() == "BuPu_r"


def test_palette_comes_from_config() -> None:
    assert viz.default_palette() == load_viz_config().palette


def test_viz_config_loads_values() -> None:
    cfg = load_viz_config()
    assert cfg.palette == "BuPu_r"
    assert cfg.max_features_per_figure == 12
    assert cfg.max_points_per_trace == 10000
    assert cfg.fig_size_per_subplot == 3.0
    assert cfg.dpi == 100
    assert cfg.min_unique_for_qq == 15
    assert cfg.qq_figure == "numeric_qq_{fig_num}.png"
    assert cfg.dist_figure == "numeric_dist_{fig_num}.png"
    assert cfg.describe_figure == "describe.png"
    assert cfg.normality_figure == "normality_qq.png"


def test_viz_config_roundtrip() -> None:
    cfg = load_viz_config()
    assert VizConfig.model_validate(cfg.model_dump()) == cfg


def test_palette_colors_returns_requested_count() -> None:
    assert len(viz.palette_colors(3)) == 3


def test_despine_is_callable() -> None:
    assert callable(viz.despine)
    assert "ax" in inspect.signature(viz.despine).parameters
