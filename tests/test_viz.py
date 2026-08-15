from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from broadway import viz
from broadway.config.viz import QqMarkersConfig, VizConfig, load_viz_config


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
    assert cfg.qq_sample_size == 10000
    assert cfg.qq_random_state == 42
    assert cfg.qq_figure == "numeric_qq_{fig_num}.png"
    assert cfg.dist_figure == "numeric_dist_{fig_num}.png"
    assert cfg.describe_figure == "describe.png"
    assert cfg.normality_figure == "normality_qq.png"
    assert cfg.diagnostics.colormap == "coolwarm"
    assert cfg.diagnostics.figure == "numeric_diagnostics.png"
    assert cfg.diagnostics.annotate is True
    assert cfg.diagnostics.thresholds.zero_rate == 0.05
    assert cfg.diagnostics.thresholds.skew == 1.0
    assert cfg.diagnostics.thresholds.kurtosis == 3.0
    assert cfg.diagnostics.thresholds.max_p99_ratio == 10.0


def test_viz_config_roundtrip() -> None:
    cfg = load_viz_config()
    assert VizConfig.model_validate(cfg.model_dump()) == cfg


def test_palette_colors_returns_requested_count() -> None:
    assert len(viz.palette_colors(3)) == 3


def test_despine_is_callable() -> None:
    assert callable(viz.despine)
    assert "ax" in inspect.signature(viz.despine).parameters


def test_qq_markers_config_parses() -> None:
    cfg = load_viz_config()
    markers = cfg.qq_markers
    assert markers.enabled is True
    assert markers.percentile_rings is True
    assert markers.tail_highlight is True
    assert markers.robust_line is True
    assert markers.percentiles == [0.5, 0.9, 0.99, 0.999]
    assert markers.tail_threshold == 3.09
    assert markers.ring_color == "#000000"
    assert markers.ring_size == 4
    assert markers.tail_color == "#d62728"
    assert markers.tail_size == 12
    assert markers.robust_line_color == "#333333"
    assert markers.robust_line_width == 1.2


def test_qq_markers_config_forbids_unknown_key() -> None:
    with pytest.raises(ValidationError):
        QqMarkersConfig(
            enabled=True,
            percentile_rings=True,
            percentiles=[0.5, 0.9, 0.99, 0.999],
            ring_color="#000000",
            ring_size=4,
            tail_highlight=True,
            tail_threshold=3.09,
            tail_color="#d62728",
            tail_size=12,
            robust_line=True,
            robust_line_color="#333333",
            robust_line_width=1.2,
            unknown_key="nope",
        )
