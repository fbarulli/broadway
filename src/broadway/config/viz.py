"""Visualization knobs sourced from ``configs/step/viz.yaml``."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict

from broadway.config import loader


class QqZonesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    central_quantiles: list[float]
    tail_threshold: float
    zero_mass_threshold: float
    central_alpha: float
    tail_alpha: float
    zone_color: str
    shelf_color: str


class VizConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    palette: str
    max_features_per_figure: int
    max_points_per_trace: int
    fig_size_per_subplot: float
    dpi: int
    min_unique_for_qq: int
    qq_sample_size: int
    qq_random_state: int
    qq_figure: str
    dist_figure: str
    describe_figure: str
    normality_figure: str
    qq_zones: QqZonesConfig


def load_viz_config() -> VizConfig:
    path = loader.CONFIGS_DIR / "step" / "viz.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return VizConfig(**data)
