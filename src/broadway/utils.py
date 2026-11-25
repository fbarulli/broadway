"""Shared utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.features.plan import ColumnPlan


def eligible_feature_columns(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Eligible model-input columns: the plan's ordered projection.

    The plan resolves the declared feature surface from the experiment config
    (including the categorical-claimed guard) and projects the frame to that
    ordered surface, excluding the target. This public shim keeps the historical
    function name/signature while delegating to the single add/drop path.
    """
    if cfg.experiment is None or cfg.dataset is None:
        raise ValueError("eligible_feature_columns requires experiment and dataset config")
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    return plan.apply(df)[0]


def require_keys(config: dict, keys: list[str], context: str) -> None:
    """Fail loudly when config is missing keys (no silent defaults)."""
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"{context}: config missing required key(s): {missing}")


def require_finite(frame: pd.DataFrame, context: str) -> None:
    """Fail loudly on NaN/Inf — a silent fit on dirty input is worse than an error."""
    if frame.isna().any().any():
        raise ValueError(f"{context}: contains NaN — aborting instead of "
                         "fitting on misaligned/dirty input")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"{context}: contains Inf — aborting instead of "
                         "fitting on misaligned/dirty input")
