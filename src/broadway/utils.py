"""Shared utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.schemas import schema_columns


def eligible_feature_columns(df: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    """Schema-declared feature columns present in the frame, target excluded (Decision 5).

    Eligible = schema_columns(cfg.experiment.data_source.schema_contract,
    cfg.dataset, features=cfg.experiment.features) ∩ df.columns − {target}, in
    frame column order. Every eligible column must be numeric unless a
    preprocessing step claims it — the dtype-driven selector is retired, so a
    non-numeric unclaimed column fails loud instead of being silently dropped
    (conflict-2 resolution).
    """
    assert cfg.experiment is not None and cfg.dataset is not None
    schema = schema_columns(
        cfg.experiment.data_source.schema_contract,
        cfg.dataset,
        features=cfg.experiment.features,
    )
    claimed = {col for step in cfg.experiment.preprocessing for col in step.columns}
    eligible = [col for col in df.columns if col in schema and col != cfg.dataset.target]
    for col in eligible:
        if col in claimed:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"categorical column '{col}' has no preprocessing step and no "
                "numeric-selector fallback applies — add a preprocessing step "
                "claiming it, or repoint the schema contract so it is not eligible"
            )
    return eligible


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
