"""Shared utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.schemas import schema_columns

# Decision 5 pin: numeric-category matching over the OBSERVED runtime dtypes
# ({int32, int64, float64}); object and datetime64 are excluded by the assertion.
_NUMERIC_CATEGORY_DTYPES = frozenset({"int32", "int64", "float64"})


def eligible_feature_columns(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Eligible model-input columns: declared feature surface ∩ frame − target.

    The declared surface comes from ``data_source.schema_contract`` (for
    ``engineered``, resolved through the features config). A categorical
    eligible column not claimed by a preprocessing step fails loud, naming the
    column (Decision 5 — dtype-driven selection is retired). Frame column
    order is preserved, NOT sorted.
    """
    if cfg.experiment is None or cfg.dataset is None:
        raise ValueError("eligible_feature_columns requires experiment and dataset config")
    declared = schema_columns(
        cfg.experiment.data_source.schema_contract,
        cfg.dataset,
        features=cfg.experiment.features,
    )
    claimed = {col for step in cfg.experiment.preprocessing for col in step.columns}
    eligible = [name for name in df.columns if name in declared and name != cfg.dataset.target]
    for name in eligible:
        if name in claimed or str(df[name].dtype) in _NUMERIC_CATEGORY_DTYPES:
            continue
        raise ValueError(
            f"categorical column '{name}' has no preprocessing step and no "
            "numeric-selector fallback applies — add a preprocessing step claiming "
            "it, or repoint the schema contract so it is not eligible"
        )
    return df[eligible]


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
