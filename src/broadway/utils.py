"""Shared utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.select_dtypes(include="number").drop(columns=[target], errors="ignore")


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
