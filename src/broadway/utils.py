"""Shared utilities."""

from __future__ import annotations

import pandas as pd


def feature_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
