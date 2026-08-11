"""Strategy-based imputation: mean, median, mode."""

from __future__ import annotations

import pandas as pd

STRATEGIES = {"mean", "median", "mode", "drop"}


def impute(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown imputation strategy: {strategy}. valid: {STRATEGIES}")
    if strategy == "drop":
        return df.dropna()
    result = df.copy()
    for col in result.columns:
        if result[col].isna().sum() == 0:
            continue
        if strategy == "mode":
            result[col] = result[col].fillna(result[col].mode().iloc[0] if not result[col].mode().empty else None)
        elif pd.api.types.is_numeric_dtype(result[col]):
            fill = result[col].mean() if strategy == "mean" else result[col].median()
            result[col] = result[col].fillna(fill)
    return result
