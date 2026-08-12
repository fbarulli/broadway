"""Stratified sampling (pandas/numpy only)."""

from __future__ import annotations

import pandas as pd


def stratified_sample(
    df: pd.DataFrame, group_col: str, frac: float, random_state: int
) -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
