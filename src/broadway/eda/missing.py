"""Missingness analysis: counts, patterns, Little's MCAR test."""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency


def null_counts(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns}


def null_patterns(df: pd.DataFrame) -> pd.DataFrame:
    null_mask = df.isna()
    patterns = null_mask.groupby(list(df.columns)).size().reset_index(name="count")
    return patterns.sort_values("count", ascending=False)


def littles_mcar_test(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2 or numeric.isna().sum().sum() == 0:
        return {"p_value": 1.0, "is_mcar": True}
    null_indicators = numeric.isna().astype(int)
    p_values = []
    for i in range(numeric.shape[1]):
        for j in range(numeric.shape[1]):
            if i >= j:
                continue
            table = pd.crosstab(null_indicators.iloc[:, i], null_indicators.iloc[:, j])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            _, p, _, _ = chi2_contingency(table)
            p_values.append(p)
    if not p_values:
        return {"p_value": None, "is_mcar": None}
    avg_p = sum(p_values) / len(p_values)
    return {"p_value": round(avg_p, 4), "is_mcar": avg_p > alpha}
