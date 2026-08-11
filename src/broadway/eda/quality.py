"""Outliers (IQR/Z-score), class imbalance, constant columns, duplicate rows."""

from __future__ import annotations

import pandas as pd


def constant_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def outlier_counts_iqr(df: pd.DataFrame) -> dict[str, int]:
    numeric = df.select_dtypes(include="number")
    counts: dict[str, int] = {}
    for col in numeric.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def class_imbalance(df: pd.DataFrame, target: str) -> dict | None:
    if target not in df.columns or df[target].dtype not in ("object", "category", "bool"):
        return None
    counts = df[target].value_counts().to_dict()
    minority = min(counts.values())
    majority = max(counts.values())
    ratio = minority / majority if majority > 0 else 0
    return {"counts": counts, "minority_ratio": round(ratio, 4)}
