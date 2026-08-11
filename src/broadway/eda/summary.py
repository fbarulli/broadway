"""Distributions, skew, kurtosis, cardinality."""

from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    col_stats = {}
    for col in df.columns:
        col_stats[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
        }
        if col in numeric.columns:
            col_stats[col].update({
                "mean": round(float(df[col].mean()), 2) if not df[col].isna().all() else None,
                "std": round(float(df[col].std()), 2) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "skew": round(float(df[col].skew()), 2) if not df[col].isna().all() else None,
                "kurtosis": round(float(df[col].kurtosis()), 2) if not df[col].isna().all() else None,
            })
    stats["columns"] = col_stats
    return stats
