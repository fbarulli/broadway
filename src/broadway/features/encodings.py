"""Smoothed target encoding and frequency encoding."""

from __future__ import annotations

import pandas as pd


def fit_target_encoding(df: pd.DataFrame, col: str, target: str, smoothing: int) -> dict[str, float]:
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["mean", "count"])
    stats["encoded"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return stats["encoded"].to_dict()


def transform_target_encoding(df: pd.DataFrame, col: str, mapping: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df[f"{col}_target_enc"] = df[col].map(mapping).fillna(mapping.get("__unknown__", 0))
    return df


def fit_frequency_encoding(df: pd.DataFrame, col: str) -> dict[str, float]:
    return (df[col].value_counts(normalize=True)).to_dict()


def transform_frequency_encoding(df: pd.DataFrame, col: str, mapping: dict[str, float], fill: float = 0) -> pd.DataFrame:
    df = df.copy()
    df[f"{col}_freq_enc"] = df[col].map(mapping).fillna(fill)
    return df
