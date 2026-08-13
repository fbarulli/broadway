"""Feature derivation — compute derived columns from source data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import DerivedFeature


def _same_borough(df: pd.DataFrame, borough_col: str, lookup_col: str) -> pd.Series:
    if borough_col not in df.columns:
        raise ValueError(f"same_borough requires column '{borough_col}'")
    if lookup_col not in df.columns:
        raise ValueError(f"same_borough requires column '{lookup_col}'")
    pu = df[borough_col]
    do = df[lookup_col]
    return (pu == do).astype(int)


def _rush_hour(df: pd.DataFrame, source: str, hours: list[int]) -> pd.Series:
    return pd.to_datetime(df[source]).dt.hour.isin(hours).astype(int)


_BUILDERS = {
    "pickup_hour": lambda df, src, **kw: pd.to_datetime(df[src]).dt.hour.astype(int),
    "pickup_day_of_week": lambda df, src, **kw: pd.to_datetime(df[src]).dt.dayofweek.astype(int),
    "pickup_month": lambda df, src, **kw: pd.to_datetime(df[src]).dt.month.astype(int),
    "is_weekend": lambda df, src, **kw: pd.to_datetime(df[src]).dt.dayofweek.isin([5, 6]).astype(int),
    "rush_hour": lambda df, src, **kw: _rush_hour(df, src, kw.get("rush_hour_hours", [7, 8, 9, 17, 18, 19])),
    "is_night": lambda df, src, **kw: pd.to_datetime(df[src]).dt.hour.isin([0, 1, 2, 3, 4, 5, 21, 22, 23]).astype(int),
    "log_distance": lambda df, src, **kw: np.log1p(df[src]),
    "same_borough": lambda df, src, **kw: _same_borough(df, kw.get("borough_col", "Borough"), kw.get("lookup_col", "Borough_lookup")),
    "price / area": lambda df, src, **kw: df[src],
}


def build_derived(df: pd.DataFrame, features: list[DerivedFeature], target: str,
                  borough_col: str = "Borough",
                  lookup_col: str = "Borough_lookup",
                  rush_hour_hours: list[int] | None = None) -> pd.DataFrame:
    _rush_hours = rush_hour_hours if rush_hour_hours is not None else [7, 8, 9, 17, 18, 19]
    builder_kwargs = {"borough_col": borough_col, "lookup_col": lookup_col, "rush_hour_hours": _rush_hours}
    result = df.copy()
    for feat in features:
        if feat.func not in _BUILDERS:
            raise ValueError(f"unknown builder function '{feat.func}' for derived feature '{feat.name}'")
        if feat.source not in df.columns:
            raise ValueError(f"derived feature '{feat.name}' references missing source column '{feat.source}'")
        result[feat.name] = _BUILDERS[feat.func](result, feat.source, **builder_kwargs)
    return result
