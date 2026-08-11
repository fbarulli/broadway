"""Feature derivation — compute derived columns from source data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import DerivedFeature


def _same_borough(df: pd.DataFrame) -> pd.Series:
    pu = df.get("Borough", pd.Series(dtype=str))
    do_col = "Borough_lookup"
    if do_col not in df.columns and "dropoff_location_id" in df.columns:
        return pd.Series(0, index=df.index)
    do = df.get(do_col, pd.Series(dtype=str))
    return (pu == do).astype(int)


_BUILDERS = {
    "pickup_hour": lambda df, src: pd.to_datetime(df[src]).dt.hour.astype(int),
    "pickup_day_of_week": lambda df, src: pd.to_datetime(df[src]).dt.dayofweek.astype(int),
    "pickup_month": lambda df, src: pd.to_datetime(df[src]).dt.month.astype(int),
    "is_weekend": lambda df, src: pd.to_datetime(df[src]).dt.dayofweek.isin([5, 6]).astype(int),
    "rush_hour": lambda df, src: pd.to_datetime(df[src]).dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int),
    "is_night": lambda df, src: pd.to_datetime(df[src]).dt.hour.isin([0, 1, 2, 3, 4, 5, 21, 22, 23]).astype(int),
    "log_distance": lambda df, src: np.log1p(df[src]),
    "same_borough": lambda df, src: _same_borough(df),
    "price / area": lambda df, src: df[src],  # handled specially by target
}


def build_derived(df: pd.DataFrame, features: list[DerivedFeature], target: str) -> pd.DataFrame:
    result = df.copy()
    for feat in features:
        if feat.func in _BUILDERS and feat.source in df.columns:
            result[feat.name] = _BUILDERS[feat.func](result, feat.source)
        elif feat.func == "price / area" and feat.source in df.columns and target in df.columns:
            result[feat.name] = df[target] / df[feat.source]
    return result
