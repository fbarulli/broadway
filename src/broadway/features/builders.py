"""Feature derivation — compute derived columns from source data."""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import DerivedFeature


def build_derived(df: pd.DataFrame, features: list[DerivedFeature], target: str) -> pd.DataFrame:
    result = df.copy()
    for feat in features:
        if feat.func == "price / area" and feat.source in df.columns and target in df.columns:
            result[feat.name] = df[target] / df[feat.source]
    return result
