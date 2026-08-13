"""Filter invalid rows and drop duplicates, returning the DataFrame and drop accounting."""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import DatasetContract


def clean(df: pd.DataFrame, dataset: DatasetContract) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    drops: list[tuple[str, int]] = []
    before = len(df)
    df = df.dropna(subset=[dataset.target])
    if len(df) < before:
        drops.append(("null target", before - len(df)))
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        drops.append(("duplicates", before - len(df)))
    return df, drops
