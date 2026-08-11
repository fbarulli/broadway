"""Filter invalid rows, drop duplicates, enforce minimum row count."""

from __future__ import annotations

import logging

import pandas as pd

from broadway.config.schema import DatasetContract

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, dataset: DatasetContract) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[dataset.target])
    target_dropped = before - len(df)
    before = len(df)
    df = df.drop_duplicates()
    dup_dropped = before - len(df)
    if target_dropped:
        logger.info(f"dropped {target_dropped} rows with null target")
    if dup_dropped:
        logger.info(f"dropped {dup_dropped} duplicate rows")
    return df
