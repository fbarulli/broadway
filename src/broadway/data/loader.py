"""Detect format (csv/parquet/excel) → load → optional lookup join."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from broadway.config.schema import DatasetContract

READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}

MERGE_HOW = "left"


def load(dataset: DatasetContract) -> pd.DataFrame:
    path = Path(dataset.path)
    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"unsupported format: {ext}")
    df = READERS[ext](path)
    for col, lookup in dataset.lookup_tables.items():
        right_on = lookup.key
        lookup_df = pd.read_csv(lookup.path)
        df = df.merge(lookup_df, left_on=col, right_on=right_on, how=MERGE_HOW, suffixes=("", "_lookup"))
    return df
