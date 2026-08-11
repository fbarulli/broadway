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
    for col, lookup_path in dataset.lookup_tables.items():
        parts = lookup_path.split(":")
        path = parts[0]
        right_on = parts[1] if len(parts) > 1 else col
        lookup = pd.read_csv(path)
        df = df.merge(lookup, left_on=col, right_on=right_on, how=MERGE_HOW, suffixes=("", "_lookup"))
    return df
