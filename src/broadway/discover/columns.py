"""Read-only column listing: print `name: dtype` per source column."""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import normalize_dtype


def run(csv: str) -> None:
    df = pd.read_csv(csv) if csv.endswith(".csv") else pd.read_parquet(csv)
    for col in df.columns:
        print(f"{col}: {normalize_dtype(str(df[col].dtype))}")
