"""Shared paths, constants, and loaders for timeseries experiments.

Continuity rule: series work needs CONTIGUOUS time coverage. The pinned
``fare_prediction_1m`` sample is a RANDOM draw — fine for cross-sectional
modeling, wrong for series work (bucket counts stay unbiased but small
buckets get noisy and gap analysis is meaningless). The default source here
is therefore the cleaned canonical frame, not the sample.
"""

from pathlib import Path

import pandas as pd

from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
RESULTS = load_project_paths().results / HERE.name

CANONICAL = Path("data/processed/taxi_canonical.parquet")
RAW_MONTHS = {
    "2024-01": Path("data/raw/yellow_tripdata_2024-01.parquet"),
    "2024-02": Path("data/raw/yellow_tripdata_2024-02.parquet"),
    "2024-03": Path("data/raw/yellow_tripdata_2024-03.parquet"),
}
DATETIME_COL = "pickup_datetime"


def load_canonical(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned canonical frame (Jan-Mar 2024, contiguous)."""
    if not CANONICAL.exists():
        raise FileNotFoundError(f"canonical frame not found: {CANONICAL}")
    df = pd.read_parquet(CANONICAL, columns=columns)
    print(f"canonical: {len(df)} rows from {CANONICAL}")
    return df


def load_raw_month(month: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one raw TLC month (provenance checks; uncleaned)."""
    if month not in RAW_MONTHS:
        raise ValueError(f"unknown month {month!r} (known: {sorted(RAW_MONTHS)})")
    path = RAW_MONTHS[month]
    if not path.exists():
        raise FileNotFoundError(f"raw month not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    print(f"raw {month}: {len(df)} rows from {path}")
    return df
