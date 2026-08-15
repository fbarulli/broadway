"""06: build a full-feature dataset (all raw columns)."""

import pandas as pd

from _common import FULL_PARQUET, RESULTS
from project.data import read_training_sample

SAMPLE_SIZE = 50_000
MIN_FARE = 2.50
MIN_DISTANCE = 0.0
MAX_DISTANCE = 50.0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["fare_amount"] > MIN_FARE)
        & (df["trip_distance"] > MIN_DISTANCE)
        & (df["trip_distance"] <= MAX_DISTANCE)
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = read_training_sample(sample=SAMPLE_SIZE)
    cleaned = clean(raw)
    cleaned.to_parquet(FULL_PARQUET)
    print(f"full-feature rows: {len(cleaned)}")
    print(f"columns: {list(cleaned.columns)}")
    print(f"wrote {FULL_PARQUET}")


if __name__ == "__main__":
    main()
