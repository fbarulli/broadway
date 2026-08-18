"""06: chronological train/val/test split, temporal features, duration×temporal interactions, raw location ids as categorical for LightGBM, log1p target."""

import numpy as np
import pandas as pd
from _common import DATETIME_SRC, RESULTS, SAMPLE_NAME, build_temporal_features

from broadway.samples import read_named_sample

TRAIN_VAL_TEST = (0.8, 0.1, 0.1)
TARGET = "fare_amount"
TARGET_LOG = "fare_amount_log"
PU_COL = "pickup_location_id"
DO_COL = "dropoff_location_id"
PREPARED_DIR = RESULTS / "prepared"
SPLIT_CSV = RESULTS / "06_prepare_model_data_split.csv"

LOCATION_COLS = (PU_COL, DO_COL)

# Temporal compounding: individual hour markers carry ~0 information; they
# matter combined with trip duration, so each flag enters as a product term.
INTERACTION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("duration_rush", "trip_duration_minutes", "is_rush_hour"),
    ("duration_weekend", "trip_duration_minutes", "is_weekend"),
    ("duration_night", "trip_duration_minutes", "is_night"),
)


def time_split(
    df: pd.DataFrame, fractions: tuple[float, float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split: train = earliest, test = latest."""
    ordered = df.sort_values(DATETIME_SRC).reset_index(drop=True)
    n = len(ordered)
    n_val = int(n * fractions[1])
    n_test = int(n * fractions[2])
    train = ordered.iloc[: n - n_val - n_test]
    val = ordered.iloc[n - n_val - n_test : n - n_test]
    test = ordered.iloc[n - n_test :]
    if any(part.empty for part in (train, val, test)):
        raise ValueError(f"empty split for fractions {fractions} over {n} rows")
    return train, val, test


def _interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration × temporal-flag product terms (float64) to ``df``."""
    for name, base, flag in INTERACTION_SPECS:
        df[name] = df[base] * df[flag]
    return df


def _prepare_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal features, duration×temporal interactions, categorical ids, log1p target."""
    df = build_temporal_features(df)
    df = _interactions(df)
    for col in LOCATION_COLS:
        df[col] = df[col].astype("category")
    df[TARGET_LOG] = np.log1p(df[TARGET])
    return df


def _evidence_rows(splits: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    """One evidence row per split: size and date range."""
    return [
        {
            "split": name,
            "n_rows": len(df),
            "date_min": str(df[DATETIME_SRC].min()),
            "date_max": str(df[DATETIME_SRC].max()),
        }
        for name, df in splits.items()
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    train, val, test = time_split(sample.df, TRAIN_VAL_TEST)

    splits = {
        "train": _prepare_split(train),
        "val": _prepare_split(val),
        "test": _prepare_split(test),
    }
    for name, df in splits.items():
        out = PREPARED_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}")

    evidence = pd.DataFrame(_evidence_rows({"train": train, "val": val, "test": test}))
    evidence.to_csv(SPLIT_CSV, index=False)
    print(f"wrote {SPLIT_CSV}")
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
