"""06: chronological train/val/test split, temporal features, duration×temporal interactions, safe pre-trip features (route target encoding, pu×time interactions), raw location ids as categorical, log1p target.

The model feature set is SAFE_FEATURES (pre-trip only — no distance/duration/
speed, which are post-trip leakage). The target encodings are fitted on the
TRAIN slice only; unseen ids fall back to the train global mean.
"""

import numpy as np
import pandas as pd
from _common import DATETIME_SRC, RESULTS, SAMPLE_NAME, build_temporal_features

from broadway.features.encodings import fit_target_encoding, transform_target_encoding
from broadway.samples import read_named_sample

TRAIN_VAL_TEST = (0.8, 0.1, 0.1)
TARGET = "fare_amount"
TARGET_LOG = "fare_amount_log"
PU_COL = "pickup_location_id"
DO_COL = "dropoff_location_id"
SMOOTHING = 10
PREPARED_DIR = RESULTS / "prepared"
SPLIT_CSV = RESULTS / "06_prepare_model_data_split.csv"

LOCATION_COLS = (PU_COL, DO_COL)

# Pickup-zone × temporal-flag interactions (pre-trip: zone + time known in
# advance). Built on the train-only pickup target encoding.
PU_INTERACTION_SPECS: tuple[tuple[str, str], ...] = (
    ("pu_rush_interaction", "is_rush_hour"),
    ("pu_night_interaction", "is_night"),
    ("pu_weekend_interaction", "is_weekend"),
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


def _prepare_split(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal features, route id, categorical ids, log1p target."""
    df = build_temporal_features(df)
    df["route_id"] = df[PU_COL].astype(str) + "_" + df[DO_COL].astype(str)
    for col in LOCATION_COLS:
        df[col] = df[col].astype("category")
    df[TARGET_LOG] = np.log1p(df[TARGET])
    return df


def _apply_encodings(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Train-only target encodings (route + pickup) + pu×time interactions.

    Fitted on the TRAIN slice; unseen ids fall back to the train global mean
    (the platform's ``__unknown__`` fallback). The pu interactions multiply
    the pickup encoding by each temporal flag.
    """
    train = splits["train"]
    encodings = {
        "route_id": fit_target_encoding(train, "route_id", TARGET, SMOOTHING),
        PU_COL: fit_target_encoding(train, PU_COL, TARGET, SMOOTHING),
    }
    out = {}
    for name, df in splits.items():
        df = transform_target_encoding(df, "route_id", encodings["route_id"]).rename(
            columns={"route_id_target_enc": "route_id_encoded"}
        )
        df = transform_target_encoding(df, PU_COL, encodings[PU_COL]).rename(
            columns={f"{PU_COL}_target_enc": "pickup_location_id_encoded"}
        )
        for feat, flag in PU_INTERACTION_SPECS:
            df[feat] = df["pickup_location_id_encoded"] * df[flag]
        out[name] = df
    return out


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
    splits = _apply_encodings(splits)
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
