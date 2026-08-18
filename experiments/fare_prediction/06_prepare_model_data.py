"""06: chronological train/val/test split, temporal features, target-encoded locations, log1p target."""

import numpy as np
import pandas as pd
from _common import DATETIME_SRC, RESULTS, SAMPLE_NAME, build_temporal_features

from broadway.features.encodings import fit_target_encoding, transform_target_encoding
from broadway.samples import read_named_sample

TRAIN_VAL_TEST = (0.8, 0.1, 0.1)
TARGET = "fare_amount"
TARGET_LOG = "fare_amount_log"
SMOOTHING = 10
PU_COL = "pickup_location_id"
DO_COL = "dropoff_location_id"
PREPARED_DIR = RESULTS / "prepared"
SPLIT_CSV = RESULTS / "06_prepare_split.csv"

LOCATION_COLS = (PU_COL, DO_COL)


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


def _apply_mapping(df: pd.DataFrame, col: str, mapping: dict[str, float]) -> pd.DataFrame:
    """Apply a train-fitted mapping as ``<col>_encoded`` (platform transform)."""
    return transform_target_encoding(df, col, mapping).rename(
        columns={f"{col}_target_enc": f"{col}_encoded"}
    )


def _prepare_split(df: pd.DataFrame, mappings: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Temporal features, target encodings, categorical ids, and the log1p target."""
    df = build_temporal_features(df)
    for col, mapping in mappings.items():
        df = _apply_mapping(df, col, mapping)
    for col in LOCATION_COLS:
        df[col] = df[col].astype("category")
    df[TARGET_LOG] = np.log1p(df[TARGET])
    return df


def _evidence_rows(
    splits: dict[str, pd.DataFrame], mappings: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    """One evidence row per split: size, date range, rows with an unseen id."""
    rows = []
    for name, df in splits.items():
        unseen = pd.Series(False, index=df.index)
        for col, mapping in mappings.items():
            unseen |= ~df[col].isin(mapping)
        rows.append(
            {
                "split": name,
                "n_rows": len(df),
                "date_min": str(df[DATETIME_SRC].min()),
                "date_max": str(df[DATETIME_SRC].max()),
                "n_unseen_encoded": int(unseen.sum()),
            }
        )
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    train, val, test = time_split(sample.df, TRAIN_VAL_TEST)

    mappings = {
        PU_COL: fit_target_encoding(train, PU_COL, TARGET, SMOOTHING),
        DO_COL: fit_target_encoding(train, DO_COL, TARGET, SMOOTHING),
    }
    print(f"PU ids: {len(mappings[PU_COL])}")
    print(f"DO ids: {len(mappings[DO_COL])}")

    splits = {
        "train": _prepare_split(train, mappings),
        "val": _prepare_split(val, mappings),
        "test": _prepare_split(test, mappings),
    }
    for name, df in splits.items():
        out = PREPARED_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}")

    evidence = pd.DataFrame(
        _evidence_rows({"train": train, "val": val, "test": test}, mappings)
    )
    evidence.to_csv(SPLIT_CSV, index=False)
    print(f"wrote {SPLIT_CSV}")
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
