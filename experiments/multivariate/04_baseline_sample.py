"""04: baseline sample setup — leak-free target/features + time-based holdout.

Builds the ML baseline sample per config: population A (manhattan_sample) or
B (borough-stratified); target total_amount; features incl. the derived
pickup_weekday. Drops rows with a missing target, adds the option-C
sample_weight column, sorts by the config datetime column and splits 80/20
chronologically (test = the future, forecasting-style). Fails loudly on
missing features or an empty split; persists train/test parquets plus a
manifest JSON and a tracked split CSV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _setup import RESULTS, load_baseline_sample, load_config

CSV_STEM = Path(__file__).stem


def derive_weekday(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the weekday column (0=Monday) when it is a requested feature."""
    if "pickup_weekday" in cfg["baseline"]["features"]:
        return df.assign(pickup_weekday=df[cfg["baseline"]["datetime_column"]].dt.weekday)
    return df


def ensure_features(df: pd.DataFrame, cfg: dict) -> None:
    """Fail loudly when a configured feature is absent from the sample."""
    missing = [f for f in cfg["baseline"]["features"] if f not in df.columns]
    if missing:
        raise ValueError(f"baseline sample missing feature(s): {missing}")


def add_weights(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Option C: penalty weight for non-reference-borough trips."""
    spec = cfg["baseline"]["weighting"]
    if not spec["enabled"]:
        return df
    ref = cfg["borough_dummies"]["reference"]
    col = cfg["borough"]["pickup"]["column"]
    return df.assign(sample_weight=np.where(df[col] == ref, 1.0,
                                            spec["outer_weight"]))


def time_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on the config datetime column."""
    dt_col = cfg["baseline"]["datetime_column"]
    ordered = df.sort_values(dt_col)
    cutoff = int(len(ordered) * (1 - cfg["baseline"]["test_fraction"]))
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_baseline_sample(cfg)
    df = derive_weekday(df, cfg)
    ensure_features(df, cfg)
    target = cfg["baseline"]["target"]
    df = df.dropna(subset=[target]).copy()
    if df.empty:
        raise ValueError("baseline sample empty after target dropna")
    df = add_weights(df, cfg)
    train_df, test_df = time_split(df, cfg)

    dt_col = cfg["baseline"]["datetime_column"]
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"Test period: {test_df[dt_col].min()} to {test_df[dt_col].max()}")

    sample_name = cfg["sample"]["name"]
    train_df.to_parquet(RESULTS / f"{sample_name}_train.parquet")
    test_df.to_parquet(RESULTS / f"{sample_name}_test.parquet")

    manifest = {
        "population": cfg["baseline"]["population"],
        "target": target,
        "features": cfg["baseline"]["features"],
        "datetime_column": dt_col,
        "test_fraction": cfg["baseline"]["test_fraction"],
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_start": str(test_df[dt_col].min()),
        "test_end": str(test_df[dt_col].max()),
        "weighted": cfg["baseline"]["weighting"]["enabled"],
    }
    out = RESULTS / f"{sample_name}_baseline_split.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")

    csv = RESULTS / f"{CSV_STEM}_split.csv"
    pd.DataFrame([manifest]).to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
