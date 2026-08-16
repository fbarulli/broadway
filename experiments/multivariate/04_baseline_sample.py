"""04: baseline sample setup — leak-free target/features + time-based holdout.

Builds the ML baseline sample for EVERY configured population (A = Manhattan-
heavy, B = borough-stratified, C = Manhattan + outer-borough weighting): drops
rows with a missing target, derives pickup_weekday, adds the option-C
sample_weight column, sorts by the config datetime column and splits 80/20
chronologically (test = the future, forecasting-style). Fails loudly on
missing features or an empty split; persists per-sample train/test parquets,
per-sample manifest JSONs, and one tracked split CSV with a row per
population.
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


def add_weights(df: pd.DataFrame, cfg: dict, spec: dict) -> pd.DataFrame:
    """Option C: penalty weight for trips to non-reference boroughs."""
    weighting = spec["weighting"]
    if not weighting["enabled"]:
        return df
    ref = cfg["borough_dummies"]["reference"]
    col = weighting["column"]
    if col not in df.columns:
        raise ValueError(f"weighting column '{col}' not in sample")
    return df.assign(sample_weight=np.where(df[col] == ref, 1.0,
                                            weighting["outer_weight"]))


def time_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on the config datetime column."""
    dt_col = cfg["baseline"]["datetime_column"]
    ordered = df.sort_values(dt_col)
    cutoff = int(len(ordered) * (1 - cfg["baseline"]["test_fraction"]))
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


def build_population(cfg: dict, name: str, spec: dict) -> dict:
    """Build one population's split; returns its manifest."""
    df = load_baseline_sample(cfg, spec)
    df = derive_weekday(df, cfg)
    ensure_features(df, cfg)
    target = cfg["baseline"]["target"]
    df = df.dropna(subset=[target]).copy()
    if df.empty:
        raise ValueError(f"population {name}: sample empty after target dropna")
    df = add_weights(df, cfg, spec)
    train_df, test_df = time_split(df, cfg)

    sample_name = spec["sample_name"]
    train_df.to_parquet(RESULTS / f"{sample_name}_train.parquet")
    test_df.to_parquet(RESULTS / f"{sample_name}_test.parquet")

    dt_col = cfg["baseline"]["datetime_column"]
    manifest = {
        "population": name,
        "sample_name": sample_name,
        "source": spec["source"],
        "target": target,
        "features": cfg["baseline"]["features"],
        "test_fraction": cfg["baseline"]["test_fraction"],
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_start": str(test_df[dt_col].min()),
        "test_end": str(test_df[dt_col].max()),
        "weighted": spec["weighting"]["enabled"],
    }
    out = RESULTS / f"{sample_name}_baseline_split.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"population {name}: train {manifest['n_train']} / test "
          f"{manifest['n_test']} | test period "
          f"{manifest['test_start']} to {manifest['test_end']}")
    return manifest


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)

    manifests = [build_population(cfg, name, spec)
                 for name, spec in cfg["baseline"]["populations"].items()]
    csv = RESULTS / f"{CSV_STEM}_split.csv"
    pd.DataFrame(manifests).to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
