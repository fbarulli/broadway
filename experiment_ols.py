"""Experiment: capped-stratified OLS baseline (Manhattan) vs incremental borough pooling.

Samples the full training parquet down to a per-borough cap (the lowest kept
borough count), then fits duration ~ trip_distance, starting Manhattan-only and
adding boroughs one at a time, saving residual Q-Q plots at each step.
"""

from pathlib import Path

import polars as pl
import pandas as pd

from broadway.stats import diagnostics, regression

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_ols"

TRAINING = ROOT / "data" / "processed" / "training_data.parquet"
LOOKUP = ROOT / "data" / "raw" / "taxi_zone_lookup.csv"

KEEP = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
TARGET = "trip_duration_minutes"
DISTANCE = "trip_distance"
BOROUGH = "pickup_borough"
SEED = 42


def build_capped_sample(cap: int | None = None) -> pd.DataFrame:
    zones = pl.read_csv(LOOKUP).select([pl.col("LocationID"), pl.col("Borough")]).lazy()
    full = (
        pl.scan_parquet(TRAINING)
        .select([pl.col("pickup_location_id"), pl.col(DISTANCE), pl.col(TARGET)])
        .join(zones, left_on="pickup_location_id", right_on="LocationID", how="left")
        .filter(pl.col("Borough").is_in(KEEP))
        .collect()
    )
    if cap is None:
        cap = int(full.group_by("Borough").len().select(pl.col("len").min()).item())
    parts = []
    for b in KEEP:
        g = full.filter(pl.col("Borough") == b)
        parts.append(g.sample(n=min(cap, g.height), seed=SEED))
    return (
        pl.concat(parts)
        .select([pl.col("Borough").alias(BOROUGH), pl.col(DISTANCE), pl.col(TARGET)])
        .to_pandas()
    )


def fit_and_plot(df: pd.DataFrame, out_dir: Path, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = regression.fit_ols(df, f"{TARGET} ~ {DISTANCE}")
    diagnostics.plot_residuals(model, str(out_dir / "residuals.png"))
    result = regression.bp_jb(model)
    dw = diagnostics.durbin_watson(model.resid)
    row = {
        "step": label,
        "n": int(len(df)),
        "r2": round(float(model.rsquared), 4),
        "jb_skew": round(result["skew"], 3),
        "jb_kurtosis": round(result["kurtosis"], 3),
        "jb_p": result["jb_pval"],
        "dw": round(dw, 3),
        "boroughs": sorted(df[BOROUGH].unique().tolist()),
    }
    print(row)
    return row


def main() -> None:
    sample = build_capped_sample()
    print(f"capped stratified sample: {len(sample)} rows")
    print(sample.groupby(BOROUGH).size().to_string())
    steps = [
        ("manhattan", ["Manhattan"]),
        ("plus_queens", ["Manhattan", "Queens"]),
        ("plus_brooklyn", ["Manhattan", "Queens", "Brooklyn"]),
        ("plus_bronx", ["Manhattan", "Queens", "Brooklyn", "Bronx"]),
    ]
    for label, boroughs in steps:
        df = sample[sample[BOROUGH].isin(boroughs)].copy()
        fit_and_plot(df, OUT / label, label)
    print("done")


if __name__ == "__main__":
    main()
