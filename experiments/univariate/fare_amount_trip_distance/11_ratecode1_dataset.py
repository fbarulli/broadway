"""11: build the RatecodeID == 1 dataset (standard metered trips)."""

import polars as pl

from _common import RATECODE1_PARQUET, RAW_DIR, RESULTS

SAMPLE_SIZE = 50_000
SEED = 42
RATE_CODE = 1


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))
    raw = (
        pl.scan_parquet([str(f) for f in files])
        .collect()
        .sample(n=SAMPLE_SIZE, seed=SEED)
    )
    df = raw.filter(
        (pl.col("RatecodeID") == RATE_CODE)
        & (pl.col("fare_amount") >= 0)
        & (pl.col("trip_distance") >= 0)
    )
    df.write_parquet(RATECODE1_PARQUET)
    print(f"RatecodeID == {RATE_CODE} rows: {len(df)}")
    print(f"columns: {df.columns}")
    print(f"wrote {RATECODE1_PARQUET}")


if __name__ == "__main__":
    main()
