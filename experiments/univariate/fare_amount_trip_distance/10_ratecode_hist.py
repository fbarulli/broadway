"""10: histogram of fare_amount for RatecodeID == 2 (JFK flat fare)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from _common import RAW_DIR, RESULTS

OUT = RESULTS / f"{Path(__file__).stem}.png"

SAMPLE_SIZE = 50_000
SEED = 42
MIN_FARE = 2.50
MIN_DISTANCE = 0.0
MAX_DISTANCE = 50.0
RATE_CODE = 2


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("yellow_tripdata_*.parquet"))
    raw = (
        pl.scan_parquet([str(f) for f in files])
        .select(["RatecodeID", "fare_amount", "trip_distance"])
        .collect()
        .sample(n=SAMPLE_SIZE, seed=SEED)
    )
    filtered = raw.filter(
        (pl.col("RatecodeID") == RATE_CODE)
        & (pl.col("fare_amount") > MIN_FARE)
        & (pl.col("trip_distance") > MIN_DISTANCE)
        & (pl.col("trip_distance") <= MAX_DISTANCE)
    )
    values = filtered["fare_amount"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=60, edgecolor="white", linewidth=0.5)
    ax.set_xlim(float(values.min()), float(values.max()))
    ax.set_xlabel("fare_amount ($)")
    ax.set_ylabel("count")
    ax.set_title(f"fare_amount histogram (RatecodeID == {RATE_CODE}, N={len(filtered)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"RatecodeID == {RATE_CODE} trips: {len(filtered)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
