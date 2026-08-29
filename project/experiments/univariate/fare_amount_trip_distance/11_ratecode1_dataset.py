"""11: build the RatecodeID == 1 dataset (standard metered trips) + density scatter.

Merged from former steps 11 + 12 (reuse directive): build `ratecode1_sample`
from the raw yellow-cab parquets, then render the fare-vs-distance density
scatter (previously step 12) on the working dataset. Outputs follow the
producing step: the parquet is the working dataset (`project.working`), the
plot is `11_ratecode1_scatter.png`.
"""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from _common import RATECODE1_PARQUET, RAW_DIR, RESULTS, load_working

SAMPLE_SIZE = 50_000
SEED = 42
RATE_CODE = 1

OUT = RESULTS / "11_ratecode1_scatter.png"


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

    working = load_working()
    x = working["trip_distance"].to_numpy()
    y = working["fare_amount"].to_numpy()
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=1.5, alpha=0.15, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("trip_distance (miles)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title(f"fare_amount vs trip_distance (RatecodeID == 1, N={len(working)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"working rows: {len(working)}")
    print(f"axis limits: trip_distance [{x_min:.2f}, {x_max:.2f}], fare_amount [{y_min:.2f}, {y_max:.2f}]")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
