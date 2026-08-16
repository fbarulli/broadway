"""12: density scatter of fare_amount vs trip_distance on the ratecode1 dataset.

Reads the RatecodeID == 1 dataset built by 11_ratecode1_dataset.py and
renders the same fare-vs-distance scatter as 01, but on the clean
standard-metered-trip subset (no airport flat fares).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from _common import RATECODE1_PARQUET, RESULTS

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(RATECODE1_PARQUET)
    x = df["trip_distance"].to_numpy()
    y = df["fare_amount"].to_numpy()
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=1.5, alpha=0.15, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("trip_distance (miles)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title(f"fare_amount vs trip_distance (RatecodeID == 1, N={len(df)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    print(f"ratecode1 rows: {len(df)}")
    print(f"axis limits: trip_distance [{x_min:.2f}, {x_max:.2f}], fare_amount [{y_min:.2f}, {y_max:.2f}]")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
