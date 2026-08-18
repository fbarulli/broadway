"""02: filter the 1M sample with the project's existing data policy and profile it."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import CLEAN_PARQUET, RESULTS
from project.working import MIN_FARE

COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0]

CSV_OUT = RESULTS / "02_filtered_profile_describe.csv"
PNG_OUT = RESULTS / "02_filtered_profile.png"


def plot_profiles(df: pd.DataFrame, out_path: Path) -> None:
    """One figure, 3 histograms with dashed lines at the PERCENTILES quantiles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, COLS):
        values = df[col]
        ax.hist(values, bins=60, edgecolor="white", linewidth=0.5)
        for p in PERCENTILES:
            ax.axvline(values.quantile(p), color="red", linestyle="--",
                       linewidth=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(f"{col} (N={len(values)})")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CLEAN_PARQUET)
    n_before = len(df)

    # MIN_FARE comes from working.yaml (config-driven, not a new constant);
    # the other two guards are consistent with upstream cleaning.
    df = df[(df["fare_amount"] > MIN_FARE) & (df["trip_distance"] > 0)
            & (df["trip_duration_minutes"] > 0)]

    n_after = len(df)
    print(f"rows before: {n_before}")
    print(f"rows after: {n_after}")
    print(f"rows removed: {n_before - n_after}")

    desc = df[COLS].describe(percentiles=PERCENTILES)
    print(desc)
    desc.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")

    plot_profiles(df, PNG_OUT)
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
