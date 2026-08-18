"""02: filter the 1M sample with the project's existing data policy and profile it."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CLEAN_PARQUET, RESULTS
from project.working import MIN_FARE

COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0]

CSV_OUT = RESULTS / "02_filtered_profile_describe.csv"
PNG_OUT = RESULTS / "02_filtered_profile.png"


# Per-percentile annotation placement: p -> (vertical anchor, horizontal
# anchor, offset in points). The 0.95-1.0 lines sit within 0.05 of each other,
# so the top-cluster labels are anchored above/below their lines and staggered
# to avoid overlapping one another and the curve.
LABEL_PLACEMENT = {
    0.01: ("center", "left", (3, 0)),
    0.05: ("center", "left", (3, 0)),
    0.50: ("center", "left", (3, 0)),
    0.95: ("top", "left", (3, -8)),
    0.99: ("top", "left", (3, -7)),
    0.999: ("bottom", "left", (3, 2)),
    1.0: ("top", "right", (-3, 0)),
}


def plot_profiles(df: pd.DataFrame, out_path: Path) -> None:
    """One figure, 3 ECDF subplots with dashed percentile gridlines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, col in zip(axes, COLS):
        values = df[col].sort_values()
        n = len(values)
        ax.plot(values, np.arange(1, n + 1) / n, linewidth=1.0)
        for p in PERCENTILES:
            ax.axhline(p, color="red", linestyle="--", linewidth=0.8)
        for p in PERCENTILES:
            va, ha, offset = LABEL_PLACEMENT[p]
            x = values.quantile(p)
            ax.annotate(f"{p * 100:g}% x={x:.1f}", xy=(x, p), xytext=offset,
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=6, color="red")
        ax.set_xscale("log")
        ax.set_xlabel("value")
        ax.set_ylabel("cumulative share")
        ax.set_title(f"{col} (N={n})")
        ax.grid(True, alpha=0.3)
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
