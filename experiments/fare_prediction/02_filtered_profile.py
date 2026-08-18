"""02: filter the 1M sample with the project's existing data policy and profile it."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _common import CLEAN_PARQUET, RESULTS
from project.working import MIN_FARE

COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0]

CSV_OUT = RESULTS / "02_filtered_profile_describe.csv"
PNG_OUT = RESULTS / "02_filtered_profile.png"


def plot_profiles(df: pd.DataFrame, out_path: Path) -> None:
    """One figure, 3 box-whisker plots on log-y with the key values annotated.

    Whiskers span min→max (whis=[0, 100]); the mean is marked; the describe
    percentiles are printed on the panel so the exact amounts are readable.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, col in zip(axes, COLS):
        values = df[col]
        sns.boxplot(
            y=values, log_scale=True, color="#4c72b0", whis=[0, 100],
            width=0.35, ax=ax, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "#d62728",
                       "markeredgecolor": "#d62728", "markersize": 5},
        )
        lines = [f"mean = {float(values.mean()):.3g}"]
        for p in (0.50, 0.95, 0.99, 0.999):
            lines.append(f"{p * 100:g}% = {float(values.quantile(p)):.3g}")
        lines.append(f"max = {float(values.max()):.3g}")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=7, family="monospace")
        ax.set_title(f"{col} (N={len(values)})")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3, axis="y")
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
