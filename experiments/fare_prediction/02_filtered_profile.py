"""02: profile the named fare_prediction_1m sample.

The fare/distance/duration policy lives in the sample definition
(configs/sample/fare_prediction_1m.yaml), applied once at generation; this
step only consumes the validated sample by name.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from broadway.samples import read_named_sample

from _common import RESULTS, SAMPLE_NAME

COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0]
UNIT_FMT = {
    "fare_amount": "${:g}",
    "trip_distance": "{:g} mi",
    "trip_duration_minutes": "{:.0f} min",
}

CSV_OUT = RESULTS / "02_filtered_profile_describe.csv"
PNG_OUT = RESULTS / "02_filtered_profile.png"


def plot_profiles(df: pd.DataFrame, out_path: Path) -> None:
    """One figure, 3 box-whisker plots on log-y with measurement marks.

    Whiskers span min→max; the mean is marked; min/1/5/50/95/99/99.9%/max are
    drawn as labeled dashed lines (labels left); each band's value count is
    shown on the right; the region above 99% is shaded as the tail.
    """
    mark_labels = (("min", 0.0), ("1%", 0.01), ("5%", 0.05), ("50%", 0.50),
                   ("95%", 0.95), ("99%", 0.99), ("99.9%", 0.999), ("max", 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, col in zip(axes, COLS):
        values = df[col]
        fmt = UNIT_FMT[col]
        sns.boxplot(
            y=values, log_scale=True, color="#4c72b0", whis=[0, 100],
            width=0.35, ax=ax, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "#d62728",
                       "markeredgecolor": "#d62728", "markersize": 5},
        )
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _p, fmt=fmt: fmt.format(v))
        )
        ax.yaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(base=10, subs=(2.0, 5.0))
        )
        ax.yaxis.set_minor_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _p, fmt=fmt: fmt.format(v))
        )
        ax.tick_params(axis="y", which="minor", labelsize=6)
        thresholds: list[tuple[float, str]] = []
        for label, p in mark_labels:
            y = float(values.min() if p == 0.0 else
                      values.max() if p == 1.0 else values.quantile(p))
            thresholds.append((y, label))
            ax.axhline(y, color="#d62728", linestyle="--", linewidth=0.8)
            ax.text(0.03, y, f"{label} = {fmt.format(y)}",
                    transform=ax.get_yaxis_transform(), fontsize=7,
                    color="#d62728", va="bottom")
        ax.axhspan(float(values.quantile(0.99)), float(values.max()),
                   alpha=0.08, color="#d62728")
        for (lo, _), (hi, _) in zip(thresholds[:-1], thresholds[1:]):
            n = int(((values > lo) & (values <= hi)).sum())
            ax.text(0.97, (lo * hi) ** 0.5, f"n = {n:,}",
                    transform=ax.get_yaxis_transform(), fontsize=6,
                    color="#555555", va="center", ha="right")
        ax.set_title(f"{col} (N={len(values)})")
        ax.set_ylabel("")
        ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    print(f"sample: {SAMPLE_NAME}@{sample.provenance['version']}")
    print(f"rows: {sample.provenance['row_count']}")
    print(f"artifact_sha256: {sample.provenance['artifact_sha256']}")

    df = sample.df
    desc = df[COLS].describe(percentiles=PERCENTILES)
    print(desc)
    desc.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")

    plot_profiles(df, PNG_OUT)
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
