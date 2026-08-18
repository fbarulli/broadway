"""03: profile the short trips (< 0.31 mi) subset of the named fare_prediction_1m sample.

Short trips are a slice of the validated named sample (only the distance
subset is applied here); the fare/duration policy lives in the sample
definition and was applied once at generation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _common import RESULTS, SAMPLE_NAME

from broadway.samples import read_named_sample

SHORT_DISTANCE = 0.31
SHORT_COLS = ["fare_amount", "trip_duration_minutes"]
SHORT_PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.99]

DESCRIBE_CSV = RESULTS / "03_short_trips_describe.csv"
SUMMARY_MD = RESULTS / "03_short_trips.md"
PNG_OUT = RESULTS / "03_short_trips.png"


def describe_to_markdown(title: str, desc: pd.DataFrame, n: int) -> str:
    """Render a describe frame as a clean, aligned Markdown pipe table.

    The first column carries the describe index labels (``count``, ``mean``,
    ``1%``, ...); the other columns are the described columns' values, with
    floats formatted ``:,.2f`` and the ``count`` row as integers.
    """
    rows: list[list[str]] = []
    for label, row in desc.iterrows():
        cells = [
            f"{int(value)}" if label == "count" else f"{value:,.2f}"
            for value in row
        ]
        rows.append([str(label), *cells])
    header = ["stat", *desc.columns]
    widths = [
        max(len(row[i]) for row in [header, *rows]) for i in range(len(header))
    ]

    def fmt(cells: list[str], aligns: list[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(w) if a == "r" else cell.ljust(w)
            for cell, w, a in zip(cells, widths, aligns)
        ) + " |"

    aligns = ["l"] + ["r"] * (len(header) - 1)
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join([fmt(header, aligns), sep] + [fmt(row, aligns) for row in rows])
    return f"# {title}\n\nTotal rows: {n:,}\n\n{body}\n"


def plot_short_trips(df: pd.DataFrame, short: pd.DataFrame, out_path: Path) -> None:
    """Two KDE panels: all trips vs short trips (fare, duration) on log-x.

    Shows where the short-trip slice sits inside the full sample's
    distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, col in zip(axes, SHORT_COLS):
        sns.kdeplot(df[col], label="all trips", color="#4c72b0", ax=ax)
        sns.kdeplot(short[col], label="short trips (<0.31 mi)", color="#d62728", ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.set_title(f"{col}: all vs short trips")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = sample.df
    short = df[(df["trip_distance"] > 0) & (df["trip_distance"] < SHORT_DISTANCE)]
    print(f"Total short trips: {len(short):,}")

    desc = short[SHORT_COLS].describe(percentiles=SHORT_PERCENTILES)
    print(desc)
    desc.to_csv(DESCRIBE_CSV)
    print(f"wrote {DESCRIBE_CSV}")

    SUMMARY_MD.write_text(describe_to_markdown("Short trips (< 0.31 mi)", desc, len(short)))
    print(f"wrote {SUMMARY_MD}")

    plot_short_trips(df, short, PNG_OUT)
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
