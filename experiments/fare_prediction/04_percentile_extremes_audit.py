"""04: audit the bottom and top 1% of fares for plausibility."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _common import RESULTS, SAMPLE_NAME, UNIT_FMT
from matplotlib.ticker import FuncFormatter

from broadway.samples import read_named_sample

BOTTOM_Q = 0.01
TOP_Q = 0.99
PROFILE_COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
BOTTOM_SUSPECT = {"max_distance": 5.0, "max_duration": 30.0, "max_speed": 60.0}
TOP_SUSPECT = {"min_distance": 3.0, "min_duration": 5.0, "max_speed": 100.0}

EVIDENCE_CSV = RESULTS / "04_percentile_extremes_audit_describe.csv"
SUMMARY_MD = RESULTS / "04_percentile_extremes.md"
PNG_OUT = RESULTS / "04_percentile_extremes_audit.png"

FLAG_COLS = ["fare_amount", "trip_distance", "trip_duration_minutes", "speed_mph"]
PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.99]


def _profile(subset: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Describe ``subset`` on ``cols`` with the percentile bands of interest."""
    return subset[cols].describe(percentiles=PERCENTILES)


def _flagged(df: pd.DataFrame, thresholds: dict[str, float], high: bool) -> pd.DataFrame:
    """Return rows of ``df`` violating the plausibility thresholds.

    Speed is derived as ``trip_distance / (trip_duration_minutes / 60)`` in
    mph; a zero duration yields infinite speed, which any max-speed rule
    flags. For ``high=True`` (top fares) flag distance below ``min_distance``,
    duration below ``min_duration``, or speed above ``max_speed``; for
    ``high=False`` (bottom fares) flag distance above ``max_distance``,
    duration above ``max_duration``, or speed above ``max_speed``.
    """
    flagged = df.copy()
    duration_h = flagged["trip_duration_minutes"] / 60.0
    flagged["speed_mph"] = np.where(
        duration_h > 0, flagged["trip_distance"] / duration_h, np.inf
    )
    if high:
        violates = (
            (flagged["trip_distance"] < thresholds["min_distance"])
            | (flagged["trip_duration_minutes"] < thresholds["min_duration"])
            | (flagged["speed_mph"] > thresholds["max_speed"])
        )
    else:
        violates = (
            (flagged["trip_distance"] > thresholds["max_distance"])
            | (flagged["trip_duration_minutes"] > thresholds["max_duration"])
            | (flagged["speed_mph"] > thresholds["max_speed"])
        )
    return flagged[violates]


def _describe_table(desc: pd.DataFrame) -> str:
    """Render a describe frame as a clean, aligned Markdown pipe table (03 style)."""
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
    return body


def _flagged_rows_table(flagged: pd.DataFrame, limit: int = 10) -> str:
    """Render the top ``limit`` flagged trips as a unit-formatted pipe table."""
    head = flagged[FLAG_COLS].head(limit)
    header = ["#", "fare_amount", "trip_distance", "trip_duration_minutes", "speed_mph"]
    rows = []
    for i, (_, row) in enumerate(head.iterrows(), 1):
        rows.append([
            str(i),
            f"${row['fare_amount']:,.2f}",
            f"{row['trip_distance']:,.2f} mi",
            f"{row['trip_duration_minutes']:.1f} min",
            f"{row['speed_mph']:,.1f} mph",
        ])
    widths = [
        max(len(row[i]) for row in [header, *rows]) for i in range(len(header))
    ]

    def fmt(cells: list[str], aligns: list[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(w) if a == "r" else cell.ljust(w)
            for cell, w, a in zip(cells, widths, aligns)
        ) + " |"

    aligns = ["r"] * len(header)
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join([fmt(header, aligns), sep] + [fmt(row, aligns) for row in rows])
    return body


def _summary_md(
    bottom_desc: pd.DataFrame,
    top_desc: pd.DataFrame,
    n_bottom: int,
    n_top: int,
    q01: float,
    q99: float,
    bottom_flagged: pd.DataFrame,
    top_flagged: pd.DataFrame,
) -> str:
    """Render the audit Markdown summary (03 ``describe_to_markdown`` style)."""
    return "\n".join([
        "# Percentile extremes audit (bottom/top 1% of fares)",
        "",
        (f"Bottom 1%: fares <= ${q01:,.2f} — {n_bottom:,} rows "
         f"({len(bottom_flagged):,} flagged)"),
        (f"Top 1%: fares >= ${q99:,.2f} — {n_top:,} rows "
         f"({len(top_flagged):,} flagged)"),
        "",
        "## Bottom 1% describe",
        "",
        _describe_table(bottom_desc),
        "",
        "## Top 1% describe",
        "",
        _describe_table(top_desc),
        "",
        f"Flagged rows: bottom {len(bottom_flagged):,}, top {len(top_flagged):,}.",
        "",
        "## Bottom flagged rows (head 10)",
        "",
        _flagged_rows_table(bottom_flagged),
        "",
        "## Top flagged rows (head 10)",
        "",
        _flagged_rows_table(top_flagged),
        "",
    ])


def _combined_evidence(
    bottom_desc: pd.DataFrame,
    top_desc: pd.DataFrame,
    bottom_flagged: pd.DataFrame,
    top_flagged: pd.DataFrame,
) -> pd.DataFrame:
    """Join both describes and both flagged sets into ONE evidence table.

    Index = describe stat labels / suspicious row numbers; ``group``
    distinguishes the sections; ``speed_mph`` is NaN on describe rows.
    """
    rows: list[dict[str, object]] = []
    for label, desc in (("bottom_describe", bottom_desc), ("top_describe", top_desc)):
        for stat, row in desc.iterrows():
            rows.append({
                "group": label, "stat": stat,
                "fare_amount": row["fare_amount"],
                "trip_distance": row["trip_distance"],
                "trip_duration_minutes": row["trip_duration_minutes"],
                "speed_mph": np.nan,
            })
    for label, flagged in (("suspicious_bottom", bottom_flagged),
                           ("suspicious_top", top_flagged)):
        for i, (_, row) in enumerate(flagged.iterrows(), 1):
            rows.append({
                "group": label, "stat": i,
                "fare_amount": row["fare_amount"],
                "trip_distance": row["trip_distance"],
                "trip_duration_minutes": row["trip_duration_minutes"],
                "speed_mph": row["speed_mph"],
            })
    return pd.DataFrame(rows).set_index("stat")


def plot_extremes(
    df: pd.DataFrame,
    bottom: pd.DataFrame,
    top: pd.DataFrame,
    bottom_flagged: pd.DataFrame,
    top_flagged: pd.DataFrame,
    out_path: Path,
) -> None:
    """Two-panel figure: bottom-vs-top fare boxplot plus flagged-extremes scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    extremes = pd.concat([
        bottom.assign(group="bottom 1%"),
        top.assign(group="top 1%"),
    ])
    sns.boxplot(
        x="group", y="fare_amount", data=extremes, log_scale=True,
        whis=[0, 100], showmeans=True, ax=axes[0],
        meanprops={"marker": "o", "markerfacecolor": "#d62728",
                   "markeredgecolor": "#d62728", "markersize": 5},
    )
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: UNIT_FMT["fare_amount"].format(v))
    )
    axes[0].set_title("bottom vs top 1% fares")
    axes[0].grid(True, alpha=0.3, axis="y")
    sns.scatterplot(
        x="trip_distance", y="fare_amount", hue="group", data=extremes,
        palette={"bottom 1%": "#4c72b0", "top 1%": "#d62728"},
        s=8, alpha=0.5, ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    flagged = pd.concat([bottom_flagged, top_flagged])
    axes[1].scatter(
        x=flagged["trip_distance"], y=flagged["fare_amount"],
        marker="x", s=40, color="black", label="flagged",
    )
    axes[1].legend(title=None)
    axes[1].set_title("extremes: fare vs distance (flagged = x)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = sample.df

    q01 = float(df["fare_amount"].quantile(BOTTOM_Q))
    q99 = float(df["fare_amount"].quantile(TOP_Q))
    print(f"1% quantile fare: ${q01:,.2f}")
    print(f"99% quantile fare: ${q99:,.2f}")

    bottom = df[df["fare_amount"] <= q01][PROFILE_COLS]
    top = df[df["fare_amount"] >= q99][PROFILE_COLS]
    print(f"Bottom 1% (fare <= ${q01:,.2f}): {len(bottom):,} rows")
    print(f"Top 1% (fare >= ${q99:,.2f}): {len(top):,} rows")

    print("Bottom fare value_counts (head 8):")
    print(bottom["fare_amount"].value_counts().head(8))

    bottom_desc = _profile(bottom, PROFILE_COLS)
    top_desc = _profile(top, PROFILE_COLS)
    print("Bottom 1% describe:")
    print(bottom_desc)
    print("Top 1% describe:")
    print(top_desc)

    bottom_flagged = _flagged(bottom, BOTTOM_SUSPECT, high=False)
    top_flagged = _flagged(top, TOP_SUSPECT, high=True)
    print(f"Bottom flagged: {len(bottom_flagged)}")
    print(f"Top flagged: {len(top_flagged)}")
    print("Bottom flagged rows (head):")
    print(bottom_flagged[FLAG_COLS].head(10))
    print("Top flagged rows (head):")
    print(top_flagged[FLAG_COLS].head(10))

    combined = _combined_evidence(bottom_desc, top_desc, bottom_flagged, top_flagged)
    combined.to_csv(EVIDENCE_CSV)
    print(f"wrote {EVIDENCE_CSV} ({len(combined)} rows)")

    plot_extremes(df, bottom, top, bottom_flagged, top_flagged, PNG_OUT)
    print(f"wrote {PNG_OUT}")

    SUMMARY_MD.write_text(
        _summary_md(
            bottom_desc, top_desc, len(bottom), len(top), q01, q99,
            bottom_flagged, top_flagged,
        )
    )
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
