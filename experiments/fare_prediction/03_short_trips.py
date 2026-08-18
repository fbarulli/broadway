"""03: profile the short trips (< 0.31 mi) subset.

Short trips are drawn from the policy-filtered 1M sample (FILTERED_PARQUET,
written by 02); the fare/duration policy is already applied there, so only
the distance filter is applied here.
"""

import pandas as pd

from _common import FILTERED_PARQUET, RESULTS

SHORT_DISTANCE = 0.31
SHORT_COLS = ["fare_amount", "trip_duration_minutes"]
SHORT_PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.99]

DESCRIBE_CSV = RESULTS / "03_short_trips_describe.csv"
SUMMARY_MD = RESULTS / "03_short_trips.md"


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


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FILTERED_PARQUET)
    short = df[(df["trip_distance"] > 0) & (df["trip_distance"] < SHORT_DISTANCE)]
    print(f"Total short trips: {len(short):,}")

    desc = short[SHORT_COLS].describe(percentiles=SHORT_PERCENTILES)
    print(desc)
    desc.to_csv(DESCRIBE_CSV)
    print(f"wrote {DESCRIBE_CSV}")

    SUMMARY_MD.write_text(describe_to_markdown("Short trips (< 0.31 mi)", desc, len(short)))
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
