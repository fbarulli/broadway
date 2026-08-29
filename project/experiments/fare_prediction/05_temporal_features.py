"""05: temporal features on the v3 sample — shared recipe from _common, evidence describe + summary."""

from pathlib import Path

import pandas as pd
from _common import (
    RESULTS,
    SAMPLE_NAME,
    TEMPORAL_FEATURES,
    build_temporal_features,
)

from broadway.samples import read_named_sample

DESCRIBE_CSV = RESULTS / "05_temporal_features_describe.csv"
SUMMARY_MD = RESULTS / "05_temporal_features.md"

FLAG_COLS = ["is_weekend", "is_rush_hour", "is_night"]
CYCLICAL_COLS = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]

FEATURE_COLS = [feat.name for feat in TEMPORAL_FEATURES]


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


def _summary_markdown(df: pd.DataFrame, cyclical_desc: pd.DataFrame) -> str:
    """Flag value counts + cyclical describe as clean pipe tables (03/04 style)."""
    sections = [
        "# Temporal features (v3 sample)",
        "",
        f"Total rows: {len(df):,}",
        "",
    ]
    for col in FLAG_COLS:
        counts = df[col].value_counts().sort_index()
        table = "\n".join(f"| {label} | {count:,} |" for label, count in counts.items())
        sections.append(f"## {col}\n\n| value | count |\n|---|---|\n{table}\n")
    sections.append("## Cyclical features describe\n\n" + _describe_table(cyclical_desc) + "\n")
    return "\n".join(sections)


def _write_evidence(
    csv_out: Path, md_out: Path, df: pd.DataFrame, cyclical_desc: pd.DataFrame
) -> None:
    """Persist the describe CSV and Markdown summary, printing each target."""
    df[FEATURE_COLS].describe().to_csv(csv_out)
    print(f"wrote {csv_out}")
    md_out.write_text(_summary_markdown(df, cyclical_desc))
    print(f"wrote {md_out}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    df = build_temporal_features(sample.df)
    print(f"Derived feature columns ({len(TEMPORAL_FEATURES)}): {FEATURE_COLS}")
    flag_counts = {col: df[col].value_counts().sort_index() for col in FLAG_COLS}
    print("\n\n".join(f"{col} value_counts:\n{counts}" for col, counts in flag_counts.items()))
    cyclical_desc = df[CYCLICAL_COLS].describe()
    print("Cyclical features describe:")
    print(cyclical_desc)
    _write_evidence(DESCRIBE_CSV, SUMMARY_MD, df, cyclical_desc)


if __name__ == "__main__":
    main()
