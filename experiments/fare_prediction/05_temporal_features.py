"""05: temporal features on the v2 sample — reuse platform builders, add cyclical + exact rush/night."""

from pathlib import Path

import numpy as np
import pandas as pd
from _common import RESULTS, SAMPLE_NAME

from broadway.config.schema import DerivedFeature
from broadway.features.builders import build_derived
from broadway.samples import read_named_sample

RUSH_HOURS = [7, 8, 9, 10, 16, 17, 18, 19]
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
DATETIME_SRC = "pickup_datetime"

DESCRIBE_CSV = RESULTS / "05_temporal_features_describe.csv"
SUMMARY_MD = RESULTS / "05_temporal_features.md"

FLAG_COLS = ["is_weekend", "is_rush_hour", "is_night"]
CYCLICAL_COLS = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]

FEATURES = [
    DerivedFeature(name="pickup_hour", func="pickup_hour", source=DATETIME_SRC),
    DerivedFeature(name="pickup_day_of_week", func="pickup_day_of_week", source=DATETIME_SRC),
    DerivedFeature(name="pickup_month", func="pickup_month", source=DATETIME_SRC),
    DerivedFeature(name="is_weekend", func="is_weekend", source=DATETIME_SRC),
    DerivedFeature(name="is_rush_hour", func="is_rush_hour", source=DATETIME_SRC),
    DerivedFeature(name="is_night", func="is_night", source=DATETIME_SRC),
    DerivedFeature(name="hour_sin", func="hour_sin", source=DATETIME_SRC),
    DerivedFeature(name="hour_cos", func="hour_cos", source=DATETIME_SRC),
    DerivedFeature(name="dayofweek_sin", func="dayofweek_sin", source=DATETIME_SRC),
    DerivedFeature(name="dayofweek_cos", func="dayofweek_cos", source=DATETIME_SRC),
]

FEATURE_COLS = [feat.name for feat in FEATURES]


def _is_rush_hour(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """1 on weekdays whose pickup hour falls in a peak window (7-10 or 16-19)."""
    dt = pd.to_datetime(df[source])
    return ((dt.dt.dayofweek < 5) & (dt.dt.hour.isin(RUSH_HOURS))).astype(int)


def _is_night(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """1 when the pickup hour is 20-23 or 0-5 (hour >= 20 or < 6)."""
    return pd.to_datetime(df[source]).dt.hour.isin(NIGHT_HOURS).astype(int)


def _cyclical(values: pd.Series, period: int, trig: str) -> pd.Series:
    """Map ``values`` onto one ``period`` cycle as sin or cos ([-1, 1])."""
    angle = 2.0 * np.pi * values / period
    return np.sin(angle) if trig == "sin" else np.cos(angle)


def _hour_sin(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Sine of pickup hour on a 24-hour cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.hour, 24, "sin")


def _hour_cos(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Cosine of pickup hour on a 24-hour cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.hour, 24, "cos")


def _dayofweek_sin(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Sine of pickup day-of-week on a 7-day cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.dayofweek, 7, "sin")


def _dayofweek_cos(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Cosine of pickup day-of-week on a 7-day cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.dayofweek, 7, "cos")


_CUSTOM_BUILDERS = {
    "is_rush_hour": _is_rush_hour,
    "is_night": _is_night,
    "hour_sin": _hour_sin,
    "hour_cos": _hour_cos,
    "dayofweek_sin": _dayofweek_sin,
    "dayofweek_cos": _dayofweek_cos,
}


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
        "# Temporal features (v2 sample)",
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
    df = build_derived(sample.df, FEATURES, "fare_amount", extra_builders=_CUSTOM_BUILDERS)
    print(f"Derived feature columns ({len(FEATURES)}): {FEATURE_COLS}")
    flag_counts = {col: df[col].value_counts().sort_index() for col in FLAG_COLS}
    print("\n\n".join(f"{col} value_counts:\n{counts}" for col, counts in flag_counts.items()))
    cyclical_desc = df[CYCLICAL_COLS].describe()
    print("Cyclical features describe:")
    print(cyclical_desc)
    _write_evidence(DESCRIBE_CSV, SUMMARY_MD, df, cyclical_desc)


if __name__ == "__main__":
    main()
