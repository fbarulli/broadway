"""Primitives for structural cleaning: datetime parsing and missing encoding."""

from __future__ import annotations

import pandas as pd

from broadway.cleaning.models import ParseFailure


def parse_datetime(series: pd.Series, column: str) -> tuple[pd.Series, ParseFailure | None]:
    coerced = pd.to_datetime(series, errors="coerce")
    failed = series.notna() & coerced.isna()
    failure = None
    if failed.any():
        examples = [str(v) for v in series[failed].dropna().unique()[:5]]
        failure = ParseFailure(
            column=column,
            count=int(failed.sum()),
            examples=examples,
            target_dtype="datetime",
        )
    return coerced, failure


def standardize_missing(
    series: pd.Series, column: str, encodings: list[str]
) -> tuple[pd.Series, list[str]]:
    observed: list[str] = []
    for value in series.unique():
        if value in encodings and value not in observed:
            observed.append(value)
    cleaned = series.replace(encodings, None)
    return cleaned, observed
