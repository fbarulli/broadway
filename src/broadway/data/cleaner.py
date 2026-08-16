"""Filter invalid rows and drop duplicates, returning the DataFrame and drop accounting.

``canonicalize`` performs the locked structural-cleaning order for the etl step:
duplicates → missing-encoding normalization → datetime parsing → target-null drop.
"""

from __future__ import annotations

import pandas as pd

from broadway.cleaning.models import ParseFailure
from broadway.cleaning.structural import parse_datetime, parse_numeric, standardize_missing
from broadway.config.schema import DatasetContract


def clean(df: pd.DataFrame, dataset: DatasetContract) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    drops: list[tuple[str, int]] = []
    before = len(df)
    df = df.dropna(subset=[dataset.target])
    if len(df) < before:
        drops.append(("null target", before - len(df)))
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        drops.append(("duplicates", before - len(df)))
    return df, drops


def canonicalize(
    df: pd.DataFrame,
    target: str,
    datetime_columns: list[str],
    missing_encodings: list[str],
    numeric_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[ParseFailure], dict[str, list[str]]]:
    numeric_columns = numeric_columns or {}
    reasons: list[str] = []
    parse_failures: list[ParseFailure] = []
    observed_missing: dict[str, list[str]] = {}

    before = len(df)
    df = df.drop_duplicates().copy()
    dropped = before - len(df)
    if dropped:
        reasons.append(f"duplicates: -{dropped} rows")

    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col], observed = standardize_missing(df[col], col, missing_encodings)
            if observed:
                observed_missing[col] = observed

    for col in datetime_columns:
        if col in df.columns:
            df[col], failure = parse_datetime(df[col], col)
            if failure:
                parse_failures.append(failure)

    for col, target_dtype in numeric_columns.items():
        if col in df.columns:
            df[col], failure = parse_numeric(df[col], col, target_dtype)
            if failure:
                parse_failures.append(failure)

    before = len(df)
    df = df.dropna(subset=[target])
    dropped = before - len(df)
    if dropped:
        reasons.append(f"null target: -{dropped} rows")

    return df, reasons, parse_failures, observed_missing
