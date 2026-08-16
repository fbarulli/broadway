from __future__ import annotations

import os

import pandas as pd

from broadway.discover.profile import build_profile
from broadway.onboard.models import ColumnHint, InferenceReport

_IDENTIFIER_THRESHOLD = float(os.getenv("BROADWAY_IDENTIFIER_THRESHOLD", "0.95"))


def _datetime_candidate(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().head(100)
        if sample.empty:
            return False
        converted = pd.to_datetime(sample, errors="coerce")
        return bool(converted.notna().mean() > 0.9)
    return False


def _is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    return pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 30


def infer(name: str, df: pd.DataFrame) -> InferenceReport:
    profile = build_profile(name, name, df)
    columns: dict[str, ColumnHint] = {}
    for col, col_profile in profile.columns.items():
        series = df[col]
        cardinality = col_profile.cardinality
        is_dt = _datetime_candidate(series)
        categorical = (not is_dt) and _is_categorical(series)
        if is_dt:
            role = "datetime"
        elif cardinality > 0 and col_profile.identifier_score >= _IDENTIFIER_THRESHOLD:
            role = "ignore"
        else:
            role = "feature"
        columns[col] = ColumnHint(
            dtype=col_profile.dtype,
            null_rate=round(col_profile.null_count / profile.row_count, 4) if profile.row_count else 0.0,
            cardinality=cardinality,
            identifier_score=col_profile.identifier_score,
            datetime_candidate=is_dt,
            categorical=categorical,
            suggested_role=role,
        )
    return InferenceReport(name=name, row_count=profile.row_count, columns=columns)
