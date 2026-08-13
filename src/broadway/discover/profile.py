from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str
    null_count: int
    cardinality: int
    min: str | None
    max: str | None
    datetime_min: str | None
    datetime_max: str | None
    identifier_score: float


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    row_count: int
    columns: dict[str, ColumnProfile]


def _stringify(value: object) -> str | None:
    return None if value is None else str(value)


def build_profile(name: str, path: str, df: pd.DataFrame) -> DatasetProfile:
    row_count = int(len(df))
    columns: dict[str, ColumnProfile] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        min_val = non_null.min() if not non_null.empty else None
        max_val = non_null.max() if not non_null.empty else None
        try:
            min_s = _stringify(min_val)
            max_s = _stringify(max_val)
        except Exception:
            min_s, max_s = None, None
        dt_min = min_val.isoformat() if is_dt and min_val is not None else None
        dt_max = max_val.isoformat() if is_dt and max_val is not None else None
        columns[col] = ColumnProfile(
            dtype=str(series.dtype),
            null_count=int(series.isna().sum()),
            cardinality=int(series.nunique(dropna=True)),
            min=min_s,
            max=max_s,
            datetime_min=dt_min,
            datetime_max=dt_max,
            identifier_score=round(int(series.nunique(dropna=True)) / row_count, 4) if row_count else 0.0,
        )
    return DatasetProfile(name=name, path=path, row_count=row_count, columns=columns)
