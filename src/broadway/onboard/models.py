from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ColumnHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str
    null_rate: float
    cardinality: int
    identifier_score: float
    datetime_candidate: bool
    categorical: bool
    suggested_role: str  # one of "feature" | "datetime" | "ignore"


class InferenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    row_count: int
    columns: dict[str, ColumnHint]
