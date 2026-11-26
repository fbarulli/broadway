"""Validated result shapes shared by the EDA analyzers and the HTML report.

Every boundary (config, frame results, report inputs) is a Pydantic model so
the report composes typed values only. Missingness is an OBSERVATION here —
no imputation belongs to this package (Batch 2 owns it).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class NumericStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None
    skew: float | None
    kurtosis: float | None
    zero_count: int
    negative_count: int


class CategoryFrequency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    count: int
    share: float


class CategoricalStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    unique: int
    top_values: list[CategoryFrequency]


class TemporalStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    unique: int
    minimum: str | None
    maximum: str | None


class ColumnSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    role: str
    kind: Literal["numeric", "categorical", "temporal"]
    memory_bytes: int
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    temporal: TemporalStats | None = None


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    columns: int
    total_memory_bytes: int
    numeric_columns: int
    categorical_columns: int
    temporal_columns: int
    detail: list[ColumnSummary]


class MissingColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    missing_count: int
    missing_rate: float


class MissingPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    row_count: int
    row_rate: float


class MissingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_rows: int
    total_cells: int
    missing_cells: int
    missing_rate: float
    columns: list[MissingColumn]
    complete_columns: list[str]
    patterns: list[MissingPattern]
    pattern_count: int


class OutlierStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    iqr_count: int
    iqr_rate: float
    iqr_lower: float | None
    iqr_upper: float | None
    z_count: int
    z_rate: float


class ImbalanceStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicable: bool
    reason: str | None
    target: str
    value_counts: list[CategoryFrequency]
    minority_share: float | None
    imbalance_ratio: float | None
    flagged: bool | None


class QualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duplicate_rows: int
    duplicate_rate: float
    constant_columns: list[str]
    outliers: list[OutlierStats]
    imbalance: ImbalanceStats


class ColumnShift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    before_mean: float | None
    after_mean: float | None
    before_median: float | None
    after_median: float | None
    mean_delta: float | None
    median_delta: float | None


class CompareResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    skip_reason: str | None
    before_label: str
    after_label: str
    before_rows: int | None
    after_rows: int | None
    row_delta: int | None
    added_columns: list[str]
    removed_columns: list[str]
    dtype_changes: dict[str, str]
    numeric_shifts: list[ColumnShift]
    notes: list[str]


class EmbeddedFigure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    mime_type: str
    data_base64: str


class EdaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    experiment: str | None
    task: str
    source_rows: int
    sampled_rows: int
    sampled: bool
    summary: SummaryResult
    missing: MissingResult
    quality: QualityResult
    compare: CompareResult
    figures: list[EmbeddedFigure]
