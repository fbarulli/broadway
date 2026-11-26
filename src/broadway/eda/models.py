"""Validated result shapes shared by the EDA analyzers and the HTML report.

Every boundary (config, frame results, report inputs) is a Pydantic model so
the report composes typed values only. Missingness is an OBSERVATION here —
no imputation belongs to this package (Batch 2 owns it).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from broadway.eda.kinds import FeatureKind, TargetKind
from broadway.reports.decision import Decision, Verdict


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
    n: int
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
    """One column's ETL shift: point deltas plus their bootstrap intervals."""

    model_config = ConfigDict(extra="forbid")

    column: str
    before_mean: float | None
    after_mean: float | None
    before_median: float | None
    after_median: float | None
    mean_delta: float | None
    mean_delta_low: float | None = None
    mean_delta_high: float | None = None
    median_delta: float | None
    median_delta_low: float | None = None
    median_delta_high: float | None = None


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
    decision: Decision | None = None


class EmbeddedFigure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    mime_type: str
    data_base64: str


class CorrelationPair(BaseModel):
    """One numeric pair: Pearson and Spearman coefficients, n, p and BH q."""

    model_config = ConfigDict(extra="forbid")

    left: str
    right: str
    n: int
    pearson: float | None
    pearson_p: float | None
    pearson_q: float | None
    spearman: float | None
    spearman_p: float | None
    spearman_q: float | None


class TargetCorrelation(BaseModel):
    """One feature ranked against the target (the declared method)."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    method: Literal["pearson", "spearman"]
    value: float
    p_value: float
    q_value: float
    n: int


class CorrelationResult(BaseModel):
    """Pairwise correlation evidence; ``reason`` records why it is empty."""

    model_config = ConfigDict(extra="forbid")

    target: str
    method: Literal["pearson", "spearman"]
    columns: list[str]
    pearson: list[list[float | None]]
    spearman: list[list[float | None]]
    pairs: list[CorrelationPair]
    ranking: list[TargetCorrelation]
    reason: str | None = None


class FeatureTestResult(BaseModel):
    """One feature's association with the target: test, effect, p, q, verdict."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    feature_kind: FeatureKind
    target_kind: TargetKind
    test: Literal["spearman", "welch", "mann_whitney", "anova", "kruskal", "chi2"]
    effect_name: Literal[
        "abs_spearman_r",
        "point_biserial_r",
        "rank_biserial_r",
        "eta_squared",
        "epsilon_squared",
        "cramers_v",
    ]
    effect: float
    p_value: float
    q_value: float
    n: int
    significant: bool
    verdict: Verdict | None


class FeatureTestBundle(BaseModel):
    """The feature-test family: rows for the table, decisions for the record."""

    model_config = ConfigDict(extra="forbid")

    target: str
    target_kind: TargetKind | None
    alpha: float
    min_effect: float
    features: list[FeatureTestResult]
    skipped: list[str]
    decisions: list[Decision]
    summary: Decision | None
    reason: str | None = None


class OutlierVerdict(BaseModel):
    """One column's outlier share against the configured maximum share."""

    model_config = ConfigDict(extra="forbid")

    column: str
    n: int
    iqr_count: int
    iqr_share: float
    z_count: int
    z_share: float
    max_share: float
    verdict: Verdict


class OutlierResult(BaseModel):
    """Per-column outlier shares and their decisions."""

    model_config = ConfigDict(extra="forbid")

    max_share: float
    columns: list[OutlierVerdict]
    decisions: list[Decision]


class EdaDecisionBundle(BaseModel):
    """The step's decision record: the summary plus each family's decisions."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    summary: Decision | None
    comparison: Decision | None
    features: list[Decision]
    outliers: list[Decision]


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
    correlation: CorrelationResult
    feature_tests: FeatureTestBundle
    outliers: OutlierResult
    figures: list[EmbeddedFigure]
