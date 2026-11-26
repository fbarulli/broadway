"""Pydantic result contracts for the evaluate step."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from broadway.calibration.contracts import CalibrationEvaluation
from broadway.reports.decision import Decision

# SSOT for the binary-decision tuning objective: the config, the metric
# functions, and the persisted report all name the same set.
TuningObjective = Literal["precision", "recall", "f1", "accuracy"]

# The cross-validation vocabulary. The time-aware kinds are ordered,
# forward-chaining geometries: a fold never trains on rows after the rows it
# scores, and they carry the purge gap / seasonal-block / horizon knobs.
CvKind = Literal["kfold", "time_series_split", "expanding", "rolling"]
TIME_CV_KINDS: frozenset[str] = frozenset(
    {"time_series_split", "expanding", "rolling"}
)


class ThresholdGrid(BaseModel):
    """The prediction-cutoff grid sampled (inclusive) for threshold tuning."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(allow_inf_nan=False)
    max: float = Field(allow_inf_nan=False)
    steps: int = Field(ge=2)

    @model_validator(mode="after")
    def _ordered(self) -> ThresholdGrid:
        if self.min >= self.max:
            raise ValueError(
                f"threshold grid requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class DecisionConfig(BaseModel):
    """The declared binary framing for an otherwise-regression target.

    ``threshold`` is the decision definition (the truth cut: a row is positive
    when its value reaches it); the grid holds candidate model cutoffs. A
    decision is opt-in: absent config means no threshold report is written.
    """

    model_config = ConfigDict(extra="forbid")

    # Optional because the scored target is already declared by the dataset
    # contract; when given, the module pins it to that target.
    column: str | None = None
    threshold: float = Field(allow_inf_nan=False)
    objective: TuningObjective
    grid: ThresholdGrid


class ErrorAnalysisConfig(BaseModel):
    """Artifact paths for the per-row holdout error ledger and threshold report."""

    model_config = ConfigDict(extra="forbid")

    ledger_file: str = "errors.parquet"
    thresholds_file: str = "thresholds.json"
    # Empty means "every column of the experiment's ColumnPlan"; a configured
    # list is validated against that plan (never a code-side column literal).
    segment_columns: list[str] = []


# The ledger's residual column, declared once: the error-analysis pass writes
# it and the residual-diagnostics pass reads it back from the artifact.
LEDGER_RESIDUAL_COLUMN = "residual"


class ResidualDiagnosticsConfig(BaseModel):
    """Lag windows for the residual-diagnostics pass over the holdout ledger."""

    model_config = ConfigDict(extra="forbid")

    acf_lags: int = Field(default=3, ge=1)
    ljung_box_lags: int = Field(default=3, ge=1)


class BootstrapConfig(BaseModel):
    """Percentile-bootstrap settings for the holdout metric intervals."""

    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(default=200, ge=1)
    seed: int = 42
    level: float = Field(default=0.95, gt=0.0, lt=1.0)


class MetricInterval(BaseModel):
    """Bootstrap confidence interval for one holdout metric."""

    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float


class OriginMetric(BaseModel):
    """One forecast origin's multi-step scores (horizon > 1 only)."""

    model_config = ConfigDict(extra="forbid")

    fold: int
    origin: int
    n_steps: int = Field(ge=1)
    rmse: float
    mae: float


class TimeSeriesCvResult(BaseModel):
    """The time-series CV measurement: geometry, pooled metrics, per-step table.

    ``horizon_metrics`` maps each step ahead (1..horizon) to its pooled
    rmse/mae; ``origins`` is the per-origin table for multi-step runs. The
    ``baseline``/``improvement`` dicts compare the candidate with the
    seasonal-naive (or persistence) reference on the same scored rows.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    folds: int = Field(ge=1)
    horizon: int = Field(ge=1)
    gap: int = Field(ge=0)
    seasonal_period: int | None = None
    train_size: int | None = None
    time_indexed: bool
    test_size: int = Field(ge=1)
    train_sizes: list[int]
    metrics: dict[str, float]
    horizon_metrics: dict[int, dict[str, float]]
    origins: list[OriginMetric]
    baseline: dict[str, float]
    improvement: dict[str, float]


class ResidualDiagnostics(BaseModel):
    """Statistical residual diagnostics computed from the persisted ledger.

    ``acf_lags`` maps each tested lag (1..k) to the residual autocorrelation;
    ``ljung_box_*`` is the portmanteau autocorrelation test at
    ``ljung_box_lags``; the Jarque-Bera fields and ``qq_correlation`` are the
    normality read (a perfectly straight normal QQ line is 1.0).
    """

    model_config = ConfigDict(extra="forbid")

    n_obs: int
    acf_lags: dict[int, float]
    ljung_box_lags: int
    ljung_box_statistic: float
    ljung_box_p_value: float
    jarque_bera_statistic: float
    jarque_bera_p_value: float
    jarque_bera_skew: float
    jarque_bera_kurtosis: float
    qq_correlation: float


class ThresholdRow(BaseModel):
    """Confusion counts and derived rates at one prediction cutoff."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


class DecisionReport(BaseModel):
    """The tuned binary decision persisted to ``thresholds.<json>``."""

    model_config = ConfigDict(extra="forbid")

    column: str
    objective: TuningObjective
    label_threshold: float
    chosen_threshold: float
    rows: list[ThresholdRow]


class MetricComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: float | None
    champion: float | None
    delta: float | None
    delta_pct: float | None


class ModelComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, MetricComparison]


class BaselineComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline_value: float
    candidate_value: float
    delta: float
    improvement: float | None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    promote: bool
    reason: str
    cv_metrics: dict[str, float] | None = None
    cv_detail: TimeSeriesCvResult | None = None
    cv_decision: Decision | None = None
    residuals: dict[str, float] | None = None
    residual_diagnostics: ResidualDiagnostics | None = None
    metric_intervals: dict[str, MetricInterval] | None = None
    comparison: ModelComparison | None = None
    baseline: BaselineComparison | None = None
    calibration: CalibrationEvaluation | None = None
    warnings: list[str] = []
