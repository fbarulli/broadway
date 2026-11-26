"""Pydantic result contracts for the evaluate step."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from broadway.calibration.contracts import CalibrationEvaluation

# SSOT for the binary-decision tuning objective: the config, the metric
# functions, and the persisted report all name the same set.
TuningObjective = Literal["precision", "recall", "f1", "accuracy"]


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
    residuals: dict[str, float] | None = None
    comparison: ModelComparison | None = None
    baseline: BaselineComparison | None = None
    calibration: CalibrationEvaluation | None = None
    warnings: list[str] = []
