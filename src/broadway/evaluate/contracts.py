"""Pydantic result contracts for the evaluate step."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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
    warnings: list[str] = []
