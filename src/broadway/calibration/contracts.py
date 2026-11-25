"""Pydantic contracts for split-conformal interval calibration.

Config-boundary and artifact shapes only; no broadway imports.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CalibrationConfig(BaseModel):
    """Experiment-level calibration request (``experiment.calibration``).

    ``calibration_size`` is a fraction of the TRAINING split (fit + calibration),
    not of the full dataset; the holdout is never used to fit ``q``.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["split_conformal"]
    alpha: float = Field(gt=0.0, lt=1.0)
    calibration_size: float = Field(gt=0.0, lt=1.0)


class CalibrationSummary(BaseModel):
    """Fitted calibration: residual quantile ``q`` and provenance of the fit."""

    method: Literal["split_conformal"]
    alpha: float
    q: float
    n_calibration: int
    calibration_size: float


class PeriodCoverage(BaseModel):
    """Holdout coverage for one ISO week (``YYYY-Www``)."""

    period: str
    n: int
    coverage: float


class CalibrationEvaluation(BaseModel):
    """Holdout evaluation of the calibrated interval against nominal coverage."""

    method: Literal["split_conformal"]
    alpha: float
    q: float
    nominal_coverage: float
    coverage: float
    mean_width: float
    periods: list[PeriodCoverage]
    coverage_tolerance: float | None = None
    calibrated: bool | None = None
