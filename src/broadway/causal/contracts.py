"""ExperimentDesign / ExperimentResult Pydantic models and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExperimentDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment_column: str
    outcome_column: str
    power: float
    alpha: float
    effect_size: float
    sample_size: int
    mde: float


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    passed: bool
    reason: list[str]
    warnings: list[str]


def save_design(design: ExperimentDesign, path: Path) -> None:
    path.write_text(design.model_dump_json(indent=2))


def load_design(path: Path) -> ExperimentDesign:
    return ExperimentDesign.model_validate(json.loads(path.read_text()))


def save_result(result: ExperimentResult, path: Path) -> None:
    path.write_text(result.model_dump_json(indent=2))


def load_result(path: Path) -> ExperimentResult:
    return ExperimentResult.model_validate(json.loads(path.read_text()))
