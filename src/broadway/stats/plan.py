"""AnalysisPlan Pydantic model and JSON (de)serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script: str
    analysis_type: str
    test_name: str
    statistics: dict[str, float]
    effect_sizes: dict[str, float]
    threshold_context: dict[str, float | bool]
    reason: list[str]
    warnings: list[str]
    passed: bool
    next_step: str | None


def save_plan(plan: AnalysisPlan, path: Path) -> None:
    path.write_text(plan.model_dump_json(indent=2))


def load_plan(path: Path) -> AnalysisPlan:
    return AnalysisPlan.model_validate(json.loads(path.read_text()))
