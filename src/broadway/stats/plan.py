"""AnalysisPlan dataclass and JSON (de)serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AnalysisPlan:
    script: str
    analysis_type: str
    test_name: str
    statistics: dict
    effect_sizes: dict
    threshold_context: dict
    reason: list[str]
    warnings: list[str]
    passed: bool
    next_step: str | None


def save_plan(plan: AnalysisPlan, path: Path) -> None:
    with path.open("w") as f:
        json.dump(asdict(plan), f, indent=2)


def load_plan(path: Path) -> AnalysisPlan:
    with path.open("r") as f:
        data = json.load(f)
    return AnalysisPlan(**data)
