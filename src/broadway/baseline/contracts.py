from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from broadway.analysis.contracts import AnalysisMode


class BaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: AnalysisMode
    strategy: str
    metric: str
    value: float
    details: dict[str, Any]
    notes: list[str]


def save_result(result: BaselineResult, path: Path) -> None:
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_result(path: Path) -> BaselineResult:
    return BaselineResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
