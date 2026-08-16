from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class AnalysisMode(str, Enum):
    PREDICTION = "prediction"
    HYPOTHESIS = "hypothesis"
    CAUSAL = "causal"


class HypothesisConfig(BaseModel):
    group_column: str
    group_values: list[str]


class AnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mode: AnalysisMode
    goal: str
    row_definition: str
    decision_moment: str
    available_info: list[str]
    leakage_notes: list[str]
    success_criterion: str
    hypothesis: HypothesisConfig | None = None

    @model_validator(mode="after")
    def _reject_empty_strings(self) -> AnalysisContract:
        for field in ("name", "goal", "row_definition", "decision_moment", "success_criterion"):
            if not getattr(self, field).strip():
                raise ValueError(f"'{field}' must be non-empty")
        if not self.available_info:
            raise ValueError("available_info must not be empty")
        return self

    @model_validator(mode="after")
    def _require_hypothesis_group(self) -> AnalysisContract:
        if self.mode == AnalysisMode.HYPOTHESIS and self.hypothesis is None:
            raise ValueError("hypothesis mode requires a 'hypothesis' block (group_column, group_values)")
        return self


def require_mode(analysis: AnalysisContract | None, expected: AnalysisMode) -> AnalysisContract:
    if analysis is None:
        raise ValueError(f"step requires an analysis contract (--analysis) with mode '{expected.value}'")
    if analysis.mode != expected:
        raise ValueError(
            f"analysis mode mismatch: expected '{expected.value}', got '{analysis.mode.value}'"
        )
    return analysis
