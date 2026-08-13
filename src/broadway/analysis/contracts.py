from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class AnalysisMode(str, Enum):
    PREDICTION = "prediction"
    HYPOTHESIS = "hypothesis"
    CAUSAL = "causal"


class AnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: AnalysisMode
    goal: str
    row_definition: str
    decision_moment: str
    available_info: list[str]
    leakage_notes: list[str]
    success_criterion: str

    @model_validator(mode="after")
    def _reject_empty_strings(self) -> "AnalysisContract":
        for field in ("goal", "row_definition", "decision_moment", "success_criterion"):
            if not getattr(self, field).strip():
                raise ValueError(f"'{field}' must be non-empty")
        if not self.available_info:
            raise ValueError("available_info must not be empty")
        return self
