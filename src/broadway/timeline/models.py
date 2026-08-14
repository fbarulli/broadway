from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StepStatus(str, Enum):
    COMPLETED = "completed"
    WARNING = "warning"


class AnalysisStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: str
    step_id: str
    order: int
    question: str
    status: StepStatus
    method: str | None
    source: str
    sample_name: str | None
    evidence_refs: list[str]
    result_summary: dict
    ramification: str
    decision_required: bool
    performed_at: str


class AnalysisDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: str
    id: str
    kind: Literal["omnibus", "posthoc"]
    question: str
    method: str
    reason: list[str]
    status: Literal["resolved"]
    parents: list[str]
    decided_at: str


class Alternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    command: str
    intent: Literal["support", "challenge", "alternative"]
    rationale: str


class Suggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    headline: str
    rationale: list[str]
    command: str
    alternatives: list[Alternative] = []
