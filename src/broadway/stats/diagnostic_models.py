from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    evidence: list[str]
    ramification: str
    warnings: list[str] = Field(default_factory=list)
