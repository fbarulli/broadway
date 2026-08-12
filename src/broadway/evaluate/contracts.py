"""Pydantic result contracts for the evaluate step."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    promote: bool
    reason: str
