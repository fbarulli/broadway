"""TrainingResult Pydantic contract."""

from __future__ import annotations

from pydantic import BaseModel


class TrainingResult(BaseModel):
    model_type: str
    params: dict[str, float | int | str]
    train_time_seconds: float
    artifact_path: str | None = None
