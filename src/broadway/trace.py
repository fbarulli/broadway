from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ArtifactTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    commit: str
    dataset: str | None = None
    analysis_goal: str | None = None
