from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict

from broadway.config import loader


class StatsSequence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[str]


def load_stats_sequence() -> StatsSequence:
    path = loader.CONFIGS_DIR / "flow" / "stats_sequence.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return StatsSequence(**data)
