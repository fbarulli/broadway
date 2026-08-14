from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from broadway.config import loader


class WalkthroughStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    order: int
    question: str
    kind: Literal["evidence", "decision", "analysis"]


class WalkthroughSequence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[WalkthroughStepConfig]


def load_walkthrough_sequence() -> WalkthroughSequence:
    path = loader.CONFIGS_DIR / "flow" / "hypothesis_walkthrough.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return WalkthroughSequence(**data)
