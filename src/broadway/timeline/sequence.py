from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from broadway.config import loader


class WalkthroughStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    order: int
    question: str
    kind: Literal["evidence", "decision", "analysis"]
    action: str = ""


class WalkthroughSequence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[WalkthroughStepConfig]


class DecisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methods: list[str]
    parents: list[str]


class WalkthroughConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skew_threshold: float
    kurtosis_threshold: float
    shapiro_alpha: float
    # Subsample seed consumed by stats.assumptions.check_normality; required
    # so no code-side default can shadow the YAML value.
    shapiro_seed: int
    imbalance_ratio_threshold: float
    significance_alpha: float
    max_qq_groups: int
    decisions: dict[str, DecisionSpec] = {}


def load_walkthrough_sequence() -> WalkthroughSequence:
    path = loader.CONFIGS_DIR / "flow" / "hypothesis_walkthrough.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return WalkthroughSequence(**data)


def load_walkthrough_config() -> WalkthroughConfig:
    path = loader.CONFIGS_DIR / "step" / "walkthrough.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return WalkthroughConfig(**data)
