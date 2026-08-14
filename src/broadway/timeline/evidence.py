from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NormalityGroupStat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skew: float
    kurtosis: float
    shapiro_p: float


class NormalityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: dict[str, NormalityGroupStat]
    figure: str
    standardization: str = "per-group z-score"


class VarianceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statistic: float
    p_value: float


class PosthocPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    p_value: float
    cohens_d: float
    hedges_g: float
    effect_size_note: str


class PosthocEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    pairs: list[PosthocPair]
    significant_pairs: int


class ConclusionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: str
    principal_method: str
    p_value: float
    effect_size: str | None
    significant_pairs: int
    notes: list[str]
