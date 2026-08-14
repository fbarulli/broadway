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
    figures: list[str]


class VarianceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statistic: float
    p_value: float
