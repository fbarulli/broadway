"""AIC/BIC comparison over already-fitted candidate models.

Callers pass fitted results that expose ``.aic`` and ``.bic`` (the statsmodels
convention); this module ranks them, adds the delta and the Akaike/BIC weights,
and never decides anything itself. Candidates MUST be fitted to the same
response and the same observations — the criteria are not comparable otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict


class FittedInformation(Protocol):
    """Duck type of a fitted result carrying information criteria."""

    aic: float
    bic: float


class CandidateInformation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    aic: float
    bic: float
    delta_aic: float
    delta_bic: float
    akaike_weight: float
    bic_weight: float


class InformationTable(BaseModel):
    """Candidates ranked by AIC ascending (``delta_aic`` 0 for the best)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: list[CandidateInformation]
    best_by_aic: str
    best_by_bic: str


def compare_aic_bic(results: Mapping[str, FittedInformation]) -> InformationTable:
    """Rank ``{name: fitted_result}`` by AIC/BIC with deltas and weights."""
    if not results:
        raise ValueError("compare_aic_bic needs at least one candidate")
    names = list(results)
    aic = np.array([float(results[name].aic) for name in names])
    bic = np.array([float(results[name].bic) for name in names])
    for criterion, values in (("aic", aic), ("bic", bic)):
        bad = [names[index] for index in np.flatnonzero(~np.isfinite(values))]
        if bad:
            raise ValueError(f"candidate(s) {bad} report a non-finite {criterion}")
    delta_aic = aic - aic.min()
    delta_bic = bic - bic.min()
    akaike = _weights(delta_aic)
    bic_weights = _weights(delta_bic)
    order = sorted(range(len(names)), key=lambda index: (aic[index], bic[index], names[index]))
    candidates = [
        CandidateInformation(
            name=names[index],
            aic=float(aic[index]),
            bic=float(bic[index]),
            delta_aic=float(delta_aic[index]),
            delta_bic=float(delta_bic[index]),
            akaike_weight=float(akaike[index]),
            bic_weight=float(bic_weights[index]),
        )
        for index in order
    ]
    return InformationTable(
        candidates=candidates,
        best_by_aic=names[int(np.argmin(aic))],
        best_by_bic=names[int(np.argmin(bic))],
    )


def _weights(delta: np.ndarray) -> np.ndarray:
    """Akaike-style weights: exp(-delta/2) normalized to sum to 1."""
    weights = np.exp(-0.5 * delta)
    return weights / weights.sum()
