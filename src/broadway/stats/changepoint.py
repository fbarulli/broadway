"""Change-point detection: CUSUM mean shifts, plus an optional PELT path.

The tabular two-sided CUSUM is implemented in ten vectorized numpy lines because
statsmodels ships no index-returning CUSUM (``breaks_cusumolsresid`` is a
parameter-stability test over OLS residuals, not a change-index detector) and
scipy has no equivalent — so this is the documented exception to the
library-first ladder.

PELT lives behind ``ruptures``, which is deliberately NOT a dependency: the
import is deferred and fails with the install command instead of a bare
ModuleNotFoundError.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from broadway.stats.time_series import ArrayLike, as_series

logger = logging.getLogger(__name__)

# Conventional CUSUM allowance: half the mean shift worth detecting, in sigma.
ALLOWANCE_SIGMAS = 0.5
_PELT_INSTALL_HINT = (
    "PELT change-point detection needs the optional 'ruptures' package, which is not a "
    "declared dependency. Install it in the lane venv (`uv pip install ruptures`) or call "
    "cusum_changepoints(), which needs no extra package."
)


class ChangePointMethod(str, Enum):
    CUSUM = "cusum"
    PELT = "pelt"


class ChangePointResult(BaseModel):
    """Change indices are 0-based: the first observation of each new regime."""

    model_config = ConfigDict(extra="forbid")
    method: ChangePointMethod
    change_indices: list[int]
    n: int
    min_size: int
    threshold: float | None = None
    penalty: float | None = None


def cusum_changepoints(series: ArrayLike, threshold: float, min_size: int) -> ChangePointResult:
    """Two-sided CUSUM on a mean shift; ``threshold`` is in standard deviations.

    The centre is the series median (robust to the shift itself) and the
    allowance is :data:`ALLOWANCE_SIGMAS` sigma, so ``threshold`` alone tunes
    sensitivity: larger means fewer alarms. Run it on the STL residual (see
    :mod:`broadway.stats.decompose`) when the input still carries a strong
    cycle — a daily seasonality otherwise raises an alarm every few hours.
    """
    values = as_series(series)
    if threshold <= 0.0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if values.size <= min_size:
        raise ValueError(f"series has {values.size} observations; min_size {min_size} needs more")
    sigma = float(np.std(values))
    if sigma == 0.0:
        raise ValueError("CUSUM needs a non-constant series (standard deviation is 0)")
    centre = float(np.median(values))
    allowance = ALLOWANCE_SIGMAS * sigma
    limit = threshold * sigma
    candidates = _alarms(values - centre - allowance, limit) + _alarms(
        -(values - centre) - allowance, limit
    )
    return ChangePointResult(
        method=ChangePointMethod.CUSUM,
        change_indices=_space_out(candidates, min_size, values.size),
        n=int(values.size),
        min_size=min_size,
        threshold=float(threshold),
    )


def pelt_changepoints(series: ArrayLike, min_size: int, penalty: float) -> ChangePointResult:
    """PELT segmentation (``ruptures``, l2 cost) with an explicit penalty."""
    values = as_series(series)
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if penalty <= 0.0:
        raise ValueError(f"penalty must be > 0, got {penalty}")
    try:
        import ruptures
    except ImportError:
        logger.exception("PELT is unavailable: ruptures is not installed")
        raise ImportError(_PELT_INSTALL_HINT) from None
    algorithm: Any = ruptures.Pelt(model="l2", min_size=min_size).fit(values)
    # ruptures reports segment ends (exclusive); the last one is len(values).
    breaks = [int(breakpoint) for breakpoint in algorithm.predict(pen=penalty)]
    return ChangePointResult(
        method=ChangePointMethod.PELT,
        change_indices=[index for index in breaks if index < values.size],
        n=int(values.size),
        min_size=min_size,
        penalty=float(penalty),
    )


def _alarms(drift: np.ndarray, limit: float) -> list[int]:
    """CUSUM alarm change indices for one direction, one per excursion block."""
    cumulative = np.cumsum(drift)
    excursion = cumulative - np.minimum.accumulate(cumulative)
    alarms = np.flatnonzero(excursion > limit)
    if alarms.size == 0:
        return []
    block_starts = np.concatenate(([0], np.flatnonzero(np.diff(alarms) > 1) + 1))
    return [int(np.argmin(cumulative[: alarms[start] + 1])) + 1 for start in block_starts]


def _space_out(candidates: list[int], min_size: int, n: int) -> list[int]:
    """Keep change indices inside the series, at least ``min_size`` apart."""
    kept: list[int] = []
    for index in sorted(candidates):
        if index < min_size or index >= n:
            continue
        if kept and index - kept[-1] < min_size:
            continue
        kept.append(index)
    return kept
