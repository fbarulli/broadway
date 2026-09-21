"""Train vs serve distribution drift: PSI, binned-KS, KL.

Minimal re-introduction of the trust pillar (STATE-20260921-005). Honesty
framing per agents/notes/FEEDBACK.md #1 and agents/ledger/INVENTORY.md: the
trust package was deleted as stubs (D11); only this drift submodule is built,
against a Pydantic baseline model defined here. The sibling trust submodules
(fairness, leakage, interpretability, sensitivity, uncertainty) and
``monitoring/`` remain unbuilt — structurally reserved, not implemented.

Baseline model: per-feature binned train distribution (``BaselineStats``).
Drift compares a serve frame against those bins with the same edges, so PSI /
KS / KL are all computed on a shared support. Serve values outside the
baseline range are clipped into the edge bins, surfacing as edge-bin mass
shift rather than being silently dropped (``numpy.histogram`` default).

KS note: with only binned baseline counts stored, the KS statistic is the max
absolute difference of the binned CDFs — an approximation of the exact
two-sample KS, exact only up to bin resolution. Documented here so a wrong
p-value is never mistaken for an exact test (FEEDBACK #5).

Thresholds: this module ships NO hardcoded thresholds. Every decision
boundary (``psi_threshold``, ``ks_threshold``, ``kl_threshold``) is a caller
parameter defaulting to ``None`` (no decision). ``epsilon`` is numerical
smoothing only, not a decision boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

type ServeValues = pd.Series | np.ndarray | Sequence[float]


class BaselineFeature(BaseModel):
    """Binned train distribution for one numeric feature."""

    bin_edges: list[float] = Field(min_length=2)
    counts: list[int]

    @field_validator("bin_edges")
    @classmethod
    def _edges_increasing(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(e) for e in v):
            raise ValueError("bin_edges must be finite")
        if any(b <= a for a, b in pairwise(v)):
            raise ValueError("bin_edges must be strictly increasing")
        return v

    @field_validator("counts")
    @classmethod
    def _counts_non_negative(cls, v: list[int]) -> list[int]:
        if any(c < 0 for c in v):
            raise ValueError("counts must be non-negative")
        if sum(v) <= 0:
            raise ValueError("counts must sum to a positive total")
        return v

    def model_post_init(self, _context: object) -> None:
        if len(self.counts) != len(self.bin_edges) - 1:
            raise ValueError(
                f"len(counts)={len(self.counts)} must equal "
                f"len(bin_edges)-1={len(self.bin_edges) - 1}"
            )

    @property
    def probs(self) -> list[float]:
        total = sum(self.counts)
        return [c / total for c in self.counts]


class BaselineStats(BaseModel):
    """Train baseline: feature name -> binned distribution."""

    features: dict[str, BaselineFeature] = Field(min_length=1)


class FeatureDriftResult(BaseModel):
    """Drift scores for one feature (thresholds applied by the caller)."""

    feature: str
    psi: float
    ks: float
    kl: float
    n_baseline: int
    n_serve: int
    drifted: bool | None = None


class DriftReport(BaseModel):
    """Per-feature drift results for a serve frame."""

    results: dict[str, FeatureDriftResult]


def fit_baseline_feature(values: ServeValues, bins: int | Sequence[float] = 10) -> BaselineFeature:
    """Bin train values into a ``BaselineFeature``.

    ``bins`` follows ``numpy.histogram`` semantics (bin count or explicit
    edges). NaNs are dropped; empty input raises ``ValueError``.
    """
    arr = _as_finite_array(values, name="values")
    if arr.size == 0:
        raise ValueError("cannot fit baseline on empty values")
    counts, edges = np.histogram(arr, bins=bins)
    return BaselineFeature(bin_edges=[float(e) for e in edges], counts=[int(c) for c in counts])


def fit_baseline(
    df: pd.DataFrame, features: Sequence[str], bins: int | Sequence[float] = 10
) -> BaselineStats:
    """Fit per-feature baselines from a train frame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"features missing from train frame: {missing}")
    return BaselineStats(features={f: fit_baseline_feature(df[f], bins=bins) for f in features})


def psi(expected: Sequence[float], actual: Sequence[float], *, epsilon: float = 1e-4) -> float:
    """Population Stability Index between two probability vectors."""
    e = np.asarray([float(p) for p in expected], dtype=float)
    a = np.asarray([float(p) for p in actual], dtype=float)
    if e.shape != a.shape:
        raise ValueError(f"probability vectors must match: {e.shape} vs {a.shape}")
    if e.size == 0:
        raise ValueError("probability vectors must be non-empty")
    if bool((e < 0).any()) or bool((a < 0).any()):
        raise ValueError("probabilities must be non-negative")
    e = np.clip(e, epsilon, None)
    a = np.clip(a, epsilon, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(p: Sequence[float], q: Sequence[float], *, epsilon: float = 1e-4) -> float:
    """KL divergence KL(p || q) between two probability vectors (nats)."""
    pn = np.asarray([float(x) for x in p], dtype=float)
    qn = np.asarray([float(x) for x in q], dtype=float)
    if pn.shape != qn.shape:
        raise ValueError(f"probability vectors must match: {pn.shape} vs {qn.shape}")
    if pn.size == 0:
        raise ValueError("probability vectors must be non-empty")
    if bool((pn < 0).any()) or bool((qn < 0).any()):
        raise ValueError("probabilities must be non-negative")
    pn = np.clip(pn, epsilon, None)
    qn = np.clip(qn, epsilon, None)
    pn = pn / pn.sum()
    qn = qn / qn.sum()
    return float(np.sum(pn * np.log(pn / qn)))


def binned_ks(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Max absolute CDF difference between two binned distributions.

    Approximation of the two-sample KS statistic at bin resolution (see
    module docstring); no p-value is produced.
    """
    e = np.asarray([float(p) for p in expected], dtype=float)
    a = np.asarray([float(p) for p in actual], dtype=float)
    if e.shape != a.shape:
        raise ValueError(f"probability vectors must match: {e.shape} vs {a.shape}")
    if e.size == 0:
        raise ValueError("probability vectors must be non-empty")
    e = e / e.sum()
    a = a / a.sum()
    return float(np.max(np.abs(np.cumsum(a) - np.cumsum(e))))


def serve_probs(baseline: BaselineFeature, serve: ServeValues) -> list[float]:
    """Histogram serve values on the baseline bins; returns probabilities."""
    arr = _as_finite_array(serve, name="serve")
    if arr.size == 0:
        raise ValueError("serve values are empty")
    edges = np.asarray(baseline.bin_edges, dtype=float)
    clipped = np.clip(arr, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    total = int(counts.sum())
    return [float(c) / total for c in counts]


def feature_drift(
    baseline: BaselineFeature,
    serve: ServeValues,
    *,
    feature: str = "",
    epsilon: float = 1e-4,
) -> FeatureDriftResult:
    """Score one feature's serve values against its baseline bins."""
    actual = serve_probs(baseline, serve)
    expected = baseline.probs
    arr = _as_finite_array(serve, name="serve")
    return FeatureDriftResult(
        feature=feature,
        psi=psi(expected, actual, epsilon=epsilon),
        ks=binned_ks(expected, actual),
        kl=kl_divergence(actual, expected, epsilon=epsilon),
        n_baseline=sum(baseline.counts),
        n_serve=int(arr.size),
    )


def check_drift(
    baseline: BaselineStats,
    serve_df: pd.DataFrame,
    *,
    epsilon: float = 1e-4,
    psi_threshold: float | None = None,
    ks_threshold: float | None = None,
    kl_threshold: float | None = None,
) -> DriftReport:
    """Score every baseline feature against the serve frame.

    Thresholds are caller parameters only (all default ``None`` = no
    decision). A feature's ``drifted`` flag is ``None`` when no threshold is
    given, else ``True`` if any provided threshold is breached.
    """
    if not isinstance(serve_df, pd.DataFrame):
        raise ValueError(f"serve_df must be a pandas DataFrame, got {type(serve_df).__name__}")
    missing = [f for f in baseline.features if f not in serve_df.columns]
    if missing:
        raise ValueError(f"features missing from serve frame: {missing}")
    results: dict[str, FeatureDriftResult] = {}
    for name, feat in baseline.features.items():
        result = feature_drift(feat, serve_df[name], feature=name, epsilon=epsilon)
        scored = (
            (result.psi, psi_threshold),
            (result.ks, ks_threshold),
            (result.kl, kl_threshold),
        )
        given = [(score, threshold) for score, threshold in scored if threshold is not None]
        result.drifted = None if not given else bool(any(score > t for score, t in given))
        results[name] = result
    return DriftReport(results=results)


def _as_finite_array(values: ServeValues, *, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        try:
            arr = values.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    elif isinstance(values, np.ndarray):
        try:
            arr = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric, got dtype {values.dtype}") from exc
    elif isinstance(values, Mapping):
        raise ValueError(f"{name} must be array-like, got {type(values).__name__}")
    elif isinstance(values, Sequence):
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    else:
        raise ValueError(f"{name} must be array-like, got {type(values).__name__}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    arr = arr[~np.isnan(arr)]
    if arr.size > 0 and not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} must be finite")
    return arr
