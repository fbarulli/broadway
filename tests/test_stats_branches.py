"""Public edge/validation behavior of the stats and model-selection numerics."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from broadway.selection.information import compare_aic_bic
from broadway.stats.changepoint import cusum_changepoints, pelt_changepoints
from broadway.stats.decompose import DecomposeModel, decompose, lowess_trend


class _Fit:
    def __init__(self, aic: float, bic: float) -> None:
        self.aic = aic
        self.bic = bic


def test_decompose_rejects_a_too_short_period_or_series() -> None:
    with pytest.raises(ValueError, match="period must be"):
        decompose(np.arange(10, dtype=float), period=1, model="additive")
    with pytest.raises(ValueError, match="at least two full periods"):
        decompose(np.arange(10, dtype=float), period=24, model="additive")


def test_multiplicative_decompose_requires_positive_observations() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        decompose(np.zeros(48), period=24, model=DecomposeModel.MULTIPLICATIVE)


def test_lowess_trend_rejects_a_frac_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="frac must be in"):
        lowess_trend(np.arange(10, dtype=float), frac=0.0)
    with pytest.raises(ValueError, match="frac must be in"):
        lowess_trend(np.arange(10, dtype=float), frac=1.5)


def test_cusum_validates_its_parameters() -> None:
    series = np.random.default_rng(0).normal(0.0, 1.0, 100)

    with pytest.raises(ValueError, match="threshold must be"):
        cusum_changepoints(series, threshold=0.0, min_size=5)
    with pytest.raises(ValueError, match="min_size must be"):
        cusum_changepoints(series, threshold=1.0, min_size=0)
    with pytest.raises(ValueError, match="needs more"):
        cusum_changepoints(series[:5], threshold=1.0, min_size=5)
    with pytest.raises(ValueError, match="non-constant"):
        cusum_changepoints(np.ones(100), threshold=1.0, min_size=5)


def test_pelt_validates_its_parameters_before_importing_ruptures() -> None:
    series = np.arange(20, dtype=float)

    with pytest.raises(ValueError, match="min_size must be"):
        pelt_changepoints(series, min_size=0, penalty=10.0)
    with pytest.raises(ValueError, match="penalty must be"):
        pelt_changepoints(series, min_size=2, penalty=0.0)


def test_pelt_fails_loud_with_the_install_hint_when_ruptures_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ruptures", None)

    with pytest.raises(ImportError, match="uv pip install ruptures"):
        pelt_changepoints(np.arange(20, dtype=float), min_size=2, penalty=10.0)


def test_compare_aic_bic_rejects_no_candidates_or_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        compare_aic_bic({})
    with pytest.raises(ValueError, match="non-finite aic"):
        compare_aic_bic({"broken": _Fit(aic=float("nan"), bic=1.0)})


def test_compare_aic_bic_single_candidate_holds_all_weight() -> None:
    table = compare_aic_bic({"only": _Fit(aic=10.0, bic=12.0)})

    (candidate,) = table.candidates
    assert candidate.akaike_weight == pytest.approx(1.0)
    assert candidate.bic_weight == pytest.approx(1.0)
    assert (candidate.delta_aic, candidate.delta_bic) == (0.0, 0.0)
