"""Tests for broadway.trust.drift (STATE-20260921-005).

Covers PSI/KS/KL scoring, baseline round-trip, typed-input validation, and
the no-hardcoded-thresholds contract (thresholds are caller params only).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from broadway.trust.drift import (
    BaselineFeature,
    BaselineStats,
    binned_ks,
    check_drift,
    feature_drift,
    fit_baseline,
    fit_baseline_feature,
    kl_divergence,
    psi,
    serve_probs,
)

RNG = np.random.default_rng(20260921)


def _train_frame(n: int = 2000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fare": RNG.normal(15.0, 5.0, n),
            "distance": RNG.exponential(3.0, n),
        }
    )


def test_psi_known_value() -> None:
    # Hand-computed: e=(0.5,0.5), a=(0.75,0.25):
    # 0.25*ln(1.5) + (-0.25)*ln(0.5) ~= 0.2747
    assert psi([0.5, 0.5], [0.75, 0.25]) == pytest.approx(0.2747, abs=1e-4)


def test_psi_identical_is_zero() -> None:
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_rejects_mismatched_or_empty() -> None:
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        psi([], [])


def test_kl_identical_is_zero_and_known_value() -> None:
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # Hand-computed KL([0.5,0.5] || [0.75,0.25]) ~= 0.1438
    assert kl_divergence([0.5, 0.5], [0.75, 0.25]) == pytest.approx(0.1438, abs=1e-4)
    # Asymmetric: p with mass where q has none diverges more.
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) > kl_divergence([1.0, 0.0], [0.5, 0.5])


def test_binned_ks_identical_is_zero_shifted_is_large() -> None:
    assert binned_ks([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert binned_ks([0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]) == pytest.approx(1.0)


def test_fit_round_trip_no_drift() -> None:
    train = _train_frame()
    baseline = fit_baseline(train, ["fare", "distance"], bins=10)
    report = check_drift(baseline, train)
    for name, result in report.results.items():
        assert result.drifted is None  # no thresholds -> no decision
        assert result.psi == pytest.approx(0.0, abs=1e-9)
        assert result.ks == pytest.approx(0.0, abs=1e-9)
        assert result.kl == pytest.approx(0.0, abs=1e-9)
        assert result.n_serve == len(train)
        assert result.feature == name


def test_shifted_serve_flags_drift_with_caller_thresholds() -> None:
    train = _train_frame()
    baseline = fit_baseline(train, ["fare", "distance"], bins=10)
    serve = train.copy()
    serve["fare"] = serve["fare"] + 20.0  # hard shift, far outside baseline range
    report = check_drift(baseline, serve, psi_threshold=0.25, ks_threshold=0.1, kl_threshold=0.25)
    assert report.results["fare"].drifted is True
    assert report.results["fare"].psi > 0.25
    assert report.results["fare"].ks > 0.1
    assert report.results["distance"].drifted is False


def test_partial_thresholds_decide_on_subset_only() -> None:
    train = _train_frame()
    baseline = fit_baseline(train, ["fare"], bins=10)
    report = check_drift(baseline, train, psi_threshold=999.0)
    assert report.results["fare"].drifted is False


def test_no_hardcoded_thresholds_in_signature() -> None:
    params = inspect.signature(check_drift).parameters
    for name in ("psi_threshold", "ks_threshold", "kl_threshold"):
        assert params[name].default is None


def test_out_of_range_serve_clipped_not_dropped() -> None:
    feat = BaselineFeature(bin_edges=[0.0, 1.0, 2.0], counts=[50, 50])
    probs = serve_probs(feat, pd.Series([100.0, 101.0, 102.0]))
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] == pytest.approx(1.0)  # clipped into the top edge bin


def test_feature_drift_scores_present() -> None:
    train = _train_frame()
    feat = fit_baseline_feature(train["fare"], bins=8)
    result = feature_drift(feat, train["fare"], feature="fare")
    assert result.psi >= 0.0 and result.ks >= 0.0 and result.kl >= 0.0
    assert result.n_baseline == len(train)


def test_missing_column_raises() -> None:
    baseline = fit_baseline(_train_frame(), ["fare"], bins=5)
    with pytest.raises(ValueError, match="missing from serve frame"):
        check_drift(baseline, pd.DataFrame({"other": [1.0, 2.0]}))


def test_empty_and_nonnumeric_serve_raise() -> None:
    feat = fit_baseline_feature(pd.Series([1.0, 2.0, 3.0]), bins=2)
    with pytest.raises(ValueError, match="empty"):
        serve_probs(feat, pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="numeric"):
        serve_probs(feat, pd.Series(["a", "b"]))


def test_baseline_model_validation() -> None:
    with pytest.raises(ValidationError):
        BaselineFeature(bin_edges=[1.0, 0.0], counts=[5])  # not increasing
    with pytest.raises(ValidationError):
        BaselineFeature(bin_edges=[0.0, 1.0], counts=[0])  # zero total
    with pytest.raises(ValueError, match="len\\(counts\\)"):
        BaselineFeature(bin_edges=[0.0, 1.0, 2.0], counts=[5])
    with pytest.raises(ValidationError):
        BaselineStats(features={})  # non-empty baseline required


def test_fit_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_baseline_feature(pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="missing from train frame"):
        fit_baseline(_train_frame(10), ["nope"])
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        check_drift(fit_baseline(_train_frame(10), ["fare"]), [[1.0]])  # type: ignore[arg-type]


def test_valid_ndarray_and_sequence_inputs() -> None:
    feat_nd = fit_baseline_feature(np.array([1.0, 2.0, 3.0]), bins=2)
    assert len(feat_nd.counts) == 2
    feat_seq = fit_baseline_feature([1.0, 2.0, 3.0], bins=2)
    assert len(feat_seq.counts) == 2


def test_as_finite_array_rejects_input_classes() -> None:
    with pytest.raises(ValueError, match="numeric"):
        fit_baseline_feature(np.array(["a", "b"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="array-like"):
        fit_baseline_feature({"a": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        fit_baseline_feature(["a", "b"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="array-like"):
        fit_baseline_feature(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-dimensional"):
        fit_baseline_feature(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="finite"):
        fit_baseline_feature(pd.Series([1.0, float("inf")]))


def test_baseline_validators_nonfinite_and_negative() -> None:
    with pytest.raises(ValidationError):
        BaselineFeature(bin_edges=[0.0, float("inf")], counts=[5])
    with pytest.raises(ValidationError):
        BaselineFeature(bin_edges=[0.0, 1.0, 2.0], counts=[5, -1])


def test_fit_baseline_rejects_non_dataframe() -> None:
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        fit_baseline([[1.0]], ["fare"])  # type: ignore[arg-type]


def test_psi_rejects_negative_probs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        psi([0.5, 0.5], [0.5, -0.1])


def test_kl_validation_branches() -> None:
    with pytest.raises(ValueError, match="must match"):
        kl_divergence([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        kl_divergence([], [])
    with pytest.raises(ValueError, match="non-negative"):
        kl_divergence([0.5, 0.5], [0.5, -0.1])


def test_binned_ks_validation_branches() -> None:
    with pytest.raises(ValueError, match="must match"):
        binned_ks([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        binned_ks([], [])
