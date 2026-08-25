from __future__ import annotations

import numpy as np
import pytest

from broadway.stats import assumptions as assumptions_mod
from broadway.stats.assumptions import check_normality, run_levene


def _groups() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "A": rng.normal(0.0, 1.0, 50),
        "B": rng.normal(1.0, 2.0, 50),
        "C": rng.normal(2.0, 1.0, 50),
    }


def test_run_levene_returns_statistic_and_p_value() -> None:
    result = run_levene(_groups())
    assert set(result.keys()) == {"statistic", "p_value"}
    assert isinstance(result["statistic"], float)
    assert 0.0 <= result["p_value"] <= 1.0


def test_check_normality_returns_per_group_stats() -> None:
    result = check_normality(_groups())
    assert set(result.keys()) == {"A", "B", "C"}
    for stats in result.values():
        assert set(stats.keys()) == {"skew", "kurtosis", "shapiro_p"}
        assert isinstance(stats["skew"], float)
        assert isinstance(stats["kurtosis"], float)
        assert 0.0 <= stats["shapiro_p"] <= 1.0


def test_check_normality_caps_subsample(monkeypatch: pytest.MonkeyPatch) -> None:
    group_size = 6000
    cap_default = 5000
    shapiro_sizes: list[int] = []
    skew_sizes: list[int] = []

    real_shapiro = assumptions_mod.stats.shapiro
    real_skew = assumptions_mod.stats.skew

    def spy_shapiro(values: np.ndarray) -> object:
        shapiro_sizes.append(len(values))
        return real_shapiro(values)

    def spy_skew(values: np.ndarray) -> object:
        skew_sizes.append(len(values))
        return real_skew(values)

    # The module imports `from scipy import stats`, so wrap the stats
    # module attributes it calls through and delegate to the originals.
    monkeypatch.setattr(assumptions_mod.stats, "shapiro", spy_shapiro)
    monkeypatch.setattr(assumptions_mod.stats, "skew", spy_skew)

    rng = np.random.default_rng(1)
    groups = {
        "big": rng.normal(0.0, 1.0, group_size),
        "other": rng.normal(0.0, 1.0, group_size),
    }

    result = check_normality(groups)

    # The frame is LARGER than the default cap, so no statistic may ever
    # see a full-size array: both spied calls must receive subsamples.
    assert shapiro_sizes, "shapiro was never invoked"
    assert skew_sizes, "skew was never invoked"
    assert max(shapiro_sizes) <= cap_default
    assert max(skew_sizes) <= cap_default
    assert max(shapiro_sizes) < group_size
    assert max(skew_sizes) < group_size
    assert 0.0 <= result["big"]["shapiro_p"] <= 1.0

    # An explicit tighter cap flows through the same path: both 200-row
    # groups must be handed to shapiro pre-trimmed to the cap.
    check_normality(
        {"g1": groups["big"][:200], "g2": groups["other"][:200]},
        shapiro_max_n=64,
    )
    assert shapiro_sizes[-1] == 64


def test_levene_zero_variance_raises() -> None:
    with pytest.raises(ValueError):
        run_levene({"a": np.array([1.0, 1.0, 1.0]), "b": np.array([1.0, 2.0, 3.0])})


def test_check_normality_empty_group_raises() -> None:
    with pytest.raises(ValueError):
        check_normality({"a": np.array([]), "b": np.array([1.0, 2.0, 3.0])})
