from __future__ import annotations

import numpy as np

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
    for name, stats in result.items():
        assert set(stats.keys()) == {"skew", "kurtosis", "shapiro_p"}
        assert isinstance(stats["skew"], float)
        assert isinstance(stats["kurtosis"], float)
        assert 0.0 <= stats["shapiro_p"] <= 1.0


def test_check_normality_caps_subsample() -> None:
    rng = np.random.default_rng(1)
    groups = {"big": rng.normal(0.0, 1.0, 6000)}
    result = check_normality(groups)
    assert "big" in result
    assert "shapiro_p" in result["big"]
