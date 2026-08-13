from __future__ import annotations

import numpy as np
import pytest

from broadway.stats.anova import run_anova, run_kruskal, run_welch


def _groups() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "A": rng.normal(0.0, 1.0, 40),
        "B": rng.normal(1.0, 1.0, 40),
        "C": rng.normal(2.0, 1.0, 40),
    }


def test_run_anova_passes_and_has_effect_sizes() -> None:
    plan = run_anova(_groups())
    assert plan.analysis_type == "group_comparison"
    assert plan.passed is True
    assert "eta_squared" in plan.effect_sizes
    assert "omega_squared" in plan.effect_sizes
    assert plan.effect_sizes["eta_squared"] > 0
    assert plan.effect_sizes["omega_squared"] > 0
    assert plan.next_step == "posthoc"
    assert "statistic" in plan.statistics
    assert "p_value" in plan.statistics
    assert plan.threshold_context["imbalance_ratio"] == pytest.approx(1.0)
    assert plan.threshold_context["any_small_group"] is False


def test_run_anova_not_passed_when_means_equal() -> None:
    rng = np.random.default_rng(7)
    groups = {g: rng.normal(0.0, 1.0, 40) for g in "ABC"}
    plan = run_anova(groups)
    assert plan.passed is False
    assert plan.next_step is None


def test_run_anova_requires_two_groups() -> None:
    with pytest.raises(ValueError):
        run_anova({"a": np.array([1.0, 2.0, 3.0])})


def test_run_welch_valid_plan() -> None:
    plan = run_welch(_groups())
    assert plan.analysis_type == "group_comparison"
    assert plan.test_name == "Welch's ANOVA"
    assert "statistic" in plan.statistics
    assert "p_value" in plan.statistics
    assert "eta_squared" in plan.effect_sizes
    assert "omega_squared" in plan.effect_sizes
    assert isinstance(plan.passed, bool)


def test_run_kruskal_valid_plan() -> None:
    plan = run_kruskal(_groups())
    assert plan.analysis_type == "group_comparison"
    assert plan.test_name == "Kruskal-Wallis"
    assert plan.passed is True
    assert "statistic" in plan.statistics
    assert "p_value" in plan.statistics
    assert "eta_squared" in plan.effect_sizes
    assert "omega_squared" in plan.effect_sizes


def test_anova_empty_group_raises() -> None:
    with pytest.raises(ValueError):
        run_anova({"a": np.array([]), "b": np.array([1.0, 2.0, 3.0])})


def test_anova_zero_variance_group_warns() -> None:
    plan = run_anova(
        {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([1.0, 2.0, 3.0, 4.0])}
    )
    assert any("zero variance" in w for w in plan.warnings)
    assert isinstance(plan.passed, bool)


def test_anova_all_zero_variance_raises() -> None:
    with pytest.raises(ValueError):
        run_anova({"a": np.array([1.0, 1.0, 1.0]), "b": np.array([2.0, 2.0, 2.0])})


def test_anova_non_finite_raises() -> None:
    with pytest.raises(ValueError):
        run_anova({"a": np.array([1.0, np.nan, 3.0]), "b": np.array([1.0, 2.0, 3.0])})


def test_welch_zero_variance_raises() -> None:
    with pytest.raises(ValueError):
        run_welch({"a": np.array([1.0, 1.0, 1.0]), "b": np.array([1.0, 2.0, 3.0])})
