from __future__ import annotations

import numpy as np
import pytest

from broadway.causal.analysis import analyze_two_groups
from broadway.causal.assignment import assign_randomly, assign_stratified
from broadway.causal.contracts import ExperimentDesign, ExperimentResult
from broadway.causal.design import design_experiment, minimum_detectable_effect
from broadway.causal.multiple import correct_pvalues


def test_design_experiment_returns_valid_design() -> None:
    design = design_experiment(
        effect_size=0.5,
        power=0.8,
        alpha=0.05,
        treatment_column="treatment",
        outcome_column="price",
    )
    assert isinstance(design, ExperimentDesign)
    assert design.sample_size > 0
    assert design.mde > 0
    assert design.treatment_column == "treatment"
    assert design.outcome_column == "price"


def test_design_larger_effect_size_needs_smaller_sample() -> None:
    small_effect = design_experiment(0.2, 0.8, 0.05, "t", "o")
    large_effect = design_experiment(0.8, 0.8, 0.05, "t", "o")
    assert large_effect.sample_size < small_effect.sample_size


def test_minimum_detectable_effect_positive() -> None:
    mde = minimum_detectable_effect(100, 0.8, 0.05)
    assert mde > 0


def test_assign_randomly_exact_treatment_count() -> None:
    flags = assign_randomly(100, 30, random_state=42)
    assert flags.sum() == 30
    assert set(np.unique(flags)) <= {0, 1}


def test_assign_randomly_deterministic() -> None:
    first = assign_randomly(100, 30, random_state=42)
    second = assign_randomly(100, 30, random_state=42)
    assert np.array_equal(first, second)


def test_assign_randomly_invalid_count_raises() -> None:
    with pytest.raises(ValueError):
        assign_randomly(10, 11, random_state=0)


def test_assign_stratified_preserves_strata_and_count() -> None:
    strata = np.repeat(["a", "b", "c"], [40, 30, 30])
    flags = assign_stratified(strata, 50, random_state=7)
    assert flags.sum() == 50
    assert set(np.unique(flags)) <= {0, 1}
    for label in ("a", "b", "c"):
        stratum_flags = flags[strata == label]
        assert 0 < stratum_flags.sum() < stratum_flags.size


def test_analyze_two_groups_detects_effect() -> None:
    rng = np.random.default_rng(42)
    treated = rng.normal(1.0, 1.0, 200)
    control = rng.normal(0.0, 1.0, 200)
    result = analyze_two_groups(treated, control, alpha=0.05, small_group_threshold=30)
    assert isinstance(result, ExperimentResult)
    assert result.passed is True
    assert result.effect_size > 0
    assert result.ci_lower < result.ci_upper


def test_analyze_two_groups_identical_fails() -> None:
    rng = np.random.default_rng(42)
    group = rng.normal(0.0, 1.0, 200)
    result = analyze_two_groups(group, group, alpha=0.05, small_group_threshold=30)
    assert result.passed is False
    assert result.effect_size == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_correct_pvalues_bonferroni() -> None:
    pvalues = [0.01, 0.04, 0.20, 0.50]
    corrected = correct_pvalues(pvalues, "bonferroni")
    assert len(corrected) == len(pvalues)
    assert all(0.0 <= p <= 1.0 for p in corrected)
    assert corrected[0] == pytest.approx(0.04)


def test_correct_pvalues_fdr_bh() -> None:
    pvalues = [0.01, 0.04, 0.20, 0.50]
    corrected = correct_pvalues(pvalues, "fdr_bh")
    assert len(corrected) == len(pvalues)
    assert all(0.0 <= p <= 1.0 for p in corrected)


def test_correct_pvalues_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        correct_pvalues([0.01, 0.04], "bogus")


def test_experiment_design_json_roundtrip() -> None:
    design = design_experiment(0.5, 0.8, 0.05, "treatment", "price")
    restored = ExperimentDesign.model_validate_json(design.model_dump_json())
    assert restored == design


def test_experiment_result_json_roundtrip() -> None:
    rng = np.random.default_rng(42)
    result = analyze_two_groups(
        rng.normal(1.0, 1.0, 100),
        rng.normal(0.0, 1.0, 100),
        alpha=0.05,
        small_group_threshold=30,
    )
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored == result
