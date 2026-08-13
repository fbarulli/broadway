from __future__ import annotations

import math

from broadway.causal.design import design_experiment, minimum_detectable_effect


def test_sample_size_increases_with_power() -> None:
    assert design_experiment(0.5, 0.9, 0.05, "t", "o").sample_size > design_experiment(
        0.5, 0.8, 0.05, "t", "o"
    ).sample_size


def test_sample_size_increases_with_stricter_alpha() -> None:
    assert design_experiment(0.5, 0.8, 0.01, "t", "o").sample_size > design_experiment(
        0.5, 0.8, 0.05, "t", "o"
    ).sample_size


def test_sample_size_decreases_with_larger_effect() -> None:
    assert design_experiment(0.2, 0.8, 0.05, "t", "o").sample_size > design_experiment(
        0.8, 0.8, 0.05, "t", "o"
    ).sample_size


def test_mde_decreases_with_larger_sample() -> None:
    assert minimum_detectable_effect(1000, 0.8, 0.05) < minimum_detectable_effect(
        100, 0.8, 0.05
    )


def test_mde_finite_positive_for_small_sample() -> None:
    for n in (2, 5, 10):
        mde = minimum_detectable_effect(n, 0.8, 0.05)
        assert math.isfinite(mde)
        assert mde > 0


def test_extreme_effect_sizes_finite_positive_ordered() -> None:
    tiny = design_experiment(0.01, 0.8, 0.05, "t", "o")
    huge = design_experiment(2.0, 0.8, 0.05, "t", "o")
    assert math.isfinite(tiny.sample_size)
    assert math.isfinite(huge.sample_size)
    assert tiny.sample_size > 0
    assert huge.sample_size > 0
    assert huge.sample_size < tiny.sample_size


def test_high_power_finite() -> None:
    high = design_experiment(0.5, 0.99, 0.05, "t", "o")
    assert math.isfinite(high.sample_size)
    assert high.sample_size > 0
    assert high.sample_size > design_experiment(0.5, 0.8, 0.05, "t", "o").sample_size
