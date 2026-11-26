"""Coverage tolerance: the band shrinks with the holdout's own noise floor."""

from __future__ import annotations

import pytest

from broadway.calibration.metrics import coverage_tolerance


def test_coverage_tolerance_shrinks_with_n_and_respects_the_business_cap() -> None:
    small_n = coverage_tolerance(0.95, n=1_000, business_tolerance=0.05)
    large_n = coverage_tolerance(0.95, n=100_000, business_tolerance=0.05)

    assert large_n < small_n <= 0.05
    assert small_n == pytest.approx(min(0.05, 3.0 * (0.95 * 0.05 / 1_000) ** 0.5))
    assert large_n == pytest.approx(3.0 * (0.95 * 0.05 / 100_000) ** 0.5)


def test_coverage_tolerance_rejects_degenerate_inputs() -> None:
    import pytest

    from broadway.calibration.metrics import coverage_tolerance

    with pytest.raises(ValueError, match="at least one row"):
        coverage_tolerance(0.95, n=0, business_tolerance=0.05)
    with pytest.raises(ValueError, match="nominal coverage"):
        coverage_tolerance(1.0, n=100, business_tolerance=0.05)
    with pytest.raises(ValueError, match="non-negative"):
        coverage_tolerance(0.95, n=100, business_tolerance=-0.1)
