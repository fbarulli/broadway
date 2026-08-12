from __future__ import annotations

import numpy as np
import pytest

from broadway.stats.effect_size import (
    cohens_d,
    eta_squared,
    group_imbalance,
    hedges_g,
    omega_squared,
)


def test_eta_squared_hand_computed() -> None:
    assert eta_squared(4.0, 2, 30) == pytest.approx((2 * 4.0) / (2 * 4.0 + 30))


def test_omega_squared_hand_computed() -> None:
    assert omega_squared(4.0, 2, 30, 33) == pytest.approx(
        (2 * (4.0 - 1)) / (2 * (4.0 - 1) + 33)
    )


def test_omega_squared_less_than_eta_squared() -> None:
    f_stat, df1, df2, n_total = 4.0, 2, 30, 33
    omega = omega_squared(f_stat, df1, df2, n_total)
    eta = eta_squared(f_stat, df1, df2)
    assert omega < eta


def test_cohens_d_identical_mean_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0]) - 1.0
    assert cohens_d(a, b) == pytest.approx(0.0)


def test_cohens_d_sign_correct() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert cohens_d(a, b) < 0
    assert cohens_d(b, a) > 0


def test_hedges_g_smaller_magnitude_than_cohens_d() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    assert abs(g) < abs(d)


def test_group_imbalance_imbalanced() -> None:
    assert group_imbalance({"A": 100, "B": 10}) == pytest.approx(10.0)


def test_group_imbalance_balanced() -> None:
    assert group_imbalance({"A": 5, "B": 5}) == pytest.approx(1.0)


def test_group_imbalance_edge_cases() -> None:
    assert group_imbalance({"A": 10}) == 1.0
    assert group_imbalance({"A": 0, "B": 10}) == 1.0


def test_cohens_d_empty_array_returns_zero() -> None:
    assert cohens_d(np.array([]), np.array([1.0, 2.0])) == 0.0


def test_cohens_d_zero_variance_returns_zero() -> None:
    assert cohens_d(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])) == 0.0
