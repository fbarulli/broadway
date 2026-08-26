"""LinearRegression wrapper with sklearn."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression


def create(**params: float | str) -> LinearRegression:
    return LinearRegression(**params)
