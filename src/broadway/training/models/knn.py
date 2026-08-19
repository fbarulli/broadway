"""KNeighborsRegressor wrapper (fare regression)."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor


def create(**params: float | str) -> KNeighborsRegressor:
    return KNeighborsRegressor(**{k: v for k, v in params.items() if k != "type"})
