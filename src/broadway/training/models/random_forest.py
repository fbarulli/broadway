"""sklearn RandomForestRegressor / RandomForestClassifier wrapper."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def create(**params: float | str) -> RandomForestRegressor:
    return RandomForestRegressor(**params)
