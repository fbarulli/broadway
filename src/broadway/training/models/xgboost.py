"""XGBRegressor / XGBClassifier wrapper."""

from __future__ import annotations

from xgboost import XGBRegressor


def create(**params: float | str) -> XGBRegressor:
    return XGBRegressor(**params)
