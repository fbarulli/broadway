"""LGBMRegressor / LGBMClassifier wrapper."""

from __future__ import annotations

from lightgbm import LGBMRegressor


def create(**params: float | int | str) -> LGBMRegressor:
    return LGBMRegressor(**{k: v for k, v in params.items() if k != "type"})
