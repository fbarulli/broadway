"""LGBMRegressor / LGBMClassifier wrapper."""

from __future__ import annotations

from lightgbm import LGBMRegressor


def create(**params: float | str) -> LGBMRegressor:
    return LGBMRegressor(**params)  # type: ignore[arg-type]
