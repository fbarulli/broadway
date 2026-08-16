"""LGBMRegressor / LGBMClassifier wrapper."""

from __future__ import annotations

from lightgbm import LGBMRegressor


def create(**params: float | str) -> LGBMRegressor:
    return LGBMRegressor(**{k: v for k, v in params.items() if k != "type"})  # type: ignore[arg-type]
