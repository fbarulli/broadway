"""LGBMRegressor and the quantile-bundle estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

QUANTILE_MID = 0.5
# Params the bundle derives per sub-model; a caller's value would be a silent
# override of the key's contract, so it never reaches the sub-models.
_BUNDLE_OWNED_PARAMS = frozenset({"objective", "alpha"})


def create(**params: float | str) -> LGBMRegressor:
    return LGBMRegressor(**params)  # type: ignore[arg-type]


def create_quantile(**params: float | str) -> QuantileLGBMRegressor:
    return QuantileLGBMRegressor(**params)  # type: ignore[arg-type]


class QuantileLGBMRegressor:
    """Three LightGBM quantile regressors fitted at one interval level.

    ``alpha`` is the interval miscoverage: ``alpha=0.05`` fits the 2.5%, 50%,
    and 97.5% quantiles. ``predict`` returns the median; ``predict_interval``
    returns the ordered ``(lower, mid, upper)`` triple for the interval owner.
    """

    def __init__(self, alpha: float, **params: float | str) -> None:
        self.alpha = _require_alpha(alpha)
        self._params: dict[str, Any] = dict(params)
        self._models: list[LGBMRegressor] | None = None

    @property
    def quantiles(self) -> tuple[float, float, float]:
        """The fitted quantiles: lower tail, median, upper tail."""
        tail = self.alpha / 2.0
        return (tail, QUANTILE_MID, 1.0 - tail)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        # ``deep`` is sklearn's contract; the sub-models carry no tunable
        # registry params, so the flat dict is the whole parameter set.
        params = {"alpha": self.alpha, **self._params}
        return dict(params) if deep else params

    def set_params(self, **params: Any) -> QuantileLGBMRegressor:
        alpha = params.pop("alpha", self.alpha)
        self.alpha = _require_alpha(alpha)
        self._params.update(params)
        self._models = None
        return self

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, **fit_params: Any
    ) -> QuantileLGBMRegressor:
        base = {
            key: value
            for key, value in self._params.items()
            if key not in _BUNDLE_OWNED_PARAMS
        }
        self._models = [
            LGBMRegressor(objective="quantile", alpha=quantile, **base).fit(
                X, y, **fit_params
            )
            for quantile in self.quantiles
        ]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._fitted()[1].predict(X), dtype=float)

    def predict_interval(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower, mid, upper = (
            np.asarray(model.predict(X), dtype=float) for model in self._fitted()
        )
        return lower, mid, upper

    def _fitted(self) -> list[LGBMRegressor]:
        if self._models is None:
            raise ValueError("QuantileLGBMRegressor is not fitted — call fit first")
        return self._models


def _require_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"quantile alpha (miscoverage) must be in (0, 1), got {alpha}")
    return value
