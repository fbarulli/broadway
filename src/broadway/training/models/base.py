"""Abstract base for trainable models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseModel(ABC):
    def __init__(self) -> None:
        self._params: dict[str, float | int | str] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """Fit the model to feature matrix X and target y."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Return predictions for feature matrix X."""

    @abstractmethod
    def feature_importance(self) -> dict[str, float]:
        """Return feature name -> importance mapping."""

    def get_params(self) -> dict[str, float | int | str]:
        return dict(self._params)

    def set_params(self, **params: float | int | str) -> None:
        self._params.update(params)
