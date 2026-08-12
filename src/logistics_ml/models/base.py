from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class BaseModel(BaseEstimator, RegressorMixin, ABC):
    """
    Base class for every regression model in the platform.

    Compatible with:
        - scikit-learn
        - MLflow
        - Optuna
        - GridSearchCV
        - RandomizedSearchCV
    """

    name = "base"

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        ...

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ):
        ...

    @abstractmethod
    def feature_importance(self):
        ...

    def get_params(self, deep=True):
        return self.params.copy()

    def set_params(self, **params):
        self.params.update(params)
        return self
