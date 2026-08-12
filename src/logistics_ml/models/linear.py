from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import BaseModel


class LinearModel(BaseModel):

    name = "linear"

    def __init__(self, **params):

        super().__init__(**params)

        self.model = LinearRegression(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):

        if hasattr(self.model, "coef_"):
            return pd.Series(self.model.coef_)

        return None
