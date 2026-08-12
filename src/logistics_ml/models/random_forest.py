from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel


class RandomForestModel(BaseModel):

    name = "random_forest"

    def __init__(self, **params):

        defaults = dict(
            random_state=42,
            n_estimators=100,
            max_depth=20,
            max_samples=0.3,
            n_jobs=-1,
        )

        defaults.update(params)

        super().__init__(**defaults)

        self.model = RandomForestRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return pd.Series(
            self.model.feature_importances_
        )
