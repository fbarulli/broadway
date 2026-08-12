from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from .base import BaseModel


class LightGBMModel(BaseModel):

    name = "lightgbm"

    def __init__(self, **params):

        defaults = dict(
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            n_jobs=-1,
        )

        defaults.update(params)

        super().__init__(**defaults)

        self.model = LGBMRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return pd.Series(
            self.model.booster_.feature_importance(
                importance_type="gain"
            )
        )
