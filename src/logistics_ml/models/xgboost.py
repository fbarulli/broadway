from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from .base import BaseModel


class XGBoostModel(BaseModel):

    name = "xgboost"

    def __init__(self, **params):

        defaults = dict(
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
        )

        defaults.update(params)

        super().__init__(**defaults)

        self.model = XGBRegressor(**defaults)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return pd.Series(
            self.model.feature_importances_
        )
