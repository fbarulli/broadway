"""LightGBM baseline with holdout metrics including tail-error benchmark."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_lgbm(X: pd.DataFrame, y: np.ndarray, **params) -> object:
    return lgb.LGBMRegressor(**params).fit(X, y)


def evaluate(model, X: pd.DataFrame, y: np.ndarray, tail_quantile: float) -> dict:
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    tail_mask = y >= np.quantile(y, tail_quantile)
    tail_mae = mean_absolute_error(y[tail_mask], preds[tail_mask])

    return {"mae": mae, "rmse": rmse, "tail_mae": tail_mae}
