"""Instantiate model from registry and fit."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from broadway.training.contracts import TrainingResult
from broadway.training.models.registry import get_model


def train(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, **params: float | str) -> tuple[Any, TrainingResult]:
    model = get_model(model_type, **params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, TrainingResult(model_type=model_type, params=params, train_time_seconds=round(elapsed, 3))
