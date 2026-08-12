"""Instantiate model from registry and fit."""

from __future__ import annotations

import time

from typing import Any

import pandas as pd

from broadway.training.models.registry import get_model


def train(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, **params: float | int | str) -> tuple[Any, float]:
    model = get_model(model_type, **params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed
