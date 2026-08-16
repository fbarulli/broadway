"""Recursive feature elimination curve helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV


def rfe_curve(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: str = "neg_mean_absolute_error",
    cv: int = 3,
    min_features: int = 1,
) -> dict[str, list[int] | list[float]]:
    """Return RFECV feature-count and score curves for `model` on `X`, `y`.

    The score is the negated mean cross-validated test score, i.e. a positive
    error metric for regression (lower is better).
    """
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError(
            "X must contain only numeric columns; found non-numeric dtypes"
        )
    selector = RFECV(
        model,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features,
    ).fit(X, y)
    return {
        "n_features": [int(v) for v in selector.cv_results_["n_features"]],
        "score": list(-selector.cv_results_["mean_test_score"]),
    }
