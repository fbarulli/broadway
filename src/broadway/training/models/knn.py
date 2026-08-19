"""KNeighbors wrappers (regressor for regression tasks, classifier for classification)."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def create(**params: float | str) -> KNeighborsRegressor:
    return KNeighborsRegressor(**{k: v for k, v in params.items() if k != "type"})


def create_classifier(**params: float | str) -> KNeighborsClassifier:
    return KNeighborsClassifier(**{k: v for k, v in params.items() if k != "type"})
