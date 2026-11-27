"""Repeated shuffled K-fold CV for cross-sectional (row-shuffled) fitting.

A time split gets the forward-chaining geometry in ``evaluate.validation``;
this owner is the other case: rows are exchangeable survey respondents, so a
shuffled KFold repeated with independent seeds is the honest estimator. Every
candidate is scored on the *same* folds (same seed, same n), and the paired
improvement interval is measured on the out-of-fold rows by the canonical
``MetricBootstrap`` — never on the (correlated) fold scores.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, KFold

from broadway.config.schema import CVGeometry
from broadway.evaluate.contracts import BootstrapConfig
from broadway.evaluate.metrics import compute_metrics


class FoldScore(BaseModel):
    """One fold's measured metrics, with the rows it was scored on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repeat: int
    fold: int
    sample_size: int
    values: dict[str, float]


class CVResult(BaseModel):
    """Every fold score of one candidate plus its out-of-fold predictions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    metric: str
    secondary_metric: str
    geometry: CVGeometry
    grouping: str
    folds: list[FoldScore]
    out_of_fold: dict[int, list[float]] = Field(exclude=True, repr=False)

    @property
    def mean(self) -> float:
        return float(np.mean([fold.values[self.metric] for fold in self.folds]))

    @property
    def secondary_mean(self) -> float:
        return float(np.mean([fold.values[self.secondary_metric] for fold in self.folds]))

    @property
    def standard_error(self) -> float:
        """One standard error of the mean fold score (the one-SE selection unit)."""
        scores = np.asarray([fold.values[self.metric] for fold in self.folds], dtype=float)
        return float(np.std(scores, ddof=1) / np.sqrt(scores.size))

    def interval(self, bootstrap: BootstrapConfig) -> tuple[float, float]:
        """Fold-score interval for the candidate's own CV score (optimistic)."""
        draws = self._bootstrap(
            np.asarray([fold.values[self.metric] for fold in self.folds], dtype=float),
            bootstrap,
        )
        tail = (1.0 - bootstrap.level) / 2.0
        return (
            float(np.percentile(draws, tail * 100.0)),
            float(np.percentile(draws, (1.0 - tail) * 100.0)),
        )

    def out_of_fold_prediction(self) -> np.ndarray:
        """Row-level OOF prediction, averaged over the independent repeats."""
        return np.mean(
            np.asarray([self.out_of_fold[repeat] for repeat in sorted(self.out_of_fold)]),
            axis=0,
        )

    @staticmethod
    def _bootstrap(values: np.ndarray, bootstrap: BootstrapConfig) -> np.ndarray:
        generator = np.random.default_rng(bootstrap.seed)
        draws = np.empty(bootstrap.iterations)
        for index in range(bootstrap.iterations):
            sample = generator.integers(0, values.size, size=values.size)
            draws[index] = float(np.mean(values[sample]))
        return draws


class RepeatedCV:
    """Score one model factory over ``repeats x folds`` shuffled partitions.

    ``groups`` turns the shuffled KFold into a grouped split: every group (a
    person, a household) stays wholly inside one fold, so a repeat buyer cannot
    appear in both the fit and the scored rows. Repeats stay independent because
    each repeat permutes the row order with its own seed before the grouped
    assignment (GroupKFold assigns groups in first-appearance order).
    """

    def __init__(self, geometry: CVGeometry, metric: str, secondary_metric: str) -> None:
        self._geometry = geometry
        self._metric = metric
        self._secondary = secondary_metric

    def run(
        self,
        model: str,
        factory: Callable[[], BaseEstimator],
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None = None,
    ) -> CVResult:
        truth = y.to_numpy(dtype=float)
        group_values = None if groups is None else np.asarray(groups)
        if group_values is not None and len(group_values) != len(truth):
            raise ValueError(
                f"groups carries {len(group_values)} rows but the frame has {len(truth)}"
            )
        scores: list[FoldScore] = []
        out_of_fold: dict[int, list[float]] = {}
        for repeat in range(self._geometry.repeats):
            order = self._order(repeat, group_values, len(truth))
            splitter = self._splitter(group_values)
            fold_groups = None if group_values is None else group_values[order]
            prediction = np.full(truth.shape, np.nan)
            for fold, (train_position, test_position) in enumerate(
                splitter.split(X, truth, fold_groups)
            ):
                train_index, test_index = order[train_position], order[test_position]
                estimator = factory()
                estimator.fit(X.iloc[train_index], truth[train_index])
                fold_prediction = np.asarray(
                    estimator.predict(X.iloc[test_index]), dtype=float
                )
                prediction[test_index] = fold_prediction
                metrics = compute_metrics(
                    truth[test_index],
                    fold_prediction,
                    metrics=(self._metric, self._secondary),
                )
                scores.append(
                    FoldScore(
                        repeat=repeat,
                        fold=fold,
                        sample_size=len(test_index),
                        values=metrics,
                    )
                )
            out_of_fold[repeat] = prediction.tolist()
        if any(np.isnan(values).any() for values in (np.asarray(p) for p in out_of_fold.values())):
            raise ValueError(f"repeated CV left rows unscored for model '{model}'")
        return CVResult(
            model=model,
            metric=self._metric,
            secondary_metric=self._secondary,
            geometry=self._geometry,
            grouping=self._grouping(groups),
            folds=scores,
            out_of_fold=out_of_fold,
        )

    def _order(
        self, repeat: int, group_values: np.ndarray | None, row_count: int
    ) -> np.ndarray:
        """Row order for this repeat: identity ungrouped, a seeded permutation grouped."""
        if group_values is None:
            return np.arange(row_count)
        generator = np.random.default_rng(self._geometry.random_state + repeat)
        return generator.permutation(row_count)

    def _splitter(self, group_values: np.ndarray | None) -> KFold | GroupKFold:
        if group_values is not None:
            return GroupKFold(n_splits=self._geometry.folds)
        return KFold(
            n_splits=self._geometry.folds,
            shuffle=True,
            random_state=self._geometry.random_state,
        )

    @staticmethod
    def _grouping(groups: pd.Series | None) -> str:
        if groups is None:
            return "ungrouped shuffled KFold"
        return (
            f"grouped by '{groups.name}': GroupKFold, no group spans a "
            "train/validation fold"
        )


class NaiveMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the fitted training-mean for every row — the naive reference."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveMeanRegressor:
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class NaivePrevalencePredictor(BaseEstimator, RegressorMixin):
    """Predict the fitted training prevalence for every row — the classification no-skill reference.

    A constant score makes PR-AUC equal the prevalence itself, which is the
    no-skill value a classification candidate must beat. It uses the same
    ``fit``/``predict`` contract as the registry models so selection scores it on
    the identical folds.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaivePrevalencePredictor:
        labels = np.asarray(y, dtype=float)
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError(
                "the naive prevalence reference needs a binary 0/1 target, got "
                "values outside {0, 1}"
            )
        self.prevalence_ = float(np.mean(labels))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.prevalence_)
