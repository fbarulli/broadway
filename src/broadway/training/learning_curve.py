"""Training-size learning curve: metric vs fraction of the training frame.

Answers "how little data do we need": refit the configured pipeline on
growing fractions of the training frame, measure the configured metric on the
validation frame, and report the smallest fraction whose mean stays within a
relative tolerance of the full-data result. The curve never mutates the frames
it is handed (every subset is a fresh frame) and never changes the final fit.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from broadway import viz
from broadway.config.schema import LearningCurveConfig, PipelineConfig, TaskType
from broadway.evaluate.metrics import binary_metrics, compute_metrics
from broadway.features.plan import ColumnPlan
from broadway.observability.context import current_run_id
from broadway.observability.profiling import phase

logger = logging.getLogger(__name__)

SplitType = Literal["time", "random", "stratified"]
Direction = Literal["minimize", "maximize"]

# Classification metrics that consume scores, not labels. ``binary_metrics``
# exposes only these today; a label-based metric added to it would be listed in
# _LABEL_METRICS and scored from predict() labels instead.
_SCORE_METRICS = frozenset({"roc_auc", "pr_auc"})
_LABEL_METRICS: frozenset[str] = frozenset()
_SUPPORTED_CLASSIFICATION_METRICS = _SCORE_METRICS | _LABEL_METRICS

_CSV_COLUMNS = (
    "fraction",
    "rows",
    "repeats",
    "failed_repeats",
    "mean",
    "std",
    "within_tolerance",
    "failure",
)


class LearningCurvePoint(BaseModel):
    """One fraction's curve point, aggregated over its repeats."""

    model_config = ConfigDict(extra="forbid")

    fraction: float
    rows: int
    repeats: int
    failed_repeats: int
    mean: float | None
    std: float | None
    values: list[float]
    failure: str | None = None
    within_tolerance: bool = False


class LearningCurveResult(BaseModel):
    """The curve table plus the minimum-data answer and its artifacts."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    split: SplitType
    target_metric: str
    direction: Direction
    tolerance: float
    absolute_tolerance: float
    metric_threshold: float | None
    points: list[LearningCurvePoint]
    reference: LearningCurvePoint
    min_fraction: float | None
    min_rows: int | None


class LearningCurve:
    """Fit the deployed pipeline across training-size fractions.

    Callers pass the slice the final model trains on (after calibration) and
    the resolved params (after HPO), so the 100% point is exactly the deployed
    model's training data. ``pipeline_factory`` must return a fresh, unfitted
    pipeline per call: each subset refits its own preprocessing (target
    encoding included), so no statistic leaks from the full frame into a
    small subset.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        config: LearningCurveConfig,
        pipeline_factory: Callable[[], Pipeline],
    ) -> None:
        if cfg.dataset is None or cfg.experiment is None:
            raise ValueError("learning curve requires dataset and experiment config")
        if (
            cfg.experiment.split.type == "stratified"
            and cfg.dataset.task is not TaskType.CLASSIFICATION
        ):
            raise ValueError(
                "stratified learning curve requires a classification task, "
                f"but dataset task is '{cfg.dataset.task.value}'"
            )
        if cfg.dataset.task is TaskType.CLASSIFICATION:
            metric = cfg.experiment.target_metric
            if metric not in _SUPPORTED_CLASSIFICATION_METRICS:
                raise ValueError(
                    f"target_metric '{metric}' is not supported for classification "
                    f"(supported: {sorted(_SUPPORTED_CLASSIFICATION_METRICS)})"
                )
            if config.metric_threshold is None:
                raise ValueError(
                    "classification learning curve requires "
                    "learning_curve.metric_threshold (the binary target split)"
                )
        self._cfg = cfg
        self._config = config
        self._pipeline_factory = pipeline_factory

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> LearningCurveResult:
        """Compute, write, and return the curve (profiled as one phase)."""
        with phase("learning_curve"):
            self._require_score_path()
            result = self._compute(train_df, val_df)
            self._write(result)
            return result

    def _compute(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> LearningCurveResult:
        assert self._cfg.dataset is not None and self._cfg.experiment is not None
        split = self._cfg.experiment.split.type
        metric = self._cfg.experiment.target_metric
        fractions = sorted(self._config.fractions)
        X_val, y_val = self._xy(val_df)
        if self._cfg.dataset.task is TaskType.REGRESSION:
            available = compute_metrics(y_val.to_numpy(), y_val.to_numpy())
            if metric not in available:
                raise ValueError(
                    f"target_metric '{metric}' is not a computed metric "
                    f"(available: {sorted(available)})"
                )
        raw_points = [
            self._point(fraction, train_df, X_val, y_val, split, metric) for fraction in fractions
        ]
        reference = next(point for point in raw_points if point.fraction == 1.0)
        points = [
            point.model_copy(update={"within_tolerance": self._within(point, reference)})
            for point in raw_points
        ]
        reference = next(point for point in points if point.fraction == 1.0)
        min_fraction, min_rows = self._minimum(points, reference)
        return LearningCurveResult(
            run_id=current_run_id(),
            split=split,
            target_metric=metric,
            direction=self._config.direction,
            tolerance=self._config.tolerance,
            absolute_tolerance=self._config.absolute_tolerance,
            metric_threshold=self._config.metric_threshold,
            points=points,
            reference=reference,
            min_fraction=min_fraction,
            min_rows=min_rows,
        )

    def _point(
        self,
        fraction: float,
        train_df: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        split: SplitType,
        metric: str,
    ) -> LearningCurvePoint:
        subsets = self._subsets(train_df, fraction, split)
        values: list[float] = []
        failures: list[str] = []
        rows = 0
        for subset in subsets:
            rows = len(subset)
            try:
                values.append(self._fit_metric(subset, X_val, y_val, metric))
            except Exception as exc:  # a tiny subset may be unfittable; record it
                logger.exception(
                    "learning curve: fraction %.4f (rows %d) failed to fit",
                    fraction,
                    rows,
                )
                failures.append(f"{type(exc).__name__}: {exc}")
        mean = float(np.mean(values)) if values else None
        std = float(np.std(values)) if values else None
        return LearningCurvePoint(
            fraction=fraction,
            rows=rows,
            repeats=len(subsets),
            failed_repeats=len(failures),
            mean=mean,
            std=std,
            values=values,
            failure=None if values else "; ".join(failures),
        )

    def _subsets(
        self, train_df: pd.DataFrame, fraction: float, split: SplitType
    ) -> list[pd.DataFrame]:
        """Subsets for one fraction: one chronological prefix, or N random draws."""
        k = max(1, math.ceil(len(train_df) * fraction))
        if split == "time":
            return [self._time_prefix(train_df, k)]
        assert self._cfg.experiment is not None
        assert self._cfg.dataset is not None
        base_seed = self._cfg.experiment.random_state
        # Classification prefers a stratified draw even on a "random" split so
        # a small fraction keeps every class; regression draws uniformly.
        stratified = self._cfg.dataset.task is TaskType.CLASSIFICATION
        return [
            self._random_subset(train_df, k, base_seed + repeat, stratified)
            for repeat in range(self._config.repeats)
        ]

    def _time_prefix(self, train_df: pd.DataFrame, k: int) -> pd.DataFrame:
        assert self._cfg.dataset is not None
        datetime_column = self._cfg.dataset.datetime_column
        if not datetime_column:
            raise ValueError(
                "time learning curve requires dataset.datetime_column "
                "(the prefix is the earliest history)"
            )
        ordered = train_df.sort_values(datetime_column, kind="mergesort")
        return ordered.iloc[:k]

    def _random_subset(
        self, train_df: pd.DataFrame, k: int, seed: int, stratified: bool
    ) -> pd.DataFrame:
        assert self._cfg.dataset is not None
        if k >= len(train_df):
            return train_df
        if stratified:
            try:
                subset, _ = train_test_split(
                    train_df,
                    train_size=k,
                    stratify=train_df[self._cfg.dataset.target],
                    random_state=seed,
                )
                return subset
            except ValueError:
                logger.warning(
                    "learning curve: stratified draw of %d rows unusable; "
                    "falling back to a random draw",
                    k,
                    exc_info=True,
                )
        return train_df.sample(n=k, random_state=seed)

    def _require_score_path(self) -> None:
        """Fail loud before fitting when a score metric meets a proba-less model."""
        assert self._cfg.dataset is not None and self._cfg.experiment is not None
        metric = self._cfg.experiment.target_metric
        if self._cfg.dataset.task is not TaskType.CLASSIFICATION:
            return
        if metric not in _SCORE_METRICS:
            return
        if not hasattr(self._pipeline_factory(), "predict_proba"):
            raise ValueError(
                f"target_metric '{metric}' requires positive-class scores, but "
                f"model '{self._cfg.experiment.model.type}' has no predict_proba"
            )

    def _fit_metric(
        self,
        subset: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str,
    ) -> float:
        X_sub, y_sub = self._xy(subset)
        pipeline = self._pipeline_factory()
        pipeline.fit(X_sub, y_sub)
        return self._score(pipeline, X_val, y_val.to_numpy(), metric)

    def _score(
        self, pipeline: Pipeline, X_val: pd.DataFrame, y_true: np.ndarray, metric: str
    ) -> float:
        """Same metric module as evaluate: regression or binary classification.

        Score metrics (AUC) take raw positive-class probabilities; label metrics
        take predict() labels. The threshold defines the positive target inside
        ``binary_metrics`` — it is never applied to the scores.
        """
        assert self._cfg.dataset is not None
        if self._cfg.dataset.task is TaskType.REGRESSION:
            return compute_metrics(y_true, pipeline.predict(X_val))[metric]
        threshold = self._config.metric_threshold
        assert threshold is not None
        if metric in _SCORE_METRICS:
            y_pred = self._positive_class_scores(pipeline, X_val, metric)
        else:
            y_pred = pipeline.predict(X_val)
        return binary_metrics(y_true, y_pred, threshold)[metric]

    def _positive_class_scores(
        self, pipeline: Pipeline, X_val: pd.DataFrame, metric: str
    ) -> np.ndarray:
        assert self._cfg.dataset is not None
        proba = np.asarray(pipeline.predict_proba(X_val))
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError(
                f"target_metric '{metric}' requires binary predict_proba "
                f"(positive class in column 1); got shape {proba.shape}"
            )
        return proba[:, 1]

    def _xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        assert self._cfg.dataset is not None and self._cfg.experiment is not None
        X, _ = ColumnPlan.from_config(self._cfg.dataset, self._cfg.experiment).apply(df)
        return X, df[self._cfg.dataset.target]

    def _within(self, point: LearningCurvePoint, reference: LearningCurvePoint) -> bool:
        if point.mean is None or reference.mean is None:
            return False
        # Relative band with an absolute floor, so a zero reference does not
        # collapse the band to exact equality.
        slack = max(
            self._config.tolerance * abs(reference.mean),
            self._config.absolute_tolerance,
        )
        if self._config.direction == "minimize":
            return point.mean <= reference.mean + slack
        return point.mean >= reference.mean - slack

    def _minimum(
        self, points: list[LearningCurvePoint], reference: LearningCurvePoint
    ) -> tuple[float | None, int | None]:
        if reference.mean is None:
            return None, None
        eligible = [point for point in points if point.within_tolerance]
        if not eligible:
            return None, None
        best = min(eligible, key=lambda point: point.fraction)
        return best.fraction, best.rows

    def _write(self, result: LearningCurveResult) -> None:
        output_path = self._config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self._write_csv(result)
        self._write_png(result)
        logger.info(
            "learning curve written to %s (min fraction %s, %s rows)",
            output_path,
            result.min_fraction,
            result.min_rows,
        )

    def _write_csv(self, result: LearningCurveResult) -> None:
        rows = [
            {column: getattr(point, column) for column in _CSV_COLUMNS} for point in result.points
        ]
        pd.DataFrame(rows, columns=list(_CSV_COLUMNS)).to_csv(self._config.csv_path, index=False)

    def _write_png(self, result: LearningCurveResult) -> None:
        fractions = [point.fraction for point in result.points]
        means = [point.mean if point.mean is not None else np.nan for point in result.points]
        stds = [point.std if point.std is not None else 0.0 for point in result.points]
        color = viz.palette_colors(1)[0]
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
        ax.errorbar(
            fractions,
            means,
            yerr=stds,
            marker="o",
            color=color,
            capsize=3,
            label=f"{result.target_metric} (mean ± std)",
        )
        reference = result.reference.mean
        if reference is not None:
            slack = result.tolerance * abs(reference)
            ax.axhline(
                reference,
                color="red",
                linestyle="--",
                linewidth=1,
                label="full data",
            )
            threshold = reference + slack if result.direction == "minimize" else reference - slack
            ax.axhline(
                threshold,
                color="red",
                linestyle=":",
                linewidth=1,
                label=f"{result.direction} tolerance {result.tolerance:g}",
            )
        if result.min_fraction is not None:
            ax.axvline(result.min_fraction, color="black", linestyle="-.", linewidth=1)
            ax.annotate(
                f"min fraction {result.min_fraction:g} ({result.min_rows} rows)",
                xy=(result.min_fraction, np.nanmax(means) if np.isfinite(means).any() else 0.0),
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=viz.TICK_FONTSIZE,
            )
        ax.set_xlabel("training fraction")
        ax.set_ylabel(result.target_metric)
        ax.set_title("Learning curve — minimum training data")
        ax.grid(alpha=viz.GRID_ALPHA)
        ax.legend(fontsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
        fig.savefig(self._config.png_path)
        plt.close(fig)
