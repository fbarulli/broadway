"""Cross-validation, residual summaries, and residual diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate

from broadway.config.schema import PipelineConfig, TrainStep
from broadway.evaluate.contracts import (
    LEDGER_RESIDUAL_COLUMN,
    TIME_CV_KINDS,
    BootstrapConfig,
    OriginMetric,
    ResidualDiagnostics,
    ResidualDiagnosticsConfig,
    TimeSeriesCvResult,
)
from broadway.evaluate.metrics import (
    METRIC_DECIMALS,
    METRIC_NAMES,
    MetricBootstrap,
    compute_metrics,
    require_mape_defined,
)
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.stats.diagnostics import jb_test, qq_correlation
from broadway.stats.time_series import acf, as_series, ljung_box

logger = logging.getLogger(__name__)

_SCORING: dict[str, str] = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
    "max_error": "max_error",
    "median_ae": "neg_median_absolute_error",
    "explained_var": "explained_variance",
}

# The only negated metric whose scorer name lacks the neg_ prefix: sklearn
# registers max_error with greater_is_better=False
# (sklearn.metrics.get_scorer("max_error")._sign < 0), so its values are
# negated despite the plain name. Everything else is derivable from _SCORING.
_NEGATED_WITHOUT_PREFIX: frozenset[str] = frozenset({"max_error"})
_NEGATED_METRICS: frozenset[str] = frozenset(
    metric for metric, scorer in _SCORING.items() if scorer.startswith("neg_")
) | _NEGATED_WITHOUT_PREFIX


def _make_cv(
    cv_kind: str,
    cv_folds: int,
    random_state: int,
    gap: int = 0,
    *,
    seasonal_period: int | None = None,
    train_size: int | None = None,
    dates: pd.Series | np.ndarray | None = None,
) -> BaseCrossValidator:
    """Build the cross-validator for a split kind.

    ``expanding`` is forward-chaining (train on the chronological prefix,
    validate the next block); ``time_series_split`` is its pre-existing alias;
    ``rolling`` keeps a fixed-length training window (``train_size``) that
    slides forward. ``gap`` drops that many steps between a training prefix and
    its validation block (the purge gap — a forecast horizon of h needs
    ``gap >= h``, enforced at the config boundary). ``seasonal_period`` cuts
    fold boundaries on whole cycles, and ``dates`` makes the cuts time-indexed
    instead of row-indexed. ``kfold`` stays the shuffled-random split for
    random/stratified experiments.
    """
    if gap < 0:
        raise ValueError(f"cv gap must be >= 0, got {gap}")
    if cv_kind in TIME_CV_KINDS:
        axis = TimeAxis.build(dates)
        if axis is None:
            logger.info(
                "cv: folding on row order — no regular datetime axis was supplied; "
                "a missing timestamp can shrink the effective gap"
            )
        kind = "rolling" if cv_kind == "rolling" else "expanding"
        return TimeSeriesCvSplitter(
            n_splits=cv_folds,
            kind=kind,
            gap=gap,
            seasonal_period=seasonal_period,
            train_size=train_size,
            axis=axis,
        )
    if cv_kind == "kfold":
        if gap:
            raise ValueError("cv gap only applies to an expanding (time-ordered) split")
        return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    raise ValueError(f"unknown cv_kind: {cv_kind!r}")


class TimeAxis:
    """Row order projected onto a regular time grid.

    ``positions[i]`` is the grid step of row ``i`` (``(t_i - t_0) / frequency``)
    and ``n_positions`` is the grid length. A regular grid means a missing
    timestamp still occupies its step, so a fold boundary cut in grid units
    cannot silently shrink the purge gap. Irregular or duplicate timestamps
    have no single-step grid and return ``None`` (the caller then folds on row
    order).
    """

    def __init__(self, positions: np.ndarray, n_positions: int) -> None:
        self._positions = positions
        self.n_rows = int(positions.size)
        self.n_positions = n_positions

    @classmethod
    def build(cls, dates: pd.Series | np.ndarray | None) -> TimeAxis | None:
        if dates is None:
            return None
        values = pd.to_datetime(pd.Series(dates)).to_numpy()
        if values.size < 2:
            return None
        if pd.isna(values).any():
            raise ValueError("the datetime column contains missing values")
        if np.any(values[1:] < values[:-1]):
            raise ValueError(
                "the datetime column must be non-decreasing for time-series CV"
            )
        unique = np.unique(values)
        if unique.size != values.size:
            logger.info("cv: duplicate timestamps — folding on row order")
            return None
        steps = np.diff(unique)
        frequency = steps.min()
        ratios = steps / frequency
        if not np.all(ratios == np.round(ratios)):
            logger.info(
                "cv: timestamps are not a regular grid — folding on row order; "
                "a missing timestamp can shrink the effective gap"
            )
            return None
        positions = ((values - unique[0]) // frequency).astype(np.int64)
        return cls(positions=positions, n_positions=int(positions[-1]) + 1)

    def row_bounds(self, start: int, end: int) -> tuple[int, int]:
        """Row indices covering grid positions ``[start, end)``."""
        return (
            int(np.searchsorted(self._positions, start, side="left")),
            int(np.searchsorted(self._positions, end, side="left")),
        )


class TimeSeriesCvSplitter(BaseCrossValidator):
    """Deterministic forward-chaining folds with purge, rolling, and seasonal cuts.

    Folds are contiguous and ordered: every training prefix ends at least
    ``gap`` steps before its validation block begins. ``expanding`` trains from
    the series start; ``rolling`` keeps the last ``train_size`` steps. With
    ``seasonal_period`` set, the scored geometry is trimmed to whole periods so
    every fold boundary is a period multiple and each validation block is a
    whole number of cycles — a fold can never split a cycle.
    """

    def __init__(
        self,
        n_splits: int,
        kind: str,
        gap: int,
        seasonal_period: int | None,
        train_size: int | None,
        axis: TimeAxis | None,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"cv needs at least one fold, got {n_splits}")
        if kind not in {"expanding", "rolling"}:
            raise ValueError(f"unknown time-series cv kind: {kind!r}")
        if kind == "rolling" and train_size is None:
            raise ValueError("rolling cv requires an explicit train_size")
        self._n_splits = n_splits
        self._kind = kind
        self._gap = gap
        self._seasonal_period = seasonal_period
        self._train_size = train_size
        self._axis = axis
        self.time_indexed = axis is not None
        self.test_size: int = 0
        self.train_sizes: list[int] = []

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self._n_splits

    def split(self, X, y=None, groups=None):
        if self._axis is not None and self._axis.n_rows != len(X):
            raise ValueError(
                f"the datetime axis covers {self._axis.n_rows} rows but X has "
                f"{len(X)}; the time index must align with the scored frame"
            )
        units = self._axis.n_positions if self._axis is not None else len(X)
        period = self._seasonal_period
        if period is None:
            test_size = units // (self._n_splits + 1)
            if test_size <= 0:
                raise ValueError(
                    f"cv needs more than {self._n_splits} time steps; got {units}"
                )
            base_end = test_size * (self._n_splits + 1)
        else:
            usable = (units // period) * period
            required = (self._n_splits + 1) * period
            if usable < required:
                raise ValueError(
                    f"seasonal cv needs at least {required} time steps for "
                    f"{self._n_splits} folds of whole {period}-step periods; "
                    f"got {units}"
                )
            test_size = (usable // (self._n_splits + 1)) // period * period
            if self._train_size is not None and self._train_size % period:
                raise ValueError(
                    f"cv_train_size {self._train_size} must be a whole number of "
                    f"{period}-step seasonal periods"
                )
            base_end = usable
        self.test_size = test_size
        self.train_sizes = []
        for step in range(1, self._n_splits + 1):
            test_end = base_end - (self._n_splits - step) * test_size
            test_start = test_end - test_size
            train_end = test_start - self._gap
            train_start = 0 if self._kind == "expanding" else train_end - (
                self._train_size or 0
            )
            if train_start < 0:
                raise ValueError(
                    f"rolling cv_train_size {self._train_size} plus gap {self._gap} "
                    f"does not fit before fold {step}'s validation block"
                )
            if train_end <= train_start:
                raise ValueError(f"cv gap {self._gap} leaves no training rows for fold {step}")
            train_idx = self._rows(train_start, train_end)
            test_idx = self._rows(test_start, test_end)
            if train_idx.size == 0 or test_idx.size == 0:
                raise ValueError(
                    f"fold {step} mapped to an empty split: {train_idx.size} train / "
                    f"{test_idx.size} test rows"
                )
            self.train_sizes.append(int(train_idx.size))
            yield train_idx, test_idx

    def _rows(self, start: int, end: int) -> np.ndarray:
        if self._axis is None:
            return np.arange(start, end)
        first, last = self._axis.row_bounds(start, end)
        return np.arange(first, last)


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    cv_kind: str,
    decimals: int = METRIC_DECIMALS,
    gap: int = 0,
) -> dict[str, float]:
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    # The scored vocabulary always includes mape; a zero truth value makes it
    # undefined, so the CV path fails loud exactly like the holdout path.
    require_mape_defined(y)
    cv = _make_cv(cv_kind, cv_folds, random_state, gap)
    scores = sklearn_cross_validate(model, X, y, cv=cv, scoring=_SCORING)
    return _mean_metrics(scores, decimals)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Signed and absolute residual summaries (residual = prediction - truth).

    ``mean_residual`` is the SIGNED mean, so positive means over-prediction.
    ``mean_abs_residual`` carries the mean absolute residual under a name that
    says so (the historical ``mean_residual`` value). ``signed_bias`` divides
    the signed mean by the mean absolute truth level, giving a scale-free
    relative bias.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {truth.shape} "
            f"and {prediction.shape}"
        )
    residuals = prediction - truth
    scale = float(np.mean(np.abs(truth)))
    if scale == 0.0:
        raise ValueError(
            "signed_bias is undefined when the mean absolute truth level is zero"
        )
    return {
        "mean_residual": float(np.mean(residuals)),
        "mean_abs_residual": float(np.mean(np.abs(residuals))),
        "signed_bias": float(np.mean(residuals)) / scale,
        "std_residual": float(np.std(residuals)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


class ResidualDiagnosticsPass:
    """Statistical residual diagnostics over the persisted holdout ledger.

    Reads the residual column of the ledger the error-analysis pass wrote —
    the artifact a consumer can inspect — instead of re-deriving predictions.
    Ledger rows preserve the holdout frame's order, which is chronological for
    a time split; that order is what the ACF and Ljung-Box reads assume.
    """

    def __init__(self, config: ResidualDiagnosticsConfig) -> None:
        self._config = config

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> ResidualDiagnosticsPass:
        # run() establishes the evaluate section before this pass is built.
        assert cfg.evaluate is not None
        return cls(cfg.evaluate.residuals)

    def compute(self, ledger_path: Path) -> ResidualDiagnostics:
        ledger = pd.read_parquet(ledger_path, columns=[LEDGER_RESIDUAL_COLUMN])
        values = as_series(ledger[LEDGER_RESIDUAL_COLUMN].to_numpy())
        autocorrelation = acf(values, self._config.acf_lags)
        portmanteau = ljung_box(values, self._config.ljung_box_lags)
        statistic, p_value, skew, kurtosis = jb_test(values)
        return ResidualDiagnostics(
            n_obs=int(values.size),
            acf_lags={
                lag: float(autocorrelation[lag])
                for lag in range(1, self._config.acf_lags + 1)
            },
            ljung_box_lags=portmanteau.lags,
            ljung_box_statistic=portmanteau.statistic,
            ljung_box_p_value=portmanteau.p_value,
            jarque_bera_statistic=statistic,
            jarque_bera_p_value=p_value,
            jarque_bera_skew=skew,
            jarque_bera_kurtosis=kurtosis,
            qq_correlation=qq_correlation(values),
        )


class TimeSeriesCV:
    """The time-aware CV pass: purged folds, multi-step origins, a baseline decision.

    Scores every fold's validation block with the deployed Pipeline shape,
    partitions multi-step runs into forecast origins (one row per origin in
    ``origins``) and per-step pooled metrics (``horizon_metrics``), and — when a
    decision rule is configured — tests the candidate's improvement over the
    seasonal-naive (or persistence) reference with a paired bootstrap interval.
    """

    def __init__(
        self,
        config: TrainStep,
        target: str,
        decision_rule: DecisionRule | None,
        bootstrap: BootstrapConfig,
    ) -> None:
        self._config = config
        self._target = target
        self._decision_rule = decision_rule
        self._bootstrap = bootstrap

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> TimeSeriesCV:
        assert cfg.train is not None and cfg.dataset is not None
        evaluate = cfg.evaluate
        return cls(
            cfg.train,
            cfg.dataset.target,
            evaluate.cv_decision if evaluate else None,
            evaluate.bootstrap if evaluate else BootstrapConfig(),
        )

    def run(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.Series | np.ndarray | None = None,
    ) -> tuple[TimeSeriesCvResult, Decision | None]:
        config = self._config
        if (
            self._decision_rule is not None
            and self._decision_rule.metric not in METRIC_NAMES
        ):
            raise ValueError(
                f"cv_decision.metric {self._decision_rule.metric!r} is not in the "
                f"scored vocabulary {sorted(METRIC_NAMES)}"
            )
        splitter = _make_cv(
            config.cv_kind,
            config.cv_folds,
            config.random_state,
            config.cv_gap,
            seasonal_period=config.cv_seasonal_period,
            train_size=config.cv_train_size,
            dates=dates,
        )
        assert isinstance(splitter, TimeSeriesCvSplitter)
        truth = np.asarray(y, dtype=float)
        matrix = np.asarray(X, dtype=float)
        lag = config.cv_seasonal_period or 1
        truth_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        reference_parts: list[np.ndarray] = []
        fold_improvements: list[SegmentEvidence] = []
        origins: list[OriginMetric] = []
        horizon_rows: dict[int, list[tuple[float, float]]] = {}
        for fold, (train_idx, test_idx) in enumerate(splitter.split(matrix)):
            fitted = clone(model).fit(matrix[train_idx], truth[train_idx])
            prediction = np.asarray(fitted.predict(matrix[test_idx]), dtype=float)
            reference = truth[test_idx - lag]
            truth_parts.append(truth[test_idx])
            prediction_parts.append(prediction)
            reference_parts.append(reference)
            if self._decision_rule is not None:
                metric = self._decision_rule.metric
                fold_scores = compute_metrics(truth[test_idx], prediction)
                fold_reference = compute_metrics(truth[test_idx], reference)
                fold_improvements.append(
                    SegmentEvidence(
                        segment=f"fold {fold}",
                        value=fold_reference[metric] - fold_scores[metric],
                        sample_size=int(test_idx.size),
                    )
                )
            if config.cv_horizon > 1:
                for start in range(
                    0, test_idx.size - config.cv_horizon + 1, config.cv_horizon
                ):
                    block = slice(start, start + config.cv_horizon)
                    scores = compute_metrics(
                        truth[test_idx[block]], prediction[block], metrics=("rmse", "mae")
                    )
                    origins.append(
                        OriginMetric(
                            fold=fold,
                            origin=int(test_idx[start]),
                            n_steps=config.cv_horizon,
                            rmse=scores["rmse"],
                            mae=scores["mae"],
                        )
                    )
                    for position in range(block.start, block.start + config.cv_horizon):
                        row = int(test_idx[position])
                        step = position - block.start + 1
                        horizon_rows.setdefault(step, []).append(
                            (float(truth[row]), float(prediction[position]))
                        )
        all_truth = np.concatenate(truth_parts)
        all_prediction = np.concatenate(prediction_parts)
        all_reference = np.concatenate(reference_parts)
        metrics = compute_metrics(all_truth, all_prediction)
        baseline = compute_metrics(all_truth, all_reference)
        improvement = {
            name: baseline[name] - value for name, value in metrics.items()
        }
        if config.cv_horizon > 1:
            horizon_metrics = {
                step: compute_metrics(
                    np.array([row[0] for row in rows]),
                    np.array([row[1] for row in rows]),
                    metrics=("rmse", "mae"),
                )
                for step, rows in sorted(horizon_rows.items())
            }
        else:
            horizon_metrics = {1: {"rmse": metrics["rmse"], "mae": metrics["mae"]}}
        decision = None
        if self._decision_rule is not None:
            decision = self._decide(
                all_truth, all_prediction, all_reference, improvement, fold_improvements
            )
        result = TimeSeriesCvResult(
            kind=config.cv_kind,
            folds=config.cv_folds,
            horizon=config.cv_horizon,
            gap=config.cv_gap,
            seasonal_period=config.cv_seasonal_period,
            train_size=config.cv_train_size,
            time_indexed=splitter.time_indexed,
            test_size=splitter.test_size,
            train_sizes=splitter.train_sizes,
            metrics=metrics,
            horizon_metrics=horizon_metrics,
            origins=origins,
            baseline=baseline,
            improvement=improvement,
        )
        return result, decision

    def _decide(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        reference: np.ndarray,
        improvement: dict[str, float],
        segments: list[SegmentEvidence],
    ) -> Decision:
        assert self._decision_rule is not None
        metric = self._decision_rule.metric
        interval = MetricBootstrap(self._bootstrap).improvement_interval(
            truth, prediction, reference, metric
        )
        value = improvement[metric]
        lag = self._config.cv_seasonal_period or 1
        baseline_name = (
            f"seasonal-naive (lag {lag})"
            if self._config.cv_seasonal_period
            else f"persistence (lag {lag})"
        )
        evidence = Evidence(
            baseline=baseline_name,
            delta=value,
            segments=segments,
            sensitivity={
                "cv_gap": float(self._config.cv_gap),
                "cv_horizon": float(self._config.cv_horizon),
                "seasonal_period": float(self._config.cv_seasonal_period or 1),
            },
            falsifier=(
                "a paired-bootstrap interval whose upper bound falls at or below "
                f"the declared min_effect {self._decision_rule.min_effect}"
            ),
            provenance={
                "target": self._target,
                "cv_kind": self._config.cv_kind,
                "folds": str(self._config.cv_folds),
                "seed": str(self._config.random_state),
                "rows": str(truth.size),
            },
        )
        return Decision.evaluate(
            rule=self._decision_rule,
            value=value,
            # The percentile interval contains the point estimate by
            # construction except for rare small-sample draws; widening to it
            # keeps the published interval truthful to its own value.
            interval=(min(interval.lower, value), max(interval.upper, value)),
            sample_size=int(truth.size),
            question=(
                f"Does the candidate beat the {baseline_name} baseline on "
                f"{self._target} under time-series CV?"
            ),
            evidence=evidence,
            reference=0.0,
        )


def _mean_metrics(
    scores: dict[str, np.ndarray], decimals: int = METRIC_DECIMALS
) -> dict[str, float]:
    means: dict[str, float] = {}
    for name, fold_values in scores.items():
        if not name.startswith("test_"):
            continue
        metric = name.removeprefix("test_")
        values = -fold_values if metric in _NEGATED_METRICS else fold_values
        rounded_folds = [round(float(value), decimals) for value in values]
        means[metric] = round(float(np.mean(rounded_folds)), decimals)
    return means
