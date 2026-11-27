"""Measured imputation impact: per-strategy metric deltas on the holdout.

This is a measurement surface, never a gate. For every configured imputation
strategy (plus the ``complete_case`` baseline) the study fits the experiment's
model on the train split and scores the val split through
``broadway.evaluate.metrics.compute_metrics``. The result is a typed delta
table: one row per (strategy, scaling) variant with the metric value, the rows
kept/dropped on each split, and the delta against ``complete_case`` at the same
scaling.

Missingness is injected into the TRAIN split only (MCAR at the configured
rate): the validation split stays clean, so every variant is scored on exactly
the same holdout rows and the comparison measures the training-time impact
rather than a change of yardstick.

Run it explicitly (the features artifacts are read, never regenerated):

    BROADWAY_CONFIG_OVERLAY_DIR=project/config \\
        python -m broadway.selection.imputation --dataset olist --experiment olist
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from broadway.config.loader import DEFAULT_ENVIRONMENT, load_config, load_environment
from broadway.config.schema import (
    ExperimentConfig,
    ImputationStrategy,
    ImputationStudyConfig,
    PipelineConfig,
    ScalingStrategy,
)
from broadway.data.loader import ProcessedStore
from broadway.evaluate.metrics import METRIC_DECIMALS, compute_metrics
from broadway.features.plan import ColumnPlan
from broadway.features.recipe import build_pipeline
from broadway.features.transformers import FeatureImputer, FeatureScaler
from broadway.observability.logging import setup_logging
from broadway.observability.profiling import timed
from broadway.training.models.registry import get_model

logger = logging.getLogger(__name__)

COMPLETE_CASE: Final = "complete_case"
NO_SCALING: Final = "none"


class ImpactRow(BaseModel):
    """One measured variant: strategy x scaling, its score, and its deltas.

    ``delta_vs_baseline`` is rounded to the metric's own reported precision
    (``broadway.evaluate.metrics.METRIC_DECIMALS``) — a delta finer than the
    metric's resolution is zero at this table's resolution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: ImputationStrategy | Literal["complete_case"]
    scaling: ScalingStrategy | Literal["none"]
    value: float
    delta_vs_baseline: float
    train_rows_kept: int = Field(ge=0)
    train_rows_dropped: int = Field(ge=0)
    val_rows_kept: int = Field(ge=0)
    val_rows_dropped: int = Field(ge=0)


class ImpactTable(BaseModel):
    """The ordered delta table: baseline rows first, then best delta first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    metric_direction: Literal["minimize", "maximize"]
    baseline: str
    rows: list[ImpactRow]

    @classmethod
    def assemble(
        cls,
        *,
        metric: str,
        metric_direction: Literal["minimize", "maximize"],
        baseline: str,
        rows: Sequence[ImpactRow],
    ) -> ImpactTable:
        """Order rows: the baseline variants first, then by improving delta."""
        if not rows:
            raise ValueError("impact table requires at least one row")
        baseline_rows = [row for row in rows if row.strategy == baseline]
        if not baseline_rows:
            raise ValueError(f"impact table has no baseline '{baseline}' row")
        others = sorted(
            (row for row in rows if row.strategy != baseline),
            key=lambda row: row.delta_vs_baseline,
            reverse=metric_direction == "maximize",
        )
        return cls(
            metric=metric,
            metric_direction=metric_direction,
            baseline=baseline,
            rows=[*baseline_rows, *others],
        )

    def to_frame(self) -> pd.DataFrame:
        """The table as a display frame, in its declared order."""
        return pd.DataFrame(
            [
                {
                    "strategy": row.strategy,
                    "scaling": row.scaling,
                    self.metric: row.value,
                    "delta_vs_baseline": row.delta_vs_baseline,
                    "train_rows_kept": row.train_rows_kept,
                    "train_rows_dropped": row.train_rows_dropped,
                    "val_rows_kept": row.val_rows_kept,
                    "val_rows_dropped": row.val_rows_dropped,
                }
                for row in self.rows
            ]
        )


class _Measurement(BaseModel):
    """One variant's metric plus the row counts it used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    train_rows_kept: int = Field(ge=0)
    train_rows_dropped: int = Field(ge=0)
    val_rows_kept: int = Field(ge=0)
    val_rows_dropped: int = Field(ge=0)

    def row_counts(self) -> dict[str, int]:
        """The kept/dropped counts as ImpactRow fields."""
        return {
            "train_rows_kept": self.train_rows_kept,
            "train_rows_dropped": self.train_rows_dropped,
            "val_rows_kept": self.val_rows_kept,
            "val_rows_dropped": self.val_rows_dropped,
        }


class ImputationImpactStudy:
    """Fit every configured imputation/scaling variant and score the holdout."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.dataset is None or cfg.experiment is None or cfg.features is None:
            raise ValueError(
                "imputation study requires dataset, experiment, and features config"
            )
        if cfg.etl is None:
            raise ValueError("imputation study requires the etl config (feature file names)")
        if cfg.features.imputation_study is None:
            raise ValueError(
                "features config has no imputation_study block — add one to "
                "configs/step/features.yaml before running the study"
            )
        self._cfg = cfg
        self._study: ImputationStudyConfig = cfg.features.imputation_study
        self._experiment: ExperimentConfig = cfg.experiment
        self._train_file = cfg.etl.train_features_file
        self._val_file = cfg.etl.val_features_file
        self._processed_dir = ProcessedStore.from_config(cfg).directory

    def run(self) -> ImpactTable:
        """Measure every variant and write the delta table to its config path."""
        X_train, y_train = self._bounded(*self._load(self._train_file))
        X_val, y_val = self._bounded(*self._load(self._val_file))
        X_train = self._inject(X_train)

        baseline_values: dict[str, float] = {}
        rows: list[ImpactRow] = []
        for scaling in self._study.scalings:
            counts = self._score(X_train, y_train, X_val, y_val, COMPLETE_CASE, scaling)
            baseline_values[str(scaling)] = counts.value
            rows.append(
                ImpactRow(
                    strategy=COMPLETE_CASE,
                    scaling=scaling,
                    value=counts.value,
                    delta_vs_baseline=0.0,
                    **counts.row_counts(),
                )
            )
        for strategy in self._study.strategies:
            for scaling in self._study.scalings:
                measured = self._score(X_train, y_train, X_val, y_val, strategy, scaling)
                rows.append(
                    ImpactRow(
                        strategy=strategy,
                        scaling=scaling,
                        value=measured.value,
                        delta_vs_baseline=round(
                            measured.value - baseline_values[str(scaling)], METRIC_DECIMALS
                        ),
                        **measured.row_counts(),
                    )
                )
        table = ImpactTable.assemble(
            metric=self._experiment.target_metric,
            metric_direction=self._study.metric_direction,
            baseline=COMPLETE_CASE,
            rows=rows,
        )
        path = self._study.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "imputation impact study: metric=%s rows=%d -> %s\n%s",
            table.metric,
            len(table.rows),
            path,
            table.to_frame().to_string(index=False),
        )
        return table

    def _load(self, filename: str) -> tuple[pd.DataFrame, pd.Series]:
        """Read one persisted feature split and project it through the column plan."""
        assert self._cfg.dataset is not None and self._cfg.experiment is not None
        path = self._processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"imputation study requires the feature split: {path}")
        plan = ColumnPlan.from_config(self._cfg.dataset, self._cfg.experiment)
        # Project on read: the plan's columns plus the inputs its ops name (the
        # datetime source of a derived column, an encoder's key columns). A
        # column that is genuinely absent then fails loud in plan.apply.
        needed = {
            *plan.names,
            *(source for op in plan.columns for source in op.sources),
            self._cfg.dataset.target,
        }
        frame = pd.read_parquet(path, columns=sorted(needed))
        features, _ = plan.apply(frame)
        return features, frame[self._cfg.dataset.target]

    def _bounded(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Draw the configured row sample (0 = every row) with aligned indices."""
        size = self._study.sample_size
        if size == 0 or size >= len(X):
            return X.reset_index(drop=True), y.reset_index(drop=True)
        rng = np.random.default_rng(self._experiment.random_state)
        picked = np.sort(rng.choice(len(X), size=size, replace=False))
        return X.iloc[picked].reset_index(drop=True), y.iloc[picked].reset_index(drop=True)

    def _inject(self, X: pd.DataFrame) -> pd.DataFrame:
        """Churn the configured columns of the TRAIN split at the configured rate."""
        injection = self._study.missing_injection
        missing = [column for column in injection.columns if column not in X.columns]
        if missing:
            raise ValueError(
                f"missing_injection.columns {missing} are not in the model surface "
                f"(available: {list(X.columns)})"
            )
        rng = np.random.default_rng(self._experiment.random_state)
        injected = X.copy()
        for column in injection.columns:
            values = injected[column].astype(float)
            values[rng.random(len(values)) < injection.rate] = np.nan
            injected[column] = values
        return injected

    def _pipeline(
        self,
        columns: list[str],
        strategy: ImputationStrategy | Literal["complete_case"],
        scaling: ScalingStrategy | Literal["none"],
    ) -> Pipeline:
        """Compose the experiment recipe, the variant steps, and the registry model."""
        assert self._cfg.experiment is not None
        steps: list[tuple[str, object]] = list(build_pipeline(self._cfg).steps)
        if strategy != COMPLETE_CASE:
            steps.append(
                (
                    "imputation",
                    FeatureImputer(
                        columns=columns,
                        strategy=ImputationStrategy(strategy),
                        knn_neighbors=self._study.knn_neighbors,
                    ),
                )
            )
        if scaling != NO_SCALING:
            steps.append(
                ("scaling", FeatureScaler(columns=columns, strategy=ScalingStrategy(scaling)))
            )
        steps.append(
            (
                "model",
                get_model(
                    self._experiment.model.type,
                    **self._experiment.model.params,
                ),
            )
        )
        return Pipeline(steps)

    def _score(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        strategy: ImputationStrategy | Literal["complete_case"],
        scaling: ScalingStrategy | Literal["none"],
    ) -> _Measurement:
        """Fit one variant on train, score the holdout, and count the rows used."""
        if strategy == COMPLETE_CASE:
            train_mask = X_train.notna().all(axis=1)
            val_mask = X_val.notna().all(axis=1)
            fit_X, fit_y = X_train[train_mask], y_train[train_mask]
            score_X, score_y = X_val[val_mask], y_val[val_mask]
        else:
            fit_X, fit_y = X_train, y_train
            score_X, score_y = X_val, y_val
        if fit_X.empty or score_X.empty:
            raise ValueError(
                f"strategy '{strategy}' leaves no rows to fit/score "
                f"(train={len(fit_X)}, val={len(score_X)})"
            )
        pipeline = self._pipeline(list(X_train.columns), strategy, scaling)
        with timed() as elapsed:
            pipeline.fit(fit_X, fit_y)
            predictions = np.asarray(pipeline.predict(score_X), dtype=float)
        metrics = compute_metrics(score_y.to_numpy(dtype=float), predictions)
        metric = self._experiment.target_metric
        if metric not in metrics:
            raise ValueError(
                f"target_metric '{metric}' is not produced by compute_metrics "
                f"(available: {sorted(metrics)})"
            )
        # The KNN variant is the slow one; timing is what justifies sample_size.
        logger.info(
            "imputation variant %s/%s: %s=%.4f in %.2fs",
            strategy,
            scaling,
            metric,
            metrics[metric],
            elapsed(),
        )
        return _Measurement(
            value=metrics[metric],
            train_rows_kept=len(fit_X),
            train_rows_dropped=len(X_train) - len(fit_X),
            val_rows_kept=len(score_X),
            val_rows_dropped=len(X_val) - len(score_X),
        )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: measure the configured study and print its delta table."""
    parser = argparse.ArgumentParser(prog="ds-pipeline-imputation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--environment", type=str, default=DEFAULT_ENVIRONMENT)
    args = parser.parse_args(argv)
    setup_logging(load_environment(args.environment))
    cfg = load_config(
        "features",
        dataset=args.dataset,
        experiment=args.experiment,
        environment=args.environment,
    )
    table = ImputationImpactStudy(cfg).run()
    logger.info("imputation impact study complete: %d rows", len(table.rows))


if __name__ == "__main__":
    main()
