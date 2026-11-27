"""Aggregate report for the time-series lane: grain, window, CV and verdicts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from broadway.config.schema import TimeseriesStep
from broadway.reports.decision import Decision
from broadway.timeseries.backtest import BacktestResult


class TimeseriesReport:
    """Render the human-readable report behind the published decision record."""

    def __init__(
        self,
        stem: str,
        step: TimeseriesStep,
        results: dict[str, BacktestResult],
        decisions: dict[str, Decision],
        backend: str,
    ) -> None:
        self._stem = stem
        self._step = step
        self._results = results
        self._decisions = decisions
        self._backend = backend

    def render(self) -> str:
        """The markdown report (aggregates only, no raw rows)."""
        names = list(self._step.series)
        return "\n".join(
            [
                f"# {self._stem}",
                "",
                "## question",
                "",
                self._question(),
                "",
                "## decision",
                "",
                self._decision_block(),
                "",
                "## method",
                "",
                self._method_block(),
                "",
                "## baseline vs model",
                "",
                self._summary_table(names),
                "",
                "## cross-validation folds",
                "",
                self._fold_table(names),
                "",
                "## evidence",
                "",
                self._evidence_block(names),
                "",
                "## figures",
                "",
                f"![{self._stem}]({self._stem}.png)",
                *[
                    f"![{name}]({self._stem}_{name}.png)"
                    for name in names
                    if name != self._step.primary_series
                ],
                "",
            ]
        )

    def _question(self) -> str:
        labels = ", ".join(spec.label for spec in self._step.series.values())
        return (
            "Does a weekly-seasonal forecast beat a same-weekday seasonal-naive "
            f"baseline on the daily series ({labels})?"
        )

    def _decision_block(self) -> str:
        rows = [
            "| series | metric | value | 95% interval | null | n | verdict |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for name in self._step.series:
            decision = self._decisions[name]
            null = (
                f"{decision.null_threshold:,.4f}"
                if decision.null_threshold is not None
                else "none"
            )
            rows.append(
                f"| {name} | {decision.metric} | {decision.value:,.4f} | "
                f"[{decision.interval_low:,.4f}, {decision.interval_high:,.4f}] | "
                f"{null} | {decision.sample_size:,} | {decision.verdict} |"
            )
        return "\n".join(rows)

    def _method_block(self) -> str:
        step = self._step
        return "\n".join(
            [
                (
                    f"- grain: one canonical row per `{step.grain_column}`; a count "
                    "series counts rows per day, a sum series sums its declared "
                    "column per day"
                ),
                (
                    f"- datetime source: `{step.datetime_column}` in the canonical "
                    "table (the dataset contract's `datetime_column` is null; the "
                    "timestamp is context, not a model feature)"
                ),
                (
                    f"- window: {step.start} to {step.end} (inclusive), one value per "
                    "calendar day; days with no rows are zero-filled"
                ),
                f"- sparse/partial handling: {step.window_note or 'not declared'}",
                (
                    "- baseline first: same-weekday seasonal naive (lag 7), recursive "
                    "over each test block"
                ),
                (
                    f"- model: {self._backend}, weekly seasonality (season length "
                    f"{step.season_length}), alpha={step.alpha}"
                ),
                (
                    f"- CV: {step.cv.kind} forward chaining, {step.cv.folds} folds, "
                    f"whole {step.cv.seasonal_period}-day cycles, gap {step.cv.gap}; "
                    "no shuffled or single split"
                ),
                (
                    "- interval: seeded fold-level percentile bootstrap "
                    f"({step.bootstrap.iterations} draws, seed {step.bootstrap.seed}, "
                    f"level {step.bootstrap.level}); each fold is one evidence unit, "
                    "so a high-volume fold cannot dominate the headline"
                ),
                (
                    "- null: seeded paired swap of each day's model/baseline absolute "
                    "errors within its fold, then the mean across folds "
                    f"({step.permutation.iterations} draws, seed "
                    f"{step.permutation.seed})"
                ),
            ]
        )

    def _evidence_block(self, names: Sequence[str]) -> str:
        rows = [
            "| series | slice / check | value | n |",
            "| --- | --- | ---: | ---: |",
        ]
        for name in names:
            decision = self._decisions[name]
            for segment in decision.evidence.segments:
                rows.append(
                    f"| {name} | segment: {segment.segment} | "
                    f"{segment.value:+,.4f} | {segment.sample_size:,} |"
                )
            for check, value in decision.evidence.sensitivity.items():
                rows.append(f"| {name} | sensitivity: {check} | {value:+,.4f} | — |")
        return "\n".join(rows)

    def _summary_table(self, names: Sequence[str]) -> str:
        rows = [
            "| series | baseline MAE | model MAE | mean fold improvement | fold spread |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in names:
            result = self._results[name]
            improvements = np.asarray(
                [fold.relative_improvement for fold in result.folds], dtype=float
            )
            rows.append(
                f"| {name} | {result.baseline_mae:,.4f} | {result.model_mae:,.4f} | "
                f"{float(np.mean(improvements)):+,.4f} | "
                f"{float(np.std(improvements, ddof=1)) if improvements.size > 1 else 0.0:,.4f} |"
            )
        return "\n".join(rows)

    def _fold_table(self, names: Sequence[str]) -> str:
        rows = [
            "| series | fold | train | test | test n | baseline MAE | model MAE | improvement |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for name in names:
            for fold in self._results[name].folds:
                rows.append(
                    f"| {name} | {fold.index} | {fold.train_start.date()} → "
                    f"{fold.train_end.date()} | {fold.test_start.date()} → "
                    f"{fold.test_end.date()} | {fold.test_size} | "
                    f"{fold.baseline_mae:,.4f} | {fold.model_mae:,.4f} | "
                    f"{fold.relative_improvement:+,.4f} |"
                )
        return "\n".join(rows)


def cv_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """The evidence table written next to the record (one row per fold)."""
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        for fold in result.folds:
            rows.append(
                {
                    "series": name,
                    "fold": fold.index,
                    "train_start": fold.train_start.date(),
                    "train_end": fold.train_end.date(),
                    "test_start": fold.test_start.date(),
                    "test_end": fold.test_end.date(),
                    "train_n": fold.train_size,
                    "test_n": fold.test_size,
                    "baseline_mae": fold.baseline_mae,
                    "model_mae": fold.model_mae,
                    "relative_improvement": fold.relative_improvement,
                }
            )
    return pd.DataFrame(rows)
