"""Before/after ETL comparison when both frames exist; otherwise a loud skip.

A mean or median delta without an interval is not evidence: the ETL sample is a
draw, so a shift is only real when its bootstrap interval clears the
distribution's own sampling noise. Every shift therefore carries a bootstrap
interval (``reports.measurement`` records n and seed), and the comparison
publishes one decision on the worst column, standardized by the pre-ETL spread
(``|mean_after - mean_before| / std_before``, dimensionless and comparable to
the configured minimum effect) with a label-shuffle null at the same n.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import EdaStep
from broadway.eda.models import ColumnShift, CompareResult
from broadway.eda.segments import SegmentPlan
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec

_GROUP_COLUMN = "etl_group"
_VALUE_COLUMN = "etl_value"
_AFTER = 1.0
_DECIMALS = 6


class DeltaStatistic:
    """``after - before`` mean or median on a stacked two-group frame."""

    def __init__(self, kind: str) -> None:
        self._kind = kind

    def __call__(self, frame: pd.DataFrame) -> float:
        values = frame[_VALUE_COLUMN].to_numpy(dtype=float)
        groups = frame[_GROUP_COLUMN].to_numpy(dtype=float)
        before = values[groups < _AFTER]
        after = values[groups >= _AFTER]
        if before.size == 0 or after.size == 0:
            return 0.0
        if self._kind == "mean":
            return float(after.mean() - before.mean())
        return float(np.median(after) - np.median(before))


class StandardizedShiftStatistic:
    """``|mean_after - mean_before| / std_before`` — the dimensionless ETL shift."""

    def __call__(self, frame: pd.DataFrame) -> float:
        values = frame[_VALUE_COLUMN].to_numpy(dtype=float)
        groups = frame[_GROUP_COLUMN].to_numpy(dtype=float)
        before = values[groups < _AFTER]
        after = values[groups >= _AFTER]
        if before.size < 2 or after.size == 0:
            return 0.0
        spread = float(before.std(ddof=1))
        if spread == 0.0:
            return 0.0
        return float(abs(after.mean() - before.mean()) / spread)


class EtlComparator:
    """Compares the pre-split and post-split frames on counts and numeric drift."""

    def __init__(self, before_label: str, after_label: str, step: EdaStep) -> None:
        self._before_label = before_label
        self._after_label = after_label
        self._step = step
        self._estimator = MeasurementEstimator()

    def compare(
        self,
        before: pd.DataFrame | None,
        after: pd.DataFrame | None,
        before_rows: int,
        after_rows: int,
        skip_reason: str | None,
        notes: list[str],
        plan: SegmentPlan,
        provenance: dict[str, str],
    ) -> CompareResult:
        notes = list(notes)
        if before is None or after is None:
            return CompareResult(
                available=False,
                skip_reason=skip_reason or "required before/after frame is missing",
                before_label=self._before_label,
                after_label=self._after_label,
                before_rows=before_rows if before is not None else None,
                after_rows=after_rows if after is not None else None,
                row_delta=None,
                added_columns=[],
                removed_columns=[],
                dtype_changes={},
                numeric_shifts=[],
                notes=notes,
                decision=None,
            )
        before_columns = [str(name) for name in before.columns]
        after_columns = [str(name) for name in after.columns]
        common = [name for name in before_columns if name in after_columns]
        dtype_changes = {
            name: f"{before[name].dtype} -> {after[name].dtype}"
            for name in common
            if str(before[name].dtype) != str(after[name].dtype)
        }
        numeric = [
            name
            for name in common
            if not pd.api.types.is_bool_dtype(before[name])
            and pd.api.types.is_numeric_dtype(before[name])
            and not pd.api.types.is_bool_dtype(after[name])
            and pd.api.types.is_numeric_dtype(after[name])
        ]
        shifts = [self._shift(before[name], after[name], name) for name in numeric]
        decision = self._decide(before, after, numeric, plan, provenance)
        return CompareResult(
            available=True,
            skip_reason=None,
            before_label=self._before_label,
            after_label=self._after_label,
            before_rows=before_rows,
            after_rows=after_rows,
            row_delta=after_rows - before_rows,
            added_columns=[name for name in after_columns if name not in before_columns],
            removed_columns=[name for name in before_columns if name not in after_columns],
            dtype_changes=dtype_changes,
            numeric_shifts=shifts,
            notes=notes,
            decision=decision,
        )

    def _shift(self, before: pd.Series, after: pd.Series, name: str) -> ColumnShift:
        before_mean, before_median = self._mean(before), self._median(before)
        after_mean, after_median = self._mean(after), self._median(after)
        stacked = self._stacked(before, after)
        mean_delta = self._delta_interval(stacked, DeltaStatistic("mean"))
        median_delta = self._delta_interval(stacked, DeltaStatistic("median"))
        return ColumnShift(
            column=name,
            before_mean=before_mean,
            after_mean=after_mean,
            before_median=before_median,
            after_median=after_median,
            mean_delta=None
            if before_mean is None or after_mean is None
            else self._round(after_mean - before_mean),
            mean_delta_low=None if mean_delta is None else self._round(mean_delta[0]),
            mean_delta_high=None if mean_delta is None else self._round(mean_delta[1]),
            median_delta=None
            if before_median is None or after_median is None
            else self._round(after_median - before_median),
            median_delta_low=None if median_delta is None else self._round(median_delta[0]),
            median_delta_high=None if median_delta is None else self._round(median_delta[1]),
        )

    def _decide(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        numeric: list[str],
        plan: SegmentPlan,
        provenance: dict[str, str],
    ) -> Decision | None:
        """One decision on the column whose standardized shift is largest."""
        shifts = {
            name: StandardizedShiftStatistic()(self._stacked(before[name], after[name]))
            for name in numeric
            if len(before[name].dropna()) > 1 and len(after[name].dropna()) > 0
        }
        if not shifts:
            return None
        column = max(shifts, key=lambda name: (shifts[name], name))
        stacked = self._stacked(before[column], after[column])
        segments = self._segments(before, after, column, plan)
        measurement = self._estimator.estimate(
            stacked,
            StandardizedShiftStatistic(),
            MeasurementSpec(
                bootstrap="iid",
                null="shuffle",
                null_column=_GROUP_COLUMN,
                iterations=self._step.permutations,
                seed=self._step.random_state,
                confidence=1.0 - self._step.alpha,
            ),
            name=f"standardized_shift_{column}",
        )
        uncovered = max(measurement.sample_size - sum(s.sample_size for s in segments), 0)
        evidence = Evidence(
            baseline=f"{self._before_label} distribution (no ETL transformation applied)",
            segments=segments,
            uncovered=uncovered,
            sensitivity={
                "mean_delta": float(after[column].mean() - before[column].mean()),
                "median_delta": float(after[column].median() - before[column].median()),
                "iterations": float(self._step.permutations),
            },
            falsifier=(
                "the standardized shift's interval falls at or below the minimum effect "
                "and the label-shuffle null band at this n"
            ),
            provenance={
                **provenance,
                "column": column,
                "before": self._before_label,
                "after": self._after_label,
                "rows": str(measurement.sample_size),
                "seed": str(measurement.seed),
                "resampling": measurement.resampling,
                "segments": plan.rule,
            },
        )
        rule = DecisionRule(
            metric=f"standardized_shift_{column}",
            direction="max",
            min_effect=self._step.min_effect,
            min_sample_size=self._step.min_sample_size,
            action=self._step.decisions.comparison.action,
            owner=self._step.decisions.comparison.owner,
        )
        return measurement.to_decision(
            rule,
            question=(
                f"Did the ETL change the {column} distribution beyond sampling noise "
                f"({self._after_label} vs {self._before_label})?"
            ),
            evidence=evidence,
            reference=0.0,
        )

    def _segments(
        self, before: pd.DataFrame, after: pd.DataFrame, column: str, plan: SegmentPlan
    ) -> list[SegmentEvidence]:
        """Re-measure the standardized shift inside each semantic segment."""
        segments: list[SegmentEvidence] = []
        before_labels = plan.labels(before).reset_index(drop=True)
        after_labels = plan.labels(after).reset_index(drop=True)
        for name, _index in SegmentPlan.groups(before_labels):
            first = before[column][before_labels.to_numpy() == name].dropna()
            second = after[column][after_labels.to_numpy() == name].dropna()
            if len(first) < 2 or second.empty:
                continue
            spread = float(first.std(ddof=1))
            value = 0.0 if spread == 0.0 else float(abs(second.mean() - first.mean()) / spread)
            segments.append(
                SegmentEvidence(segment=name, value=value, sample_size=int(len(first) + len(second)))
            )
        if segments:
            return segments
        return [
            SegmentEvidence(
                segment="all rows",
                value=0.0,
                sample_size=int(len(before[column].dropna()) + len(after[column].dropna())),
            )
        ]

    def _delta_interval(
        self, stacked: pd.DataFrame, statistic: DeltaStatistic
    ) -> tuple[float, float] | None:
        clean = stacked.dropna()
        if clean.empty or clean[_GROUP_COLUMN].nunique() < 2:
            return None
        measurement = self._estimator.estimate(
            clean,
            statistic,
            MeasurementSpec(
                bootstrap="iid",
                iterations=self._step.permutations,
                seed=self._step.random_state,
                confidence=1.0 - self._step.alpha,
            ),
            name="delta",
        )
        return measurement.interval_low, measurement.interval_high

    @staticmethod
    def _stacked(before: pd.Series, after: pd.Series) -> pd.DataFrame:
        first = before.dropna().to_numpy(dtype=float)
        second = after.dropna().to_numpy(dtype=float)
        return pd.DataFrame(
            {
                _VALUE_COLUMN: np.concatenate([first, second]),
                _GROUP_COLUMN: np.concatenate(
                    [np.zeros(first.size), np.ones(second.size)]
                ),
            }
        )

    @staticmethod
    def _mean(column: pd.Series) -> float | None:
        values = column.dropna()
        return float(values.mean()) if not values.empty else None

    @staticmethod
    def _median(column: pd.Series) -> float | None:
        values = column.dropna()
        return float(values.median()) if not values.empty else None

    @staticmethod
    def _round(value: float) -> float:
        return round(value, _DECIMALS)
