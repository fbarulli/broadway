"""Outlier share as a decision: winsorize this column, or leave it alone?

``quality.QualityAnalyzer`` observes the share of rows outside the IQR fence and
beyond the z threshold; this class turns that observation into a decision. The
share's interval comes from ``reports.measurement`` (one bootstrap of the
observed rows, recording n and seed); the null band is **binomial** at the same
n around the share a Gaussian reference produces at the same multiplier, because
a permutation of a column cannot move a share — shuffling leaves it identical,
so a shuffle "null" would only echo the measurement. Segments are semantic
(``eda.segments``), never row halves of a sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from broadway.config.schema import EdaStep
from broadway.eda.models import OutlierResult, OutlierStats, OutlierVerdict, QualityResult
from broadway.eda.segments import SegmentPlan
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec

_QUARTILES = (0.25, 0.75)
_PERCENT = 100.0
_CONTEXT_ACTION = "context only — measured for interpretation; never a predictor"


class OutlierShareStatistic:
    """The share of rows outside fixed fences — the estimator's callable statistic.

    The fences come from the observed column and are held fixed on every
    resample: re-fitting the fence per draw would measure the sampling noise of
    a quantile, not the share the decision is about.
    """

    def __init__(self, column: str, lower: float, upper: float) -> None:
        self._column = column
        self._lower = lower
        self._upper = upper

    def __call__(self, frame: pd.DataFrame) -> float:
        values = frame[self._column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            return 0.0
        return float(((values < self._lower) | (values > self._upper)).sum() / values.size)


class OutlierDecisionAnalyzer:
    """Per-column IQR/z outlier share against the configured maximum share."""

    def __init__(self, step: EdaStep) -> None:
        self._step = step
        self._max_share = step.outlier_max_share
        self._gaussian_share = self._gaussian_outlier_share(step.iqr_multiplier)
        self._estimator = MeasurementEstimator()

    def analyze(
        self,
        quality: QualityResult,
        frame: pd.DataFrame,
        provenance: dict[str, str],
        plan: SegmentPlan,
        roles: dict[str, str] | None = None,
    ) -> OutlierResult:
        """Decide every screened numeric column; the IQR share is the metric."""
        roles = roles or {}
        verdicts: list[OutlierVerdict] = []
        decisions: list[Decision] = []
        for item in quality.outliers:
            role = roles.get(item.column, "feature")
            decision = self._decide(item, frame, provenance, plan, role)
            if role == "feature":
                decisions.append(decision)
            verdicts.append(
                OutlierVerdict(
                    column=item.column,
                    role=role,
                    n=item.n,
                    iqr_count=item.iqr_count,
                    iqr_share=item.iqr_rate,
                    z_count=item.z_count,
                    z_share=item.z_rate,
                    max_share=self._max_share,
                    verdict=decision.verdict,
                )
            )
        return OutlierResult(max_share=self._max_share, columns=verdicts, decisions=decisions)

    def _decide(
        self,
        item: OutlierStats,
        frame: pd.DataFrame,
        provenance: dict[str, str],
        plan: SegmentPlan,
        role: str,
    ) -> Decision:
        lower, upper = self._fences(item)
        segments = self._segments(frame, item, lower, upper, plan)
        measurement = self._estimator.estimate(
            frame[[item.column]].dropna(),
            OutlierShareStatistic(item.column, lower, upper),
            MeasurementSpec(
                bootstrap="iid",
                iterations=self._step.permutations,
                seed=self._step.random_state,
                confidence=1.0 - self._step.alpha,
            ),
            name=f"outlier_share_{item.column}",
        )
        uncovered = max(measurement.sample_size - sum(s.sample_size for s in segments), 0)
        null = self._binomial_null(measurement.sample_size)
        # Context columns are measured but never act: their share is a fact about
        # the outcome window, not a preprocessing instruction for the model. A
        # 1.0 ceiling makes `act` unreachable (a share cannot exceed 1).
        context = role != "feature"
        rule = DecisionRule(
            metric=f"outlier_share_{item.column}",
            direction="max",
            min_effect=1.0 if context else self._max_share,
            min_sample_size=self._step.min_sample_size,
            action=(
                _CONTEXT_ACTION if context else self._step.decisions.outliers.action
            ),
            owner=self._step.decisions.outliers.owner,
        )
        evidence = Evidence(
            baseline=(
                "binomial null at this n around the share a Gaussian reference "
                f"produces at iqr_multiplier={self._step.iqr_multiplier:g}"
            ),
            segments=segments,
            uncovered=uncovered,
            sensitivity={
                "z_score_share": item.z_rate,
                "iqr_multiplier": float(self._step.iqr_multiplier),
                "iterations": float(self._step.permutations),
            },
            falsifier=(
                "the share's interval falls at or below the declared maximum share — "
                "the column would carry no more extreme values than its fence implies"
            ),
            provenance={
                **provenance,
                "column": item.column,
                "role": role,
                "rows": str(measurement.sample_size),
                "seed": str(measurement.seed),
                "resampling": measurement.resampling,
                "segments": plan.rule,
            },
        )
        return Decision.evaluate(
            rule=rule,
            value=measurement.value,
            interval=(measurement.interval_low, measurement.interval_high),
            sample_size=measurement.sample_size,
            question=(
                f"Is more than {self._max_share * _PERCENT:.2f}% of {item.column} outside "
                f"the IQR fence of {self._step.iqr_multiplier:g} x IQR?"
            ),
            evidence=evidence,
            null_values=null,
            reference=self._gaussian_share,
        )

    @staticmethod
    def _fences(item: OutlierStats) -> tuple[float, float]:
        if item.iqr_lower is None or item.iqr_upper is None:
            return float("-inf"), float("inf")
        return item.iqr_lower, item.iqr_upper

    def _binomial_null(self, n: int) -> list[float]:
        """The no-excess-outliers null at this n: Binomial(n, gaussian share) / n."""
        generator = np.random.default_rng(self._step.random_state)
        draws = generator.binomial(n, self._gaussian_share, self._step.permutations)
        return [float(value) for value in draws / n]

    def _segments(
        self,
        frame: pd.DataFrame,
        item: OutlierStats,
        lower: float,
        upper: float,
        plan: SegmentPlan,
    ) -> list[SegmentEvidence]:
        """Re-measure the share inside each semantic segment (not row halves)."""
        labels = plan.labels(frame).reset_index(drop=True)
        segments = []
        for name, index in SegmentPlan.groups(labels):
            values = frame[item.column].iloc[index].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            share = float(((values < lower) | (values > upper)).sum() / values.size)
            segments.append(
                SegmentEvidence(segment=name, value=share, sample_size=int(values.size))
            )
        return segments or [
            SegmentEvidence(segment="all rows", value=item.iqr_rate, sample_size=max(item.n, 1))
        ]

    @staticmethod
    def _gaussian_outlier_share(iqr_multiplier: float) -> float:
        """The IQR-fence share a standard normal produces at the same multiplier.

        A normal has q25/q75 at ∓0.6745, so the fence sits at
        ``0.6745 + multiplier * 1.349`` standard deviations. This is the
        no-excess-outliers reference the binomial null is drawn around.
        """
        q25, q75 = stats.norm.ppf(list(_QUARTILES))
        fence = float(q75 + iqr_multiplier * (q75 - q25))
        return float(2.0 * (1.0 - stats.norm.cdf(fence)))
