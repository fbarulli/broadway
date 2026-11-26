"""Outlier share as a decision: winsorize this column, or leave it alone?

``quality.QualityAnalyzer`` observes the share of rows outside the IQR fence and
beyond the z threshold; this class turns that observation into a decision — the
share against the config-declared maximum, with a Wilson interval, a
half-sample re-measurement, and the share a Gaussian reference would produce at
the same multiplier (the no-excess-outliers baseline). A share is a rate, so an
absolute threshold is only meaningful next to the n it was measured on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from broadway.config.schema import EdaStep
from broadway.eda.models import (
    OutlierResult,
    OutlierStats,
    OutlierVerdict,
    QualityResult,
)
from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence

_QUARTILES = (0.25, 0.75)
_PERCENT = 100.0


class OutlierDecisionAnalyzer:
    """Per-column IQR/z outlier share against the configured maximum share."""

    def __init__(self, step: EdaStep) -> None:
        self._step = step
        self._max_share = step.outlier_max_share
        self._gaussian_share = self._gaussian_outlier_share(step.iqr_multiplier)

    def analyze(
        self,
        quality: QualityResult,
        frame: pd.DataFrame,
        provenance: dict[str, str],
    ) -> OutlierResult:
        """Decide every screened numeric column; the IQR share is the metric."""
        verdicts: list[OutlierVerdict] = []
        decisions: list[Decision] = []
        for item in quality.outliers:
            decision = self._decide(item, frame, provenance)
            decisions.append(decision)
            verdicts.append(
                OutlierVerdict(
                    column=item.column,
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
        self, item: OutlierStats, frame: pd.DataFrame, provenance: dict[str, str]
    ) -> Decision:
        low, high = self._wilson(item.iqr_count, item.n)
        rule = DecisionRule(
            metric=f"outlier_share_{item.column}",
            direction="max",
            min_effect=self._max_share,
            null_quantile=self._step.null_quantile,
            min_sample_size=self._step.min_sample_size,
            action=self._step.decisions.outliers.action,
            owner=self._step.decisions.outliers.owner,
        )
        evidence = Evidence(
            baseline=(
                "share expected under a Gaussian reference at iqr_multiplier="
                f"{self._step.iqr_multiplier:g}"
            ),
            segments=self._segments(frame, item),
            sensitivity={
                "z_score_share": item.z_rate,
                "iqr_multiplier": float(self._step.iqr_multiplier),
            },
            falsifier=(
                "the share's 95% interval falls at or below the declared maximum share — "
                "the column would carry no more extreme values than its fence implies"
            ),
            provenance={
                **provenance,
                "column": item.column,
                "rows": str(item.n),
                "seed": str(self._step.random_state),
            },
        )
        return Decision.evaluate(
            rule=rule,
            value=item.iqr_rate,
            interval=(min(low, item.iqr_rate), max(high, item.iqr_rate)),
            sample_size=item.n,
            question=(
                f"Is more than {self._max_share * _PERCENT:.2f}% of {item.column} outside "
                f"the IQR fence of {self._step.iqr_multiplier:g} x IQR?"
            ),
            evidence=evidence,
            null_values=[self._gaussian_share],
            reference=self._gaussian_share,
        )

    def _segments(self, frame: pd.DataFrame, item: OutlierStats) -> list[SegmentEvidence]:
        """Re-measure the share inside the two row-order halves."""
        if item.iqr_lower is None or item.iqr_upper is None or item.column not in frame.columns:
            return [
                SegmentEvidence(segment="whole sample", value=item.iqr_rate, sample_size=max(item.n, 1))
            ]
        values = frame[item.column].dropna().astype(float)
        half = len(values) // 2
        segments = []
        for label, part in (("first half", values.iloc[:half]), ("second half", values.iloc[half:])):
            if part.empty:
                continue
            share = float(
                ((part < item.iqr_lower) | (part > item.iqr_upper)).sum() / len(part)
            )
            segments.append(
                SegmentEvidence(segment=label, value=share, sample_size=len(part))
            )
        return segments or [
            SegmentEvidence(segment="whole sample", value=item.iqr_rate, sample_size=max(item.n, 1))
        ]

    def _wilson(self, count: int, n: int) -> tuple[float, float]:
        """Wilson score interval for a share at the config's alpha (contains the count)."""
        if n <= 0:
            return 0.0, 0.0
        share = count / n
        z = float(stats.norm.ppf(1.0 - self._step.alpha / 2.0))
        denominator = 1.0 + z * z / n
        centre = (share + z * z / (2.0 * n)) / denominator
        half_width = z * float(np.sqrt(share * (1.0 - share) / n + z * z / (4.0 * n * n))) / denominator
        return max(0.0, centre - half_width), min(1.0, centre + half_width)

    @staticmethod
    def _gaussian_outlier_share(iqr_multiplier: float) -> float:
        """The IQR-fence share a standard normal produces at the same multiplier.

        A normal has q25/q75 at ∓0.6745, so the fence sits at
        ``0.6745 + multiplier * 1.349`` standard deviations. This is the
        no-excess-outliers baseline a decided share is read against.
        """
        q25, q75 = stats.norm.ppf(list(_QUARTILES))
        fence = float(q75 + iqr_multiplier * (q75 - q25))
        return float(2.0 * (1.0 - stats.norm.cdf(fence)))
