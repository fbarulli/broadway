"""Turn a backtest into a canonical Decision: interval, null band, verdict.

The metric is the mean relative MAE reduction across chronological folds (the
CV mean); the spread is the fold standard deviation. The 95% interval is a
seeded fold-level percentile bootstrap; the no-skill band is a seeded paired
sign-flip permutation within each fold. The verdict is computed by
:meth:`Decision.evaluate`, never typed.
"""

from __future__ import annotations

import numpy as np

from broadway.config.schema import PermutationConfig
from broadway.evaluate.contracts import BootstrapConfig
from broadway.reports.decision import (
    Decision,
    DecisionRule,
    Evidence,
    SegmentEvidence,
)
from broadway.timeseries.backtest import BacktestResult, relative_improvement


class SeriesDecisionBuilder:
    """Measure one series' forecast skill against its seasonal-naive baseline."""

    def __init__(
        self,
        rule: DecisionRule,
        bootstrap: BootstrapConfig,
        null: PermutationConfig,
    ) -> None:
        self._rule = rule
        self._bootstrap = bootstrap
        self._null = null

    def build(
        self,
        *,
        result: BacktestResult,
        question: str,
        provenance: dict[str, str],
    ) -> Decision:
        """Build the validated decision for one backtest result."""
        value = result.mean_fold_improvement
        interval = self._bootstrap_interval(result)
        low, high = interval
        evidence = Evidence(
            baseline=(
                "same-weekday seasonal naive (lag 7), recursive over the test block"
            ),
            delta=value,
            segments=self._segments(result),
            sensitivity=self._sensitivity(result),
            falsifier=(
                "the mean fold-improvement interval falls at or below "
                "max(min_effect, the paired sign-flip null at this n) — the model "
                "would be indistinguishable from the seasonal-naive baseline"
            ),
            provenance=provenance,
        )
        return Decision.evaluate(
            rule=self._rule,
            value=value,
            interval=interval,
            sample_size=result.n,
            question=question,
            evidence=evidence,
            null_values=self._null_values(result),
            reference=0.0,
            unit="mean fold relative MAE improvement",
            bootstrap_biased=not (low <= value <= high),
        )

    def _bootstrap_interval(self, result: BacktestResult) -> tuple[float, float]:
        """Seeded percentile bootstrap over folds (the CV unit of evidence)."""
        values = result.fold_improvements
        rng = np.random.default_rng(self._bootstrap.seed)
        draws = np.empty(self._bootstrap.iterations, dtype=float)
        for index in range(self._bootstrap.iterations):
            sample = rng.integers(0, values.size, values.size)
            draws[index] = float(np.mean(values[sample]))
        tail = (1.0 - self._bootstrap.level) / 2.0
        low, high = np.percentile(draws, [100.0 * tail, 100.0 * (1.0 - tail)])
        return float(low), float(high)

    def _null_values(self, result: BacktestResult) -> list[float]:
        """Paired swap null within each fold, then the mean across folds.

        Under no skill the model and baseline absolute errors are exchangeable
        day by day; randomly swapping them keeps both errors non-negative and
        preserves each fold's geometry.
        """
        rng = np.random.default_rng(self._null.seed)
        draws = np.empty(self._null.iterations, dtype=float)
        for index in range(self._null.iterations):
            swap = rng.random(result.n) < 0.5
            baseline_perm = np.where(swap, result.model_errors, result.baseline_errors)
            model_perm = np.where(swap, result.baseline_errors, result.model_errors)
            fold_values: list[float] = []
            for fold in np.unique(result.fold_ids):
                selected = result.fold_ids == fold
                fold_values.append(
                    relative_improvement(
                        float(np.mean(baseline_perm[selected])),
                        float(np.mean(model_perm[selected])),
                    )
                )
            draws[index] = float(np.mean(fold_values))
        return draws.tolist()

    @staticmethod
    def _segments(result: BacktestResult) -> list[SegmentEvidence]:
        """Re-measure the fold-mean improvement inside weekday and weekend slices."""
        weekend = result.dates.dayofweek >= 5
        segments: list[SegmentEvidence] = []
        for label, mask in (("weekday", ~weekend), ("weekend", weekend)):
            if not bool(np.any(mask)):
                continue
            segments.append(
                SegmentEvidence(
                    segment=label,
                    value=result.fold_improvement(mask),
                    sample_size=int(np.count_nonzero(mask)),
                )
            )
        if not segments:
            raise ValueError("backtest has no scored days to segment")
        return segments

    @staticmethod
    def _sensitivity(result: BacktestResult) -> dict[str, float]:
        """Alternative metric and short-horizon check (each differs from the headline)."""
        rmse = relative_improvement(result.baseline_rmse, result.model_rmse)
        first_week = result.horizon_positions < 7
        return {
            "pooled_relative_improvement": result.pooled_relative_improvement,
            "rmse_relative_improvement": float(rmse),
            "first_week_relative_improvement": result.fold_improvement(first_week),
        }
