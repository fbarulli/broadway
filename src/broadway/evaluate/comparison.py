"""Candidate vs champion side-by-side metrics."""

from __future__ import annotations

from broadway.evaluate.contracts import MetricComparison, ModelComparison


def compare_models(
    candidate: dict[str, float], champion: dict[str, float] | None
) -> ModelComparison:
    metrics: dict[str, MetricComparison] = {}
    for metric, candidate_value in candidate.items():
        champion_value = champion.get(metric) if champion is not None else None
        if champion_value is None:
            delta = None
            delta_pct = None
        else:
            delta = candidate_value - champion_value
            delta_pct = delta / champion_value if champion_value != 0 else None
        metrics[metric] = MetricComparison(
            candidate=candidate_value,
            champion=champion_value,
            delta=delta,
            delta_pct=delta_pct,
        )
    return ModelComparison(metrics=metrics)
