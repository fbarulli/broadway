"""Candidate vs champion side-by-side metrics."""

from __future__ import annotations


def compare_models(
    candidate: dict[str, float], champion: dict[str, float] | None
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for metric, candidate_value in candidate.items():
        champion_value = champion.get(metric) if champion is not None else None
        if champion_value is None:
            delta = None
            delta_pct = None
        else:
            delta = candidate_value - champion_value
            delta_pct = delta / champion_value if champion_value != 0 else None
        result[metric] = {
            "candidate": candidate_value,
            "champion": champion_value,
            "delta": delta,
            "delta_pct": delta_pct,
        }
    return result
