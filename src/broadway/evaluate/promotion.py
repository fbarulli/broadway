"""Promotion decision logic."""

from __future__ import annotations


def should_promote(candidate_rmse: float, champion_rmse: float | None, threshold: float) -> tuple[bool, str]:
    if champion_rmse is None:
        return True, "no champion model — promoting unconditionally"
    if candidate_rmse > champion_rmse:
        return False, f"degradation: {candidate_rmse:.4f} vs champion {champion_rmse:.4f}"
    improvement = (champion_rmse - candidate_rmse) / champion_rmse
    if improvement > threshold:
        return True, f"{improvement:.1%} improvement over champion"
    return False, f"{improvement:.1%} below threshold {threshold:.1%}"
