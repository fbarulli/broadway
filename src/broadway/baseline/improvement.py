from __future__ import annotations

from broadway.baseline.contracts import BaselineResult
from broadway.config.schema import TaskType


def improvement_vs_baseline(candidate: float, baseline: BaselineResult, task: TaskType) -> float | None:
    if baseline.value == 0:
        return None
    if task == TaskType.REGRESSION:
        return (baseline.value - candidate) / baseline.value
    return (candidate - baseline.value) / baseline.value
