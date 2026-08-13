from __future__ import annotations

import pandas as pd

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline.contracts import BaselineResult


def run(
    df: pd.DataFrame, target: str, group_column: str, group_values: list[str]
) -> BaselineResult:
    group_means: dict[str, dict] = {}
    for g in group_values:
        vals = df[df[group_column] == g][target].dropna()
        if vals.empty:
            continue
        group_means[g] = {
            "count": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    if not group_means:
        raise ValueError("no groups present for naive effect baseline")
    means = [d["mean"] for d in group_means.values()]
    value = float(max(means) - min(means))
    return BaselineResult(
        mode=AnalysisMode.HYPOTHESIS,
        strategy="max_group_mean_diff",
        metric="mean_difference",
        value=value,
        details={"group_means": group_means},
        notes=["naive effect estimate = range of group means"],
    )
