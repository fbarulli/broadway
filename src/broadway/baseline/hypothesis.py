from __future__ import annotations

import pandas as pd

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline.contracts import BaselineResult
from broadway.stats.groups import build_declared_groups


def run(
    df: pd.DataFrame, target: str, group_column: str, group_values: list[str]
) -> BaselineResult:
    arrays, absent = build_declared_groups(df, group_column, group_values, target)
    present = {g: values for g, values in arrays.items() if values.size}
    if absent and len(absent) < len(group_values):
        raise ValueError(f"declared groups absent from data: {absent}")
    if not present:
        raise ValueError("no groups present for naive effect baseline")
    group_means: dict[str, dict] = {
        g: {
            "count": int(values.size),
            "mean": float(values.mean()),
            # pandas Series.std() yields NaN silently for n=1; match it.
            "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        }
        for g, values in present.items()
    }
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
