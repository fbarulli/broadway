from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline.contracts import BaselineResult
from broadway.config.schema import TaskType


def run(df: pd.DataFrame, target: str, task: TaskType) -> BaselineResult:
    y = df[target]
    if task == TaskType.CLASSIFICATION:
        prevalence = float(y.mean())
        return BaselineResult(
            mode=AnalysisMode.PREDICTION,
            strategy="naive_prevalence",
            metric="pr_auc",
            value=prevalence,
            details={
                "prevalence": prevalence,
                "bad": int((y == 1).sum()),
                "good": int((y == 0).sum()),
                "support": len(y),
            },
            notes=[
                (
                    "no-skill prevalence baseline: a constant prevalence score has "
                    "PR-AUC equal to the prevalence a model must beat"
                )
            ],
        )
    mean = float(y.mean())
    median = float(y.median())
    mean_mae = float(mean_absolute_error(y, np.full(len(y), mean)))
    median_mae = float(mean_absolute_error(y, np.full(len(y), median)))
    return BaselineResult(
        mode=AnalysisMode.PREDICTION,
        strategy="mean",
        metric="mae",
        value=mean_mae,
        details={
            "mean": mean,
            "median": median,
            "mean_mae": mean_mae,
            "median_mae": median_mae,
        },
        notes=["naive mean baseline; a candidate must beat mean_mae to be worth shipping"],
    )
