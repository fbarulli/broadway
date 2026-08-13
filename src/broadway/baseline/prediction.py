from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from broadway.analysis.contracts import AnalysisMode
from broadway.baseline.contracts import BaselineResult
from broadway.config.schema import TaskType


def run(df: pd.DataFrame, target: str, task: TaskType) -> BaselineResult:
    y = df[target]
    if task == TaskType.CLASSIFICATION:
        majority = y.mode().iloc[0]
        preds = np.full(len(y), majority)
        value = float(accuracy_score(y, preds))
        return BaselineResult(
            mode=AnalysisMode.PREDICTION,
            strategy="majority_class",
            metric="accuracy",
            value=value,
            details={"majority_class": str(majority), "support": int(len(y))},
            notes=[f"majority-class baseline: predict '{majority}' for every row"],
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
