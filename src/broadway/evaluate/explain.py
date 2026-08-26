"""Generic model-explainability helpers: SHAP, permutation importance, PDP/ICE, residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


class Predictable(Protocol):
    def predict(self, X: object) -> object: ...


def shap_summary(
    model: Predictable,
    X: pd.DataFrame,
    out_path: Path,
    kind: str,
    sample_size: int = 100,
    random_state: int = 0,
) -> None:
    if kind == "tree":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        explained = X
    elif kind == "kernel":
        background = X.sample(n=min(sample_size, len(X)), random_state=random_state)
        explainer = shap.KernelExplainer(model.predict, background)
        values = explainer.shap_values(background, silent=True)
        explained = background
    else:
        raise ValueError(f"unsupported kind {kind!r}; expected 'tree' or 'kernel'")
    shap.summary_plot(values, explained, show=False, max_display=min(10, X.shape[1]))
    fig = plt.gcf()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def permutation_importance_table(
    model: Predictable,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )


def pdp_ice(model: Predictable, X: pd.DataFrame, features: list[str], out_path: Path) -> None:
    display = PartialDependenceDisplay.from_estimator(model, X, features, kind="both")
    display.figure_.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(display.figure_)


def residual_plot(preds: np.ndarray, actuals: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actuals, preds, alpha=0.7)
    low = min(float(np.min(actuals)), float(np.min(preds)))
    high = max(float(np.max(actuals)), float(np.max(preds)))
    ax.plot([low, high], [low, high], "r--", label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
