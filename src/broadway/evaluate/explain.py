"""Generic model-explainability helpers: SHAP, permutation importance, PDP/ICE, LIME, residuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def shap_summary(
    model: object,
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
    model: object,
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


def pdp_ice(model: object, X: pd.DataFrame, features: list[str], out_path: Path) -> None:
    display = PartialDependenceDisplay.from_estimator(model, X, features, kind="both")
    display.figure_.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(display.figure_)


def lime_explanation(
    model: object,
    X_train: pd.DataFrame,
    row: pd.Series,
    feature_names: list[str],
    out_path: Path,
) -> None:
    explainer = LimeTabularExplainer(
        X_train.to_numpy(),
        feature_names=feature_names,
        mode="regression",
        random_state=0,
        verbose=False,
    )
    explanation = explainer.explain_instance(
        row.to_numpy(), model.predict, num_features=min(3, len(feature_names))
    )
    items = explanation.as_list()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([name for name, _ in items], [weight for _, weight in items])
    ax.set_xlabel("Weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
