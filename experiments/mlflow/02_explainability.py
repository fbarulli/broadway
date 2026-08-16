"""02: explainability suite — SHAP, permutation, PDP/ICE, LIME, built-ins, residuals.

Retrains OLS / LGBM / XGBoost on the same 1000-row holdout (numeric
features only, for clean interpretability) and produces: TreeSHAP (LGBM /
XGB) and KernelSHAP (OLS) summary plots, permutation importance,
PDP + ICE, one LIME explanation, built-in importances / coefficients, and
residual diagnostics. All figures are logged to MLflow as artifacts
(artifacts are fine — the model pkl is not tracked).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from _common import (
    CONTINUOUS_FEATURES,
    MLRUNS,
    RESULTS,
    SEED,
    TEST_FRACTION,
    load_sample,
)
from broadway.training.mlflow_utils import setup_mlflow

EXPERIMENT = "ratecode1_model_battle"
N_BACKGROUND = 100  # KernelSHAP background + explained sample sizes
N_PERM_REPEATS = 5


def load_numeric_split():
    """Seeded 80/20 split on the numeric features only (floats for PDP)."""
    df = load_sample()
    X = df[CONTINUOUS_FEATURES].astype(float)
    return train_test_split(X, df["fare_amount"], test_size=TEST_FRACTION,
                            random_state=SEED)


def train_models(X_train, y_train) -> dict:
    """Raw numeric-feature models: ols (LinearRegression), lgbm, xgb."""
    return {
        "ols": LinearRegression().fit(X_train, y_train),
        "lgbm": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                                  max_depth=5, random_state=SEED,
                                  verbosity=-1).fit(X_train, y_train),
        "xgb": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=5, random_state=SEED,
                                tree_method="hist").fit(X_train, y_train),
    }


def shap_summary(model, X_train, X_test, name: str, out_dir: Path) -> None:
    """TreeSHAP for trees, KernelSHAP for linear; saves a beeswarm."""
    if name == "ols":
        bg = X_train.sample(N_BACKGROUND, random_state=SEED)
        explained = X_test.sample(N_BACKGROUND, random_state=SEED)
        explainer = shap.KernelExplainer(model.predict, bg)
        values = explainer.shap_values(explained)
    else:
        explained = X_test
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(explained)
    shap.summary_plot(values, explained, show=False,
                      max_display=len(CONTINUOUS_FEATURES))
    out = out_dir / f"02_explainability_shap_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def importance_rows(models: dict, X_test, y_test) -> list[dict]:
    """Permutation importance (mean) + built-in importance / coef per feature."""
    rows = []
    for name, model in models.items():
        perm = permutation_importance(model, X_test, y_test,
                                      n_repeats=N_PERM_REPEATS,
                                      random_state=SEED)
        builtin = (model.coef_ if name == "ols"
                   else np.asarray(model.feature_importances_, dtype=float))
        for i, feat in enumerate(CONTINUOUS_FEATURES):
            rows.append({
                "model": name,
                "feature": feat,
                "permutation_importance": float(perm.importances_mean[i]),
                "builtin_importance": float(builtin[i]),
            })
    return rows


def plot_permutation(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, model in zip(axes, ("ols", "lgbm", "xgb")):
        sub = [r for r in rows if r["model"] == model]
        ax.barh([r["feature"] for r in sub][::-1],
                [r["permutation_importance"] for r in sub][::-1],
                color="#4C72B0")
        ax.set_title(f"{model} — permutation importance")
        ax.set_xlabel("MAE increase ($)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pdp(model, X_test, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    PartialDependenceDisplay.from_estimator(
        model, X_test, CONTINUOUS_FEATURES, kind="both", ax=ax,
        random_state=SEED)
    fig.suptitle("LGBM — PDP (solid) + ICE (faint)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def lime_explanation(model, X_train, X_test, out: Path) -> None:
    explainer = LimeTabularExplainer(
        X_train.to_numpy(), feature_names=CONTINUOUS_FEATURES,
        mode="regression", random_state=SEED, verbose=False)
    exp = explainer.explain_instance(
        X_test.iloc[0].to_numpy(), model.predict, num_features=3)
    labels, weights = zip(*exp.as_list())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(labels)[::-1], list(weights)[::-1], color="#DD8452")
    ax.set_title(f"LIME explanation for one trip (pred ${model.predict(X_test.iloc[[0]])[0]:.2f})")
    ax.set_xlabel("weight toward prediction")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_residuals(models: dict, X_test, y_test, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (name, model) in enumerate(models.items()):
        preds = model.predict(X_test)
        resid = y_test.to_numpy() - preds
        axes[0, col].scatter(preds, resid, s=8, alpha=0.4, color="black")
        axes[0, col].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[0, col].set_title(f"{name} — residuals vs fitted")
        axes[0, col].set_xlabel("predicted ($)")
        axes[0, col].set_ylabel("residual ($)")
        axes[1, col].hist(resid, bins=40, color="#4C72B0")
        axes[1, col].set_title(f"{name} — residual histogram")
        axes[1, col].set_xlabel("residual ($)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    setup_mlflow(str(MLRUNS), EXPERIMENT)
    RESULTS.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_numeric_split()
    models = train_models(X_train, y_train)

    with mlflow.start_run(run_name="explainability"):
        mlflow.log_params({"features": ",".join(CONTINUOUS_FEATURES),
                           "n_train": len(X_train), "n_test": len(X_test)})
        for name in models:
            shap_summary(models[name], X_train, X_test, name, RESULTS)

        rows = importance_rows(models, X_test, y_test)
        plot_permutation(rows, RESULTS / "02_explainability_permutation.png")
        plot_pdp(models["lgbm"], X_test, RESULTS / "02_explainability_pdp_ice.png")
        lime_explanation(models["lgbm"], X_train, X_test,
                         RESULTS / "02_explainability_lime.png")
        plot_residuals(models, X_test, y_test,
                       RESULTS / "02_explainability_residuals.png")

        for png in RESULTS.glob("02_explainability_*.png"):
            mlflow.log_artifact(str(png))
            print(f"logged {png.name}")

    csv = RESULTS / "02_explainability_importance.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")
    print("\n=== permutation + built-in importance ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
