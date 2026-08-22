"""01: MLflow model battle — OLS / LGBM / XGBoost (+ Ridge / RandomForest in parallel).

Every model run is tracked in MLflow (local file store `mlruns/`): the FULL
constructor params (the recipe — NOT the fitted artifact; the model pkl is
never logged), every regression metric plus binarized ROC/PR AUC, and
metadata (training time, prediction time, model size in bytes). Uses a
seeded 80/20 holdout on a 1000-row sample through sklearn Pipelines (the
categorical branch is ready for future pipeline steps), RFECV feature
selection with a plot, and a config-driven HPO bandit (lgbm/xgb/linear) via
the unified `broadway.training.hpo.run_hpo` — models are named by REGISTRY
key in `configs/experiments/mlflow.yaml` (`hpo.models`), defaults come from
the registry, and the final best is logged to MLflow.
"""

import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from _common import (
    BONUS_MODELS,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    MLRUNS,
    MODEL_KEYS,
    REPO,
    RESULTS,
    SEED,
    battle_pipeline_config,
    binary_threshold,
    load_sample,
    make_pipeline,
    split_data,
)
from joblib import Parallel, delayed
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

from broadway.config.schema import HPOConfig
from broadway.evaluate.metrics import binary_metrics, compute_metrics
from broadway.training import hpo
from broadway.training.mlflow_utils import (
    log_metrics,
    log_params,
    setup_mlflow,
)
from broadway.training.models.registry import display_name, get_model

EXPERIMENT = "ratecode1_model_battle"
# Platform metrics default to 4 decimals, but the battle CSV writer rounds to
# 6 — keep full precision so this refactor produces identical numbers.
METRIC_DECIMALS = 6


def fit_and_evaluate(name: str, model: object,
                     X_train, X_test, y_train, y_test, threshold: float) -> dict:
    """Fit pipeline, time it, compute holdout metrics + size; returns summary."""
    pipe = make_pipeline(model)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t1 = time.perf_counter()
    preds = pipe.predict(X_test)
    t2 = time.perf_counter()
    metrics = compute_metrics(y_test, preds, decimals=METRIC_DECIMALS)
    metrics.update(binary_metrics(y_test, preds, threshold,
                                  decimals=METRIC_DECIMALS))
    return {
        "name": name,
        "params": pipe.named_steps["model"].get_params(),
        "train_time_s": round(t1 - t0, 4),
        "predict_time_s": round(t2 - t1, 4),
        "model_size_bytes": len(pickle.dumps(pipe)),
        "metrics": metrics,
    }


def log_run(summary: dict, n_train: int, n_test: int) -> None:
    """One MLflow run per model: params, metrics, metadata — no artifact."""
    with mlflow.start_run(run_name=summary["name"]):
        log_params(summary["params"])
        log_params({
            "features": ",".join(CONTINUOUS_FEATURES + CATEGORICAL_FEATURES),
            "n_train": n_train,
            "n_test": n_test,
        })
        log_metrics(summary["metrics"])
        log_metrics({
            "train_time_s": summary["train_time_s"],
            "predict_time_s": summary["predict_time_s"],
            "model_size_bytes": summary["model_size_bytes"],
        })
        mlflow.set_tag("model", summary["name"])


def rfe_curve(name: str, model: object, X_train, y_train) -> dict:
    """RFECV over the model; returns (transformed n_features, cv MAE)."""
    pre = make_pipeline(model).named_steps["pre"]
    X_num = pre.fit_transform(X_train)
    selector = RFECV(model, step=1, cv=3,
                     scoring="neg_mean_absolute_error",
                     min_features_to_select=1)
    selector.fit(X_num, y_train)
    return {
        "name": name,
        "n_features": [int(v) for v in selector.cv_results_["n_features"]],
        "cv_mae": list(-selector.cv_results_["mean_test_score"]),
    }


def plot_metrics(summaries: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [s["name"] for s in summaries]
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [s["metrics"]["mae"] for s in summaries],
           width, label="MAE", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], [s["metrics"]["rmse"] for s in summaries],
           width, label="RMSE", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("$ (holdout)")
    ax.set_title("Model battle — MAE / RMSE (1000-row sample, 80/20 holdout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_rfe(curves: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in curves:
        ax.plot(c["n_features"], c["cv_mae"], marker="o", label=c["name"])
    ax.set_xlabel("number of transformed features (RFECV)")
    ax.set_ylabel("CV MAE ($)")
    ax.set_title("Recursive feature elimination — holdout MAE vs features")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    setup_mlflow(str(MLRUNS), EXPERIMENT)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    X_train, X_test, y_train, y_test = split_data(df)
    threshold = binary_threshold(y_train.to_numpy())
    print(f"sample: {len(df)} rows | train {len(X_train)} / test {len(X_test)}")

    # Battle comparison fits: registry defaults apply via get_model(key).
    summaries = [
        fit_and_evaluate(display_name(key), get_model(key), X_train, X_test,
                         y_train, y_test, threshold)
        for key in MODEL_KEYS.values()
    ]
    for s in summaries:
        log_run(s, len(X_train), len(X_test))
    print("\n=== MLflow runs (OLS / LGBM / XGB) ===")
    for s in summaries:
        m = s["metrics"]
        print(f"{s['name']:<6} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
              f"R2={m['r2']:.4f} ROC={m['roc_auc']:.3f} PR={m['pr_auc']:.3f} "
              f"train={s['train_time_s']}s pred={s['predict_time_s']}s "
              f"size={s['model_size_bytes']}B")

    # Bonus models computed in parallel (joblib), logged sequentially
    bonus = Parallel(n_jobs=-1)(
        delayed(fit_and_evaluate)(name, factory(**params), X_train, X_test,
                                  y_train, y_test, threshold)
        for name, (factory, params) in BONUS_MODELS.items()
    )
    for s in bonus:
        log_run(s, len(X_train), len(X_test))
        print(f"{s['name']:<6} MAE={s['metrics']['mae']:.3f} "
              f"RMSE={s['metrics']['rmse']:.3f} (parallel bonus)")
    summaries += bonus

    metrics_csv = RESULTS / "01_model_battle_metrics.csv"
    rows = []
    for s in summaries:
        rows.append({
            "model": s["name"],
            **{k: round(v, 6) for k, v in s["metrics"].items()},
            "train_time_s": s["train_time_s"],
            "predict_time_s": s["predict_time_s"],
            "model_size_bytes": s["model_size_bytes"],
        })
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"wrote {metrics_csv}")

    plot_metrics(summaries, RESULTS / "01_model_battle_metrics.png")

    curves = [
        rfe_curve(display_name(key), get_model(key), X_train, y_train)
        for key in MODEL_KEYS.values()
    ]
    rfe_csv = RESULTS / "01_model_battle_rfe.csv"
    pd.DataFrame([
        {"model": c["name"], "n_features": n, "cv_mae": mae}
        for c in curves for n, mae in zip(c["n_features"], c["cv_mae"])
    ]).to_csv(rfe_csv, index=False)
    plot_rfe(curves, RESULTS / "01_model_battle_rfe.png")
    print(f"wrote {rfe_csv}")

    print("\n=== HPO bandit (unified API, config-driven spaces) ===")
    raw_hpo = yaml.safe_load(
        (REPO / "configs" / "experiments" / "mlflow.yaml").read_text())["hpo"]
    # The config already names models by REGISTRY key (the single canonical
    # name) — no remap needed; run_hpo -> make_objective -> get_model resolves
    # them directly. Its objective fits raw registry models, so feed it the
    # encoded X (categoricals one-hot via the battle pipeline preprocessor).
    hpo_cfg = HPOConfig(**raw_hpo)
    pre = make_pipeline(LinearRegression()).named_steps["pre"]
    X_train_enc = pre.fit_transform(X_train)
    X_val_enc = pre.transform(X_test)
    result = hpo.run_hpo(battle_pipeline_config(), hpo_cfg, X_train_enc, y_train,
                         X_val_enc, y_test, SEED,
                         mlflow_tracking=True,
                         mlflow_tags={"experiment": "ratecode1_model_battle"})
    with mlflow.start_run(run_name="hpo_bandit"):
        log_params(result["best_params"])
        mlflow.log_metric("mae", result["best_value"])
        mlflow.set_tag("model", display_name(result["best_model"]))
        hpo.log_best_artifacts(battle_pipeline_config(), result["best_model"],
                               result["best_params"], X_train_enc, y_train,
                               X_val_enc, y_test)
    print("leaderboard (per-model best MAE):")
    for name, res in result["models"].items():
        print(f"{display_name(name):<6} best_mae={res['best_value']:.4f} "
              f"n_trials={res['n_trials']}")
    best = result["best_model"]
    print(f"best model: {display_name(best)} | "
          f"best MAE: {result['best_value']:.4f} | "
          f"params: {result['best_params']}")
    pd.DataFrame([
        {"model": display_name(name), "best_mae": res["best_value"],
         **res["best_params"]}
        for name, res in result["models"].items()
    ]).to_csv(RESULTS / "01_model_battle_optuna.csv", index=False)
    print(f"wrote {RESULTS / '01_model_battle_optuna.csv'}")


if __name__ == "__main__":
    main()
