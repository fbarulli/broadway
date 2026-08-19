"""Unified config-driven HPO — parallel per-model studies with bandit allocation.

Two rounds: every model gets `initial_trials_per_model` trials (run in
parallel), a leaderboard ranks the models by best objective, and the remaining
budget goes to the top-k models. Deterministic: each per-model study seeds its
TPE sampler at construction with `random_state + model_index`, so identical
search spaces still diverge deterministically and parallel == sequential.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd

from broadway.config.schema import HPOConfig, ModelHPOSpec
from broadway.evaluate.metrics import compute_metrics
from broadway.training.models.registry import get_model
from broadway.training.optuna import GRACE_PERIOD, HEARTBEAT_INTERVAL

logger = logging.getLogger(__name__)

# Params-dict objective that may also receive the active optuna trial so the
# trial can carry the full metric set as a user attr (mlflow per-trial logging).
Objective = Callable[[dict[str, float | int], optuna.Trial | None], float]


def make_objective(
    model_type: str,
    target_metric: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Objective:
    """Build the HPO objective: fit a model with params, score target metric on val.

    The returned objective takes the active optuna trial as an optional second
    argument and, when given, attaches the FULL compute_metrics set to the
    trial as the "broadway_metrics" user attr (the per-trial mlflow callback
    logs those metrics for every trial).
    """

    def objective(
        params: dict[str, float | int], trial: optuna.Trial | None = None
    ) -> float:
        model = get_model(model_type, **params)
        model.fit(X_train, y_train)
        metrics = compute_metrics(y_val.to_numpy(), model.predict(X_val))
        if trial is not None:
            trial.set_user_attr(
                "broadway_metrics", {name: float(v) for name, v in metrics.items()}
            )
        return metrics[target_metric]

    return objective


def _mlflow_callback(
    study_name: str, tags: dict[str, str]
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Build an optuna callback: one nested mlflow run per COMPLETE trial.

    Each run logs the trial params, the full "broadway_metrics" user-attr set
    (attached by make_objective) plus the target metric, and the study tags.
    Nested runs group the trials under the study's best run in the experiment
    page; the tracking URI is the ambient MLFLOW_TRACKING_URI (caller-set).
    """

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return
        with mlflow.start_run(
            run_name=f"{study_name} trial {trial.number}",
            tags={**tags, "trial": str(trial.number), "study": study_name},
            nested=True,
        ):
            mlflow.log_params(trial.params)
            metrics = trial.user_attrs.get("broadway_metrics", {})
            numeric = {
                name: float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float))
            }
            if numeric:
                mlflow.log_metrics(numeric)
            mlflow.log_metric("target_metric", float(trial.value))

    return callback


def log_best_artifacts(
    model_type: str,
    params: dict[str, float | int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Log best-run artifacts to the active mlflow run.

    Refits the model with the best params on train, logs it via its mlflow
    flavor (lgbm/xgb, sklearn for the rest), writes the val predictions CSV,
    and plots the feature importances of tree models. A model without a
    registered flavor is skipped with a warning; failures elsewhere bubble up.
    """
    model = get_model(model_type, **params)
    model.fit(X_train, y_train)
    try:
        if model_type == "lgbm":
            mlflow.lightgbm.log_model(model, "model")
        elif model_type == "xgb":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
    except Exception as exc:
        logger.warning(
            "no mlflow model flavor for %s, skipping model artifact: %s",
            model_type,
            exc,
        )
    with tempfile.TemporaryDirectory() as tmp:
        preds = model.predict(X_val)
        csv_path = Path(tmp) / "predictions.csv"
        pd.DataFrame({"actual": y_val.to_numpy(), "predicted": preds}).to_csv(
            csv_path, index=False
        )
        mlflow.log_artifact(str(csv_path))
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(importance)), importance)
            ax.set_title(f"{model_type} feature importance")
            fig.tight_layout()
            plot_path = Path(tmp) / "feature_importance.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path))


def _trial_objective(
    model: ModelHPOSpec,
    objective: Objective,
) -> Callable[[optuna.Trial], float]:
    """Wrap the params-dict objective into an Optuna trial objective."""

    def run(trial: optuna.Trial) -> float:
        params: dict[str, float | int] = {}
        for name, (low, high) in model.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)
        value = float(objective(params, trial))
        if not np.isfinite(value):
            raise optuna.TrialPruned()
        return value

    return run


def run_model_study(
    model: ModelHPOSpec,
    objective: Objective,
    n_trials: int,
    random_state: int,
    storage_url: str | None = None,
    direction: str = "minimize",
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> optuna.Study:
    """Run one model's study over its search space and return the study.

    The TPE sampler is seeded at construction — Optuna 4.x reads the seed only
    from the constructor, so the same (model, random_state) reproduces the same
    trajectory. With storage_url the study persists in RDB storage (the
    heartbeat pattern from optuna.run_study_rdb) and reopens load_if_exists.
    With mlflow_tracking, every COMPLETE trial is logged as a nested mlflow run
    (params, full metrics, tags) against the ambient MLFLOW_TRACKING_URI.
    """
    study_name = f"hpo-{model.name}"
    sampler = optuna.samplers.TPESampler(seed=random_state)
    if storage_url is None:
        study = optuna.create_study(direction=direction, sampler=sampler)
    else:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=int(HEARTBEAT_INTERVAL),
            grace_period=int(GRACE_PERIOD),
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
        )
    callbacks = (
        [_mlflow_callback(study_name, mlflow_tags or {})] if mlflow_tracking else None
    )
    study.optimize(_trial_objective(model, objective), n_trials=n_trials, callbacks=callbacks)
    return study


def _optimize_study(
    study: optuna.Study,
    model: ModelHPOSpec,
    objective: Objective,
    n_trials: int,
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> None:
    """Append n_trials to an existing study (bandit round continuation)."""
    callbacks = (
        [_mlflow_callback(f"hpo-{model.name}", mlflow_tags or {})]
        if mlflow_tracking
        else None
    )
    study.optimize(_trial_objective(model, objective), n_trials=n_trials, callbacks=callbacks)


def bandit_allocate(
    leaderboard: dict[str, float],
    remaining: int,
    top_k: int,
) -> dict[str, int]:
    """Allocate the remaining trial budget to the top-k models (minimize).

    Pure function: leaderboard maps model name to best objective value (lower
    is better). The budget is split evenly across the top-k, with the remainder
    going to the best model; models outside the top-k get no allocation.
    Guards: top_k >= len selects everyone, remaining <= 0 or an empty
    leaderboard yields an empty allocation.
    """
    if remaining <= 0 or not leaderboard or top_k <= 0:
        return {}
    ranked = [name for name, _ in sorted(leaderboard.items(), key=lambda item: item[1])]
    selected = ranked[: min(top_k, len(ranked))]
    base, remainder = divmod(remaining, len(selected))
    allocation = {name: base for name in selected}
    for name in selected[:remainder]:
        allocation[name] += 1
    return allocation


def _initial_round(
    hpo: HPOConfig,
    objectives: dict[str, Objective],
    random_state: int,
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> dict[str, optuna.Study]:
    """Run initial_trials_per_model per model in parallel, seeded per model."""
    specs = {spec.name: spec for spec in hpo.models}
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {}
        for index, name in enumerate(specs):
            tags = dict(mlflow_tags or {})
            tags["model"] = name
            futures[name] = pool.submit(
                run_model_study,
                specs[name],
                objectives[name],
                hpo.initial_trials_per_model,
                random_state + index,
                hpo.storage_url,
                hpo.direction,
                mlflow_tracking,
                tags,
            )
        return {name: future.result() for name, future in futures.items()}


def _bandit_round(
    hpo: HPOConfig,
    objectives: dict[str, Objective],
    studies: dict[str, optuna.Study],
    allocation: dict[str, int],
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> None:
    """Append the bandit-allocated trials to the selected models' studies."""
    specs = {spec.name: spec for spec in hpo.models}
    with ThreadPoolExecutor(max_workers=len(allocation)) as pool:
        futures = {}
        for name, trials in allocation.items():
            tags = dict(mlflow_tags or {})
            tags["model"] = name
            futures[name] = pool.submit(
                _optimize_study,
                studies[name],
                specs[name],
                objectives[name],
                trials,
                mlflow_tracking,
                tags,
            )
        for future in futures.values():
            future.result()


def _leaderboard(studies: dict[str, optuna.Study]) -> dict[str, float]:
    """Best objective value per study, skipping studies with no valid trial."""
    best: dict[str, float] = {}
    for name, study in studies.items():
        try:
            best[name] = float(study.best_value)
        except ValueError:
            continue
    return best


def run_hpo(
    hpo: HPOConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int,
    mlflow_tracking: bool = False,
    mlflow_tags: dict[str, str] | None = None,
) -> dict:
    """Run the two-round bandit HPO and return per-model results and the best.

    Round 1 gives every model initial_trials_per_model trials in parallel
    (seeds random_state + model_index); round 2 allocates the remaining budget
    to the top-k models via bandit_allocate and appends their trials. Returns
    {"models": {name: {best_params, best_value, n_trials}}, "best_model",
    "best_params", "best_value"}; models without a valid trial are omitted.
    The leaderboard and bandit allocation assume the minimize direction.
    With mlflow_tracking, every COMPLETE trial across both rounds is logged as
    a nested mlflow run tagged with the model plus the caller's mlflow_tags
    (e.g. {"experiment": ...}); the tracking URI stays ambient.
    """
    if not hpo.models:
        raise ValueError("hpo requires at least one model in `models`")
    objectives = {
        spec.name: make_objective(spec.name, hpo.target_metric, X_train, y_train, X_val, y_val)
        for spec in hpo.models
    }
    studies = _initial_round(hpo, objectives, random_state, mlflow_tracking, mlflow_tags)
    leaderboard = _leaderboard(studies)
    remaining = hpo.total_trials - hpo.initial_trials_per_model * len(hpo.models)
    allocation = bandit_allocate(leaderboard, remaining, hpo.top_k)
    if allocation:
        _bandit_round(hpo, objectives, studies, allocation, mlflow_tracking, mlflow_tags)
    if not leaderboard:
        raise ValueError("hpo produced no valid trial for any model")
    models = {
        name: {
            "best_params": dict(study.best_params),
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
        }
        for name, study in studies.items()
        if name in leaderboard
    }
    best_model = min(leaderboard, key=lambda name: leaderboard[name])
    return {
        "models": models,
        "best_model": best_model,
        "best_params": dict(studies[best_model].best_params),
        "best_value": float(studies[best_model].best_value),
    }
