"""Unified config-driven HPO — parallel per-model studies with bandit allocation.

Two rounds: every model gets `initial_trials_per_model` trials (run in
parallel), a leaderboard ranks the models by best objective, and the remaining
budget goes to the top-k models. Deterministic: each per-model study seeds its
TPE sampler at construction with `random_state + model_index`, so identical
search spaces still diverge deterministically and parallel == sequential.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import optuna
import pandas as pd

from broadway.config.schema import HPOConfig, ModelHPOSpec
from broadway.evaluate.metrics import compute_metrics
from broadway.training.models.registry import get_model
from broadway.training.optuna import GRACE_PERIOD, HEARTBEAT_INTERVAL

logger = logging.getLogger(__name__)


def make_objective(
    model_type: str,
    target_metric: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Callable[[dict[str, float | int]], float]:
    """Build the HPO objective: fit a model with params, score target metric on val."""

    def objective(params: dict[str, float | int]) -> float:
        model = get_model(model_type, **params)
        model.fit(X_train, y_train)
        metrics = compute_metrics(y_val.to_numpy(), model.predict(X_val))
        return metrics[target_metric]

    return objective


def _trial_objective(
    model: ModelHPOSpec,
    objective: Callable[[dict[str, float | int]], float],
) -> Callable[[optuna.Trial], float]:
    """Wrap the params-dict objective into an Optuna trial objective."""

    def run(trial: optuna.Trial) -> float:
        params: dict[str, float | int] = {}
        for name, (low, high) in model.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)
        value = float(objective(params))
        if not np.isfinite(value):
            raise optuna.TrialPruned()
        return value

    return run


def run_model_study(
    model: ModelHPOSpec,
    objective: Callable[[dict[str, float | int]], float],
    n_trials: int,
    random_state: int,
    storage_url: str | None = None,
    direction: str = "minimize",
) -> optuna.Study:
    """Run one model's study over its search space and return the study.

    The TPE sampler is seeded at construction — Optuna 4.x reads the seed only
    from the constructor, so the same (model, random_state) reproduces the same
    trajectory. With storage_url the study persists in RDB storage (the
    heartbeat pattern from optuna.run_study_rdb) and reopens load_if_exists.
    """
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
            study_name=f"hpo-{model.name}",
            storage=storage,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
        )
    study.optimize(_trial_objective(model, objective), n_trials=n_trials)
    return study


def _optimize_study(
    study: optuna.Study,
    model: ModelHPOSpec,
    objective: Callable[[dict[str, float | int]], float],
    n_trials: int,
) -> None:
    """Append n_trials to an existing study (bandit round continuation)."""
    study.optimize(_trial_objective(model, objective), n_trials=n_trials)


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
    objectives: dict[str, Callable[[dict[str, float | int]], float]],
    random_state: int,
) -> dict[str, optuna.Study]:
    """Run initial_trials_per_model per model in parallel, seeded per model."""
    specs = {spec.name: spec for spec in hpo.models}
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {
            name: pool.submit(
                run_model_study,
                specs[name],
                objectives[name],
                hpo.initial_trials_per_model,
                random_state + index,
                hpo.storage_url,
                hpo.direction,
            )
            for index, name in enumerate(specs)
        }
        return {name: future.result() for name, future in futures.items()}


def _bandit_round(
    hpo: HPOConfig,
    objectives: dict[str, Callable[[dict[str, float | int]], float]],
    studies: dict[str, optuna.Study],
    allocation: dict[str, int],
) -> None:
    """Append the bandit-allocated trials to the selected models' studies."""
    specs = {spec.name: spec for spec in hpo.models}
    with ThreadPoolExecutor(max_workers=len(allocation)) as pool:
        futures = {
            name: pool.submit(
                _optimize_study,
                studies[name],
                specs[name],
                objectives[name],
                trials,
            )
            for name, trials in allocation.items()
        }
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
) -> dict:
    """Run the two-round bandit HPO and return per-model results and the best.

    Round 1 gives every model initial_trials_per_model trials in parallel
    (seeds random_state + model_index); round 2 allocates the remaining budget
    to the top-k models via bandit_allocate and appends their trials. Returns
    {"models": {name: {best_params, best_value, n_trials}}, "best_model",
    "best_params", "best_value"}; models without a valid trial are omitted.
    The leaderboard and bandit allocation assume the minimize direction.
    """
    if not hpo.models:
        raise ValueError("hpo requires at least one model in `models`")
    objectives = {
        spec.name: make_objective(spec.name, hpo.target_metric, X_train, y_train, X_val, y_val)
        for spec in hpo.models
    }
    studies = _initial_round(hpo, objectives, random_state)
    leaderboard = _leaderboard(studies)
    remaining = hpo.total_trials - hpo.initial_trials_per_model * len(hpo.models)
    allocation = bandit_allocate(leaderboard, remaining, hpo.top_k)
    if allocation:
        _bandit_round(hpo, objectives, studies, allocation)
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
