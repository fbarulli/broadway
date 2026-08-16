"""HPO — run an Optuna study and return the best parameters."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import optuna
from sqlalchemy.exc import IntegrityError, OperationalError

_STUDY_CREATE_ATTEMPTS = 8
_STUDY_CREATE_RETRY_SECONDS = 5.0


def _create_study_with_retry(study_name: str, storage_url: str, direction: str) -> optuna.Study:
    last_error: Exception | None = None
    for _ in range(_STUDY_CREATE_ATTEMPTS):
        try:
            return optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction=direction,
            )
        except (IntegrityError, OperationalError) as exc:
            last_error = exc
            time.sleep(_STUDY_CREATE_RETRY_SECONDS)
    assert last_error is not None
    raise last_error


def run_study_rdb(
    objective: Callable[[optuna.Trial], float],
    study_name: str,
    storage_url: str,
    n_trials: int,
    direction: str = "minimize",
    random_state: int | None = None,
) -> optuna.Study:
    """Run an Optuna study persisted to an RDB storage URL and return the study."""
    study = _create_study_with_retry(study_name, storage_url, direction)
    if random_state is not None:
        study.sampler.seed = random_state
    study.optimize(objective, n_trials=n_trials)
    return study


def run_study(
    objective: Callable[[dict], float],
    search_space: dict[str, list[float | int]],
    n_trials: int,
    direction: str = "minimize",
    random_state: int | None = None,
) -> dict[str, float | int]:
    study = optuna.create_study(direction=direction)
    if random_state is not None:
        study.sampler.seed = random_state

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, float | int] = {}
        for name, (low, high) in search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)
        value = float(objective(params))
        if not np.isfinite(value):
            raise optuna.TrialPruned()
        return value

    study.optimize(_objective, n_trials=n_trials)
    try:
        return study.best_params
    except ValueError as exc:
        raise ValueError(f"HPO study produced no valid trial ({n_trials} trials): {exc}") from exc
