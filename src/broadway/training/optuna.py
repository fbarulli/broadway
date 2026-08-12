"""HPO — run an Optuna study and return the best parameters."""

from __future__ import annotations

from collections.abc import Callable

import optuna

_PARAM_NAME = "x"
_PARAM_LOW = -10.0
_PARAM_HIGH = 10.0


def run_study(
    objective: Callable[[dict], float],
    n_trials: int,
    direction: str = "minimize",
    random_state: int | None = None,
) -> dict:
    study = optuna.create_study(direction=direction)
    if random_state is not None:
        study.sampler.seed = random_state

    def _objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float(_PARAM_NAME, _PARAM_LOW, _PARAM_HIGH)
        return objective({_PARAM_NAME: x})

    study.optimize(_objective, n_trials=n_trials)
    return study.best_params
