"""HPO — run an Optuna study and return the best parameters."""

from __future__ import annotations

from collections.abc import Callable

import optuna


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
        return objective(params)

    study.optimize(_objective, n_trials=n_trials)
    return study.best_params
