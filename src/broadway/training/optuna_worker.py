"""Optuna RDB worker — run HPO trials against shared storage from a k8s worker pod.

Deployment pattern: every worker pod builds its DB URL from config parts (driver, user,
password, host, port, name) via compose_db_url — never from environment variables — so all
pods point at the same Optuna storage and share one study per study_name. run_worker drives
run_study_rdb for the pod's trial budget and returns the best trial's params so the
orchestrator can collect results without touching the storage directly.
"""

from __future__ import annotations

from collections.abc import Callable

import optuna

from broadway.training.optuna import run_study_rdb


def compose_db_url(driver: str, user: str, password: str, host: str, port: str, name: str) -> str:
    """Compose a SQLAlchemy storage URL from config parts."""
    return f"{driver}://{user}:{password}@{host}:{port}/{name}"


def run_worker(
    study_name: str,
    db_url: str,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    random_state: int,
) -> dict:
    """Run one worker's share of a shared study and return the best trial's params."""
    study = run_study_rdb(objective, study_name, db_url, n_trials, random_state=random_state)
    try:
        return dict(study.best_params)
    except ValueError as exc:
        raise ValueError(f"worker study {study_name!r} produced no valid trial: {exc}") from exc
