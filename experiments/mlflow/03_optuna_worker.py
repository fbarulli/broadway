"""03: kubernetes optuna worker — per-model phase-1 study via the unified HPO API.

Reads the mounted ConfigMap (`/etc/broadway/config.yaml`) for dataset, DB and
MLflow infra, and the unified HPO spec (search spaces + budget) from
`configs/experiments/mlflow.yaml` (repo-relative — the worker image must
include it). Composes the RDB URL via the promoted
`broadway.training.optuna_worker.compose_db_url` and runs the requested model's
study through `broadway.training.hpo.run_model_study` (heartbeat + load_if_exists
included). K8s parallelizes HPO phase 1: one pod per model, each running
`initial_trials_per_model` trials. Phase-2 bandit allocation is the in-process
orchestrator's job (01, `hpo.run_hpo`). Modes: `--model ols|lgbm|xgb` (battle
display aliases, mapped via MODEL_KEYS to the registry key that the HPO spec
lookup and objective use) runs trials; `--init-only` pre-creates the studies
(optuna-init Job).
"""

import argparse
import os
import socket
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import yaml
from _common import MODEL_KEYS
from sklearn.model_selection import train_test_split

from broadway.config.schema import HPOConfig
from broadway.training import hpo
from broadway.training.mlflow_utils import (
    log_dataset,
    log_metadata,
    log_params,
    setup_mlflow,
)
from broadway.training.optuna_worker import compose_db_url

# HPO search spaces + budgets (configs/experiments/mlflow.yaml -> `hpo`).
# Env-overridable: the k8s worker image sets BROADWAY_MLFLOW_CONFIG to its
# mounted location; local default = repo path. Branch on env presence —
# os.environ.get(default) evaluates the default eagerly, and the parents[2]
# fallback raises at pod depth (/app/worker.py has only two parents).
if "BROADWAY_MLFLOW_CONFIG" in os.environ:
    HPO_CONFIG_PATH = Path(os.environ["BROADWAY_MLFLOW_CONFIG"])
else:
    HPO_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "mlflow.yaml"


def load_secret(secret_dir: str) -> dict:
    """Read Secret files (one file per key) from the mounted volume."""
    root = Path(secret_dir)
    return {name: (root / name).read_text().strip()
            for name in ("DB_USER", "DB_PASSWORD", "DB_NAME")}


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Metered trips from the config parquet with the config filters."""
    ds = cfg["dataset"]
    df = pd.read_parquet(ds["parquet"])
    df = df[df["fare_amount"] > ds["min_fare"]]
    duration = ((df[ds["dropoff_datetime"]] - df[ds["pickup_datetime"]])
                .dt.total_seconds() / 60)
    keep = (duration > 0) & (duration < ds["max_duration_minutes"])
    df = df[keep]
    df["duration_minutes"] = duration[keep]
    df["pickup_hour"] = df[ds["pickup_datetime"]].dt.hour
    df["pickup_weekday"] = df[ds["pickup_datetime"]].dt.weekday
    return df


def _dummy_objective(params: dict) -> float:
    """Placeholder objective — init-only materializes studies, never evaluates."""
    return 0.0


def log_endpoints(model_name: str, cfg: dict, db_url: str) -> None:
    """Log the actual endpoints in use (visibility requirement)."""
    host = socket.gethostname()
    db = cfg["databases"]["optuna"]
    print(f"[worker] model={model_name} hostname={host} "
          f"ip={socket.gethostbyname(host)} "
          f"db={db['host']}:{db['port']}/{db['name']} "
          f"tracking_uri={cfg['mlflow']['tracking_uri']}")
    print(f"[worker] db_url={db_url}")


def log_to_mlflow(
    model_name: str,
    key: str,
    best: optuna.trial.FrozenTrial,
    cfg: dict,
    dataset_path: str,
    n_trials: int,
    study_name: str,
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Log the best trial (params, metrics, lineage, metadata) + model artifacts."""
    setup_mlflow(cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"optuna_{model_name}"):
        log_params(best.params)
        mlflow.log_metric("mae", float(best.value))
        log_dataset(cfg["dataset"]["name"], dataset_path, context="train")
        log_metadata({"n_trials": float(n_trials)})
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("study", study_name)
        mlflow.set_tag("seed", str(seed))
        hpo.log_best_artifacts(key, best.params, X_train, y_train, X_val, y_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ols", choices=sorted(MODEL_KEYS))
    parser.add_argument("--config", default="/etc/broadway/config.yaml")
    parser.add_argument("--secret-dir", default="/etc/broadway/secret")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mlflow_cfg = yaml.safe_load(HPO_CONFIG_PATH.read_text())
    hpo_cfg = HPOConfig(**mlflow_cfg["hpo"])
    seed = int(mlflow_cfg["seed"])
    test_fraction = float(mlflow_cfg["test_fraction"])
    secret = load_secret(args.secret_dir)
    db = cfg["databases"]["optuna"]
    db_url = compose_db_url(db["driver"], secret["DB_USER"], secret["DB_PASSWORD"],
                            db["host"], db["port"], db["name"])
    log_endpoints(args.model, cfg, db_url)

    if args.init_only:
        for spec in hpo_cfg.models:
            hpo.run_model_study(spec, _dummy_objective, n_trials=0,
                                random_state=seed, storage_url=db_url,
                                direction=hpo_cfg.direction)
        print("[worker] init complete")
        return

    ds = cfg["dataset"]
    df = load_dataset(cfg)
    X = df[ds["features"]]
    X_train, X_val, y_train, y_val = train_test_split(
        X, df[ds["target"]], test_size=test_fraction, random_state=seed)
    # Resolve the display alias to the registry key — the spec lookup and the
    # objective MUST use the canonical key (the config's hpo.models are keys).
    key = MODEL_KEYS[args.model]
    spec = next(s for s in hpo_cfg.models if s.name == key)
    objective = hpo.make_objective(
        model_type=key, target_metric=hpo_cfg.target_metric,
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    study_name = f"hpo-{key}"
    # Point the per-trial callback at the configmap tracking store + experiment
    # (same setup the final best run uses) before the study runs.
    setup_mlflow(cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment"])
    study = hpo.run_model_study(
        spec, objective, n_trials=hpo_cfg.initial_trials_per_model,
        random_state=seed, storage_url=db_url, direction=hpo_cfg.direction,
        mlflow_tracking=True,
        mlflow_tags={"model": args.model, "study": study_name, "seed": str(seed)})
    best = study.best_trial
    log_to_mlflow(args.model, key, best, cfg, ds["parquet"],
                  hpo_cfg.initial_trials_per_model, study_name, seed,
                  X_train, y_train, X_val, y_val)
    print(f"[worker] DONE model={args.model} best_mae={best.value:.4f} "
          f"params={best.params}")


if __name__ == "__main__":
    main()
