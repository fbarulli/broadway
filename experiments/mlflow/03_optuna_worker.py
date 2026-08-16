"""03: kubernetes optuna worker — config FILES only (no env vars).

Reads the mounted ConfigMap (`/etc/broadway/config.yaml`) and Secret files
(`/etc/broadway/secret/<KEY>`), composes the RDB URL via the promoted
`broadway.training.optuna_worker.compose_db_url`, runs storage-backed trials
via `run_study_rdb` (schema-race retry included), and logs the run to MLflow
via `mlflow_utils.log_metadata` / `log_dataset`. Logs the resolved endpoints
at startup. Modes: `--model ols|lgbm|xgb` runs trials; `--init-only` creates
the configured studies once (run by the optuna-init Job).
"""

import argparse
import socket
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from broadway.training.mlflow_utils import (
    log_dataset,
    log_metadata,
    log_params,
    setup_mlflow,
)
from broadway.training.models.registry import get_model
from broadway.training.optuna import run_study_rdb
from broadway.training.optuna_worker import compose_db_url

REGISTRY_KEY = {"ols": "linear", "lgbm": "lgbm", "xgb": "xgb"}


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


def make_objective(model_name: str, X_train, y_train, X_test, y_test,
                   cfg: dict):
    """Optuna objective for one model; hyperparameters sampled per model."""
    seed = cfg["optuna"]["seed"]

    def objective(trial) -> float:
        params = {}
        if model_name in ("lgbm", "xgb"):
            params["random_state"] = seed
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.05, 0.3, log=True)
            if model_name == "lgbm":
                params["num_leaves"] = trial.suggest_int("num_leaves", 8, 63)
            else:
                params["reg_lambda"] = trial.suggest_float(
                    "reg_lambda", 1e-3, 10.0, log=True)
        model = get_model(REGISTRY_KEY[model_name], **params)
        model.fit(X_train, y_train)
        return float(mean_absolute_error(y_test, model.predict(X_test)))
    return objective


def log_endpoints(model_name: str, cfg: dict, db_url: str) -> None:
    """Log the actual endpoints in use (visibility requirement)."""
    host = socket.gethostname()
    db = cfg["databases"]["optuna"]
    print(f"[worker] model={model_name} hostname={host} "
          f"ip={socket.gethostbyname(host)} "
          f"db={db['host']}:{db['port']}/{db['name']} "
          f"tracking_uri={cfg['mlflow']['tracking_uri']}")
    print(f"[worker] db_url={db_url}")


def log_to_mlflow(model_name: str, best, cfg: dict, dataset_path: str,
                  n_trials: int) -> None:
    """Log the best trial, dataset lineage, and metadata (no model artifact)."""
    setup_mlflow(cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"optuna_{model_name}"):
        log_params(best.params)
        mlflow.log_metric("mae", float(best.value))
        log_dataset(cfg["dataset"]["name"], dataset_path, context="train")
        log_metadata({"n_trials": float(n_trials)})
        mlflow.set_tag("model", model_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ols", choices=sorted(REGISTRY_KEY))
    parser.add_argument("--config", default="/etc/broadway/config.yaml")
    parser.add_argument("--secret-dir", default="/etc/broadway/secret")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    secret = load_secret(args.secret_dir)
    db = cfg["databases"]["optuna"]
    db_url = compose_db_url(db["driver"], secret["DB_USER"], secret["DB_PASSWORD"],
                            db["host"], db["port"], db["name"])
    log_endpoints(args.model, cfg, db_url)

    if args.init_only:
        for name in cfg["optuna"]["studies"]:
            run_study_rdb(lambda trial: 0.0, name, db_url, n_trials=0,
                          direction="minimize",
                          random_state=cfg["optuna"]["seed"])
        print("[worker] init complete")
        return

    ds = cfg["dataset"]
    df = load_dataset(cfg)
    X = df[ds["features"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, df[ds["target"]], test_size=cfg["optuna"]["test_fraction"],
        random_state=cfg["optuna"]["seed"])
    study = run_study_rdb(
        make_objective(args.model, X_train, y_train, X_test, y_test, cfg),
        study_name=f"ratecode1_{args.model}", storage_url=db_url,
        n_trials=cfg["optuna"]["studies"][args.model]["n_trials"],
        direction="minimize", random_state=cfg["optuna"]["seed"])
    best = study.best_trial
    log_to_mlflow(args.model, best, cfg, ds["parquet"],
                  cfg["optuna"]["studies"][args.model]["n_trials"])
    print(f"[worker] DONE model={args.model} best_mae={best.value:.4f} "
          f"params={best.params}")


if __name__ == "__main__":
    main()
