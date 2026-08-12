# src/logistics_ml/training.py

import argparse
import json
import sys
import time
from types import SimpleNamespace

import mlflow
import mlflow.pyfunc

from logistics_ml.config.mlflow import mlflow as mlflow_config
from logistics_ml.data import load_training_data
from logistics_ml.evaluation import evaluate, should_promote
from logistics_ml.features import RAW_FEATURES, prepare_dataset
from logistics_ml.mlflow_utils import (
    promote_candidate,
    register_candidate,
    setup_mlflow,
)
from logistics_ml.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and register logistics ML model")
    parser.add_argument(
        "--model",
        choices=["linear", "rf", "xgb", "lgbm"],
        default="xgb",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help='JSON string of hyperparameters (e.g., \'{"n_estimators": 500}\')',
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="If set, evaluates against Champion and attempts promotion to MLflow Registry",
    )
    return parser.parse_args()


def train_model(model_name, params, X_train, y_train):
    print(f"Starting training for model: {model_name} with params: {params}")

    # Instantiate model with external hyperparameters
    model = get_model(model_name, **params)

    # Log hyperparameters to the active MLflow run for experiment tracking
    mlflow.log_params(model.get_params())

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    print(f"Training completed in {training_time:.2f}s")
    return model, training_time


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    metrics = evaluate(model, X_test, y_test)

    print("\n=== Evaluation Results ===")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²  : {metrics['r2']:.4f}")

    return metrics


def build_input_example(df):
    """Creates a schema-valid input example for MLflow logging."""
    input_example = df[RAW_FEATURES].head(5).copy()

    input_example["pickup_location_id"] = input_example["pickup_location_id"].astype(
        "float64")
    input_example["dropoff_location_id"] = input_example["dropoff_location_id"].astype(
        "float64")

    return input_example


def get_champion_metrics(champion_model_name, champion_alias, X_test, y_test):
    """Fetches the current champion model from MLflow registry and evaluates it."""
    try:
        champion_uri = f"models:/{champion_model_name}@{champion_alias}"
        print(
            f"Loading current champion from {champion_uri} for comparison...")

        champion = mlflow.pyfunc.load_model(champion_uri)

        print("Evaluating current champion...")
        return evaluate(champion, X_test, y_test)
    except Exception as e:
        print(
            f"No existing champion found or failed to load ({e}). Will promote unconditionally.")
        return None


def main():
    args = parse_args()

    # Parse JSON hyperparameters safely
    try:
        hyperparams = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(f"Error parsing --params JSON: {e}")
        sys.exit(1)

    print("=== Initializing Training Run ===")
    run_id = setup_mlflow(mlflow_config.experiment_name)

    try:
        print(f"Active MLflow Run ID: {run_id}")
        print(f"Promotion enabled: {args.promote}")

        # 1. Data Loading
        df = load_training_data()

        # 2. Feature Engineering & Splitting
        X_train, X_test, y_train, y_test, pipeline = prepare_dataset(df)

        # 3. Model Training
        model, training_time = train_model(
            args.model, hyperparams, X_train, y_train)

        # 4. Evaluation
        metrics = evaluate_model(model, X_test, y_test)

        # 5. Bundle model and feature pipeline for MLflow PyFunc
        wrapped_model = SimpleNamespace(
            feature_pipeline=pipeline,
            model=model,
        )

        # 6. Log MLflow metrics, artifacts, and register candidate
        print("Registering candidate model in MLflow Model Registry...")
        model_uri, version = register_candidate(
            model=wrapped_model,
            model_name=mlflow_config.registered_model_name,
            rows=len(df),
            training_time=training_time,
            metrics=metrics,
            input_example=build_input_example(df),
        )
        print(f"Candidate registered as Version {version}. URI: {model_uri}")

        # 7. Champion/Challenger Evaluation & Promotion (Only if --promote is passed)
        if args.promote:
            print("\n=== Champion vs. Challenger Comparison ===")
            champion_metrics = get_champion_metrics(
                mlflow_config.registered_model_name,
                mlflow_config.champion_alias,
                X_test,
                y_test,
            )

            should_promote_flag = False
            if champion_metrics is None:
                should_promote_flag = True
            else:
                print("Champion Metrics:")
                print(f"MAE : {champion_metrics['mae']:.4f}")
                print(f"RMSE: {champion_metrics['rmse']:.4f}")
                print(f"R²  : {champion_metrics['r2']:.4f}")

                # Use RMSE for promotion criteria
                should_promote_flag = should_promote(
                    candidate_metrics=metrics,
                    champion_metrics=champion_metrics,
                    metric="rmse",
                )

            if should_promote_flag:
                print(
                    f"\n🚀 Promoting Version {version} to '{mlflow_config.champion_alias}'.")
                promote_candidate(
                    model_name=mlflow_config.registered_model_name,
                    alias=mlflow_config.champion_alias,
                    version=version,
                )
                print("=== TRAINING & PROMOTION COMPLETE ===")
            else:
                print(
                    "\n❌ Candidate did not outperform Champion. Model remains registered but NOT promoted.")
                print("=== TRAINING COMPLETE (No Promotion) ===")
        else:
            print("\n=== TRAINING COMPLETE (HPO Trial Logged) ===")
            print(f"Candidate Version: {version}")

    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        # Exit with 1 so Kubernetes knows the Job failed
        sys.exit(1)


if __name__ == "__main__":
    main()
