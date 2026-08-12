# src/logistics_ml/mlflow_utils.py
import os
import tempfile
import traceback

import joblib
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from logistics_ml.config.mlflow import mlflow as mlflow_config


def setup_mlflow(experiment_name):
    # Pull URI from config (reads from env var MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_registry_uri(mlflow_config.tracking_uri)

    experiment = mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()

    print(f"MLflow tracking to: {mlflow_config.tracking_uri}")
    print(f"MLflow experiment: {experiment.experiment_id}")
    print(f"Run ID: {run.info.run_id}")
    return run.info.run_id


def log_metrics(model_name, rows, training_time, metrics):
    mlflow.log_param("model", model_name)
    mlflow.log_param("rows", rows)
    mlflow.log_metric("training_time", training_time)
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_model(model, model_name, input_example=None):
    run = mlflow.active_run()

    wrapper_path = os.path.join(
        os.path.dirname(__file__), "models", "pyfunc_wrapper.py"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = os.path.join(tmp_dir, "model_bundle.joblib")
        joblib.dump(
            {
                "feature_pipeline": model.feature_pipeline,
                "model": model.model,
            },
            bundle_path,
        )

        try:
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapper_path,
                artifacts={"model_bundle": bundle_path},
                registered_model_name=model_name,
                input_example=input_example,
            )
        except Exception:
            traceback.print_exc()
            raise

    client = MlflowClient()

    # Verify model artifacts
    artifacts = client.list_artifacts(run.info.run_id, path="model")
    if artifacts:
        print("✓ Model artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.path}")
    else:
        print("⚠ No model artifacts found.")

    # Verify registered model
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    print(
        f"✓ Registered model '{latest.name}' "
        f"version {latest.version} ({latest.status})"
    )

    # Verify model can be loaded
    mlflow.pyfunc.load_model(model_info.model_uri)
    print("✓ Model successfully reloaded")

    return model_info.model_uri


def register_candidate(
    model, model_name, rows, training_time, metrics, input_example=None,
):
    log_metrics(model_name=model_name, rows=rows,
                training_time=training_time, metrics=metrics)
    model_uri = log_model(model, model_name, input_example=input_example)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    version = int(max(versions, key=lambda v: int(v.version)).version)

    return model_uri, version


def promote_candidate(model_name, alias, version):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name, alias=alias, version=version,
    )
    print(f"✓ Model version {version} promoted to alias '{alias}'")
