"""Shared setup for the mlflow model-battle experiment.

Loads the SAME working dataset as the univariate experiment (same metered
filters, owned by `project.working`) so results are
comparable, then provides the seeded 1000-row sample, the 80/20 holdout, the
sklearn pipeline factory (categorical branch ready for future steps), and the
full metric suite. Run knobs (sample size, split, seed, features) come from
`project/config/experiments/mlflow.yaml`; model recipes stay here because they
reference the platform model registry / sklearn classes.
"""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    SplitConfig,
    TaskType,
)
from broadway.training.mlflow_utils import log_model
from broadway.training.models.registry import display_name, model_keys
from broadway.training.trainer import build_model_pipeline
from broadway.utils import require_keys
from project.paths import load_project_paths
from project.working import PICKUP_DATETIME_COL, load_metered, time_bucket

HERE = Path(__file__).resolve().parent
PATHS = load_project_paths()
RESULTS = PATHS.results / HERE.name
REPO = PATHS.root
MLRUNS = REPO / "mlruns"

# Config path is env-overridable (the k8s worker image sets
# BROADWAY_MLFLOW_CONFIG to its mounted location; local default = repo path).
CONFIG_PATH = Path(
    os.environ.get("BROADWAY_MLFLOW_CONFIG", PATHS.experiment_configs / "mlflow.yaml")
)

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
require_keys(_cfg, ["sample_size", "test_fraction", "seed",
                    "continuous_features", "categorical_features"], "mlflow.yaml")

SAMPLE_SIZE = int(_cfg["sample_size"])
TEST_FRACTION = float(_cfg["test_fraction"])
SEED = int(_cfg["seed"])
CONTINUOUS_FEATURES = list(_cfg["continuous_features"])
CATEGORICAL_FEATURES = list(_cfg["categorical_features"])

# Display alias -> registry key, DERIVED from the registry (single source of
# truth: MODEL_META's display names). Exists only so the battle keeps its
# historical display names on the CLI and in labels (01/03); it carries NO
# params — comparison fits use get_model(key), which applies the registry's
# default params.
MODEL_KEYS = {display_name(key): key for key in model_keys()}
def _pca_pipeline(**params: float) -> Pipeline:
    """Battle-only PCA -> LinearRegression pipeline (dimensionality-reduction surface)."""
    return Pipeline([
        ("pca", PCA(n_components=int(params["n_components"]),
                    random_state=params.get("random_state"))),
        ("lr", LinearRegression()),
    ])


# BONUS_MODELS are intentionally OUTSIDE the platform model registry: they demo
# sklearn classes (Ridge / RandomForestRegressor) the registry does not own.
# PCA/K-Means are battle-only surfaces (dimensionality-reduction / clustering);
# they are NOT registry models and NOT optuna-tunable — that is the intended
# boundary. KNN, by contrast, is registry-backed and tunable via the unified
# optuna API.
BONUS_MODELS = {
    "ridge": (Ridge, {"alpha": 1.0, "random_state": SEED}),
    "rf": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 5,
                                   "random_state": SEED}),
    "pca": (_pca_pipeline, {"n_components": 5, "random_state": SEED}),
    "kmeans": (KMeans, {"n_clusters": 8, "random_state": SEED}),
}


def load_metered_with_features() -> pd.DataFrame:
    """Metered rows + pickup_hour + time_bucket columns (battle scope)."""
    df = load_metered()
    df["pickup_hour"] = df[PICKUP_DATETIME_COL].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    return df


def load_sample() -> pd.DataFrame:
    """Seeded 1000-row sample of the metered data (battle scope)."""
    return load_metered_with_features().sample(n=SAMPLE_SIZE, random_state=SEED)


def split_data(df: pd.DataFrame) -> tuple:
    """Seeded 80/20 holdout split (X, y) -> (X_train, X_test, y_train, y_test)."""
    X = df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
    return train_test_split(X, df["fare_amount"], test_size=TEST_FRACTION,
                            random_state=SEED)


def make_pipeline(model: object) -> Pipeline:
    """ColumnTransformer (one-hot for existing categoricals) + model.

    The pipeline skeleton is the extension point: future preprocessing
    steps (scalers, feature selectors) slot in before the model.
    """
    pre = ColumnTransformer([
        ("num", "passthrough", CONTINUOUS_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def binary_threshold(y_train: np.ndarray) -> float:
    """Binarization threshold from TRAIN only (no leakage into the metric)."""
    return float(np.median(y_train))


def battle_pipeline_config() -> PipelineConfig:
    """Minimal experiment config for the unified HPO API: passthrough recipe.

    The battle feeds already-encoded frames into run_hpo/make_objective/
    log_best_artifacts, whose objectives compose ``Pipeline([pre, model])``
    from a PipelineConfig; with no ``preprocessing`` block the pre step is
    the identity passthrough, so behavior matches the historical bare-model
    fits. The config is scratch-local — the platform trainer derives its own
    recipe from real experiment YAML.
    """
    environment = EnvironmentConfig(
        log_level="INFO",
        data_dir="data",
        raw_subdir="raw",
        processed_subdir="processed",
        mlflow_tracking_uri=str(MLRUNS),
        database_user="user",
        database_password="pass",
        database_name="db",
        database_host="localhost",
        database_port=5432,
        sample_size_ci=1000,
        sample_size_stats=10000,
        api_replicas_min=1,
        api_replicas_max=3,
        api_hpa_cpu_threshold=80,
        monitoring_schedule="0 * * * *",
    )
    dataset = DatasetContract(
        name="battle",
        path="battle.parquet",
        target="fare_amount",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "fare_amount": ColumnSchema(
                dtype="float64", null_count=0, role=ColumnRole.TARGET
            )
        },
        lookup_tables={},
    )
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        features=FeatureConfig(include=[], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=TEST_FRACTION),
        random_state=SEED,
        target_metric="mae",
    )
    return PipelineConfig(dataset=dataset, environment=environment, experiment=experiment)


def log_best_artifacts(
    cfg: PipelineConfig,
    model_type: str,
    params: dict[str, float | int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Log best-run artifacts to the active mlflow run.

    Refits the composed Pipeline with the best params on train, logs it via
    the sklearn flavor (a Pipeline always has one — no flavor dispatch) with
    an explicit signature and cloudpickle serialization, writes the val
    predictions CSV, and plots the feature importances of tree models.
    """
    pipeline = build_model_pipeline(cfg, model_type, params)
    pipeline.fit(X_train, y_train)
    log_model(
        pipeline,
        "model",
        signature=infer_signature(X_train, y_train),
    )
    with tempfile.TemporaryDirectory() as tmp:
        preds = pipeline.predict(X_val)
        csv_path = Path(tmp) / "predictions.csv"
        pd.DataFrame({"actual": y_val.to_numpy(), "predicted": preds}).to_csv(
            csv_path, index=False
        )
        mlflow.log_artifact(str(csv_path))
        model = pipeline.named_steps["model"]
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
