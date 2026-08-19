"""Shared setup for the mlflow model-battle experiment.

Loads the SAME working dataset as the univariate experiment (ratecode1_sample
with the same metered filters, owned by `project.working`) so results are
comparable, then provides the seeded 1000-row sample, the 80/20 holdout, the
sklearn pipeline factory (categorical branch ready for future steps), and the
full metric suite. Run knobs (sample size, split, seed, features) come from
`configs/experiments/mlflow.yaml`; model recipes stay here because they
reference the platform model registry / sklearn classes.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from broadway.training.models.registry import display_name, model_keys
from broadway.utils import require_keys
from project.working import load_metered, time_bucket

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results" / "mlflow"   # experiments/results/mlflow
# Repo root; in the k8s worker image the script lives at /app/_common.py
# (depth 1), so fall back to HERE when parents[1] does not exist.
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
MLRUNS = REPO / "mlruns"

# Config path is env-overridable (the k8s worker image sets
# BROADWAY_MLFLOW_CONFIG to its mounted location; local default = repo path).
CONFIG_PATH = Path(
    os.environ.get("BROADWAY_MLFLOW_CONFIG", REPO / "configs" / "experiments" / "mlflow.yaml")
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
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
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
