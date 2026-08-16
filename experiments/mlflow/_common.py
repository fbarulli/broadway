"""Shared setup for the mlflow model-battle experiment.

Loads the SAME working dataset as the univariate experiment (step 11's
ratecode1_sample.parquet, with the same metered filters) so results are
comparable, then provides the seeded 1000-row sample, the 80/20 holdout,
the sklearn pipeline factory (categorical branch ready for future steps),
and the full metric suite. Config lives at module level — no hardcoded
values inside functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results" / "mlflow"   # experiments/results/mlflow
REPO = HERE.parents[1]
MLRUNS = REPO / "mlruns"

WORKING_PARQUET = (
    HERE.parents[0] / "results" / "univariate" / "fare_amount_trip_distance"
    / "ratecode1_sample.parquet"
)

MIN_FARE = 2.50
MAX_DURATION_MINUTES = 240
SAMPLE_SIZE = 1000
TEST_FRACTION = 0.2
SEED = 42

CONTINUOUS_FEATURES = ["trip_distance", "duration_minutes", "pickup_hour"]
CATEGORICAL_FEATURES = ["time_bucket"]  # existing categorical from step 31

# name -> (registry key, constructor params) — registry from broadway.training
MODELS = {
    "ols": ("linear", {}),
    "lgbm": ("lgbm", {"n_estimators": 100, "learning_rate": 0.1,
                      "max_depth": 5, "random_state": SEED, "verbosity": -1}),
    "xgb": ("xgb", {"n_estimators": 100, "learning_rate": 0.1,
                    "max_depth": 5, "random_state": SEED, "verbosity": 0,
                    "tree_method": "hist"}),
}
BONUS_MODELS = {
    "ridge": (Ridge, {"alpha": 1.0, "random_state": SEED}),
    "rf": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 5,
                                   "random_state": SEED}),
}


def time_bucket(hour: int) -> str:
    """NYC-style bucket (step 31): day 6-15, peak 16-19, overnight else."""
    if 6 <= hour < 16:
        return "day"
    if 16 <= hour < 20:
        return "peak"
    return "overnight"


def load_metered() -> pd.DataFrame:
    """Ratecode1 metered trips — same filters as the univariate experiment."""
    df = pd.read_parquet(WORKING_PARQUET)
    df = df[df["fare_amount"] > MIN_FARE]
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    df["duration_minutes"] = df["trip_duration"] / 60
    df = df[(df["trip_duration"] > 0) & (df["duration_minutes"] < MAX_DURATION_MINUTES)]
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    return df


def load_sample() -> pd.DataFrame:
    """Seeded 1000-row sample of the metered data (battle scope)."""
    return load_metered().sample(n=SAMPLE_SIZE, random_state=SEED)


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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Full regression metric suite (holdout)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "explained_var": float(explained_variance_score(y_true, y_pred)),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   threshold: float) -> dict:
    """ROC / PR AUC on a binarized target (y >= threshold) — 'ROC variations'."""
    y_bin = (y_true >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_bin, y_pred)),
        "pr_auc": float(average_precision_score(y_bin, y_pred)),
    }


def binary_threshold(y_train: np.ndarray) -> float:
    """Binarization threshold from TRAIN only (no leakage into the metric)."""
    return float(np.median(y_train))
