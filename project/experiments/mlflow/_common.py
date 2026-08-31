"""Shared setup for the mlflow demo battle (main branch).

The project-owned worker binds its project config and sample evidence while
using the shared platform model registry.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from broadway.training.models.registry import display_name, model_keys
from broadway.utils import require_keys
from project.working import load_metered, time_bucket

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RESULTS = HERE.parents[0] / "results" / "mlflow"
MLRUNS = PROJECT_ROOT / "mlruns"

CONFIG_PATH = Path(
    os.environ.get("BROADWAY_MLFLOW_CONFIG", PROJECT_ROOT / "config" / "experiments" / "mlflow.yaml")
)

_cfg = yaml.safe_load(CONFIG_PATH.read_text())
require_keys(_cfg, ["sample_size", "test_fraction", "seed",
                    "continuous_features", "categorical_features"], "mlflow.yaml")

SAMPLE_SIZE = int(_cfg["sample_size"])
TEST_FRACTION = float(_cfg["test_fraction"])
SEED = int(_cfg["seed"])
CONTINUOUS_FEATURES = list(_cfg["continuous_features"])
CATEGORICAL_FEATURES = list(_cfg["categorical_features"])

MODEL_KEYS = {display_name(key): key for key in model_keys()}


def load_metered_with_features() -> pd.DataFrame:
    """Metered rows + hour + time_bucket columns (battle scope)."""
    df = load_metered()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["time_bucket"] = df["hour"].map(time_bucket)
    return df


def load_sample() -> pd.DataFrame:
    """Seeded sample of the metered data (battle scope)."""
    return load_metered_with_features().sample(n=SAMPLE_SIZE, random_state=SEED)


def split_data(df: pd.DataFrame) -> tuple:
    """Seeded holdout split (X, y) -> (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split

    X = df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES]
    return train_test_split(X, df["target"], test_size=TEST_FRACTION,
                            random_state=SEED)
