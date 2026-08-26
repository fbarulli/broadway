"""Shared setup for the mlflow demo battle (main branch).

Mirror of the taxi branch's ``experiments/mlflow/_common.py`` contract —
same import surface (``project.working`` bindings, config-driven knobs,
``MODEL_KEYS`` derived from the registry) so the shared worker image and
its CI boot checks resolve identically on main. Backed by the synthetic
demo dataset; no taxi content.
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


def load_metered_with_features() -> pd.DataFrame:
    """Metered rows + hour + time_bucket columns (battle scope)."""
    df = load_metered()
    # SSOT: the literal "pickup_datetime" here mirrors working.yaml's
    # columns.pickup_datetime binding (project.working.PICKUP_DATETIME_COL) —
    # the binding, not this literal, is the single source of truth for the
    # column name; they coincide on main by construction.
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
