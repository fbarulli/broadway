"""Single project-level binding for the tutorial working dataset.

One owner for the working dataset every tutorial experiment operates on:
parquet path, column names, dataset-level filter knobs, the loaders, and the
NYC time-bucket mapping. Everything comes from
`project/config/experiments/working.yaml` — nothing hardcoded, nothing from env.

Consumed by `project/experiments/univariate/.../_common.py` (re-export),
`project/experiments/multivariate/_setup.py`, and `project/experiments/mlflow/_common.py`,
which is what kills the old importlib hack + the duplicated loaders.
"""

from __future__ import annotations

import pandas as pd
import yaml

from broadway.utils import require_keys
from project.paths import load_project_paths

PATHS = load_project_paths()
CONFIG = PATHS.experiment_configs / "working.yaml"
_cfg = yaml.safe_load(CONFIG.read_text())
require_keys(_cfg, ["parquet", "columns", "min_target_value", "max_duration_minutes",
                    "time_buckets", "time_bucket_default"], "working.yaml")
require_keys(_cfg["columns"], ["target", "pickup_datetime", "dropoff_datetime"],
             "working.yaml columns")

WORKING_DATASET = PATHS.root / _cfg["parquet"]
TARGET_COL = _cfg["columns"]["target"]
PICKUP_DATETIME_COL = _cfg["columns"]["pickup_datetime"]
DROPOFF_DATETIME_COL = _cfg["columns"]["dropoff_datetime"]
MIN_TARGET_VALUE = float(_cfg["min_target_value"])
MAX_DURATION_MINUTES = float(_cfg["max_duration_minutes"])


def time_bucket(hour: int) -> str:
    """NYC-style surcharge bucket from config boundaries (else = default)."""
    for label, bounds in _cfg["time_buckets"].items():
        if bounds["start"] <= hour < bounds["end"]:
            return label
    return _cfg["time_bucket_default"]


def load_working() -> pd.DataFrame:
    """Working dataset with dataset-level filters applied."""
    df = pd.read_parquet(WORKING_DATASET)
    return df[df[TARGET_COL] > MIN_TARGET_VALUE]


def load_metered() -> pd.DataFrame:
    """Working dataset + duration derived; duration filters applied."""
    df = load_working()
    df["trip_duration"] = (
        df[DROPOFF_DATETIME_COL] - df[PICKUP_DATETIME_COL]
    ).dt.total_seconds()
    df["duration_minutes"] = df["trip_duration"] / 60
    return df[(df["trip_duration"] > 0) & (df["duration_minutes"] < MAX_DURATION_MINUTES)]
