"""Demo working-data binding (main branch, synthetic demo).

Mirror of the taxi branch's ``project/working.py`` contract (``load_metered``
+ ``time_bucket`` + config-driven knobs) so the shared worker image layout —
and its CI boot checks — resolve identically on both branches. Backed by the
synthetic demo dataset; no taxi content.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from broadway.utils import require_keys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "working.yaml"
_cfg = yaml.safe_load(CONFIG.read_text())
require_keys(_cfg, ["parquet", "columns", "min_fare", "max_duration_minutes",
                    "time_buckets", "time_bucket_default"], "working.yaml")
require_keys(_cfg["columns"], ["fare", "pickup_datetime", "dropoff_datetime"],
             "working.yaml columns")

WORKING_DATASET = ROOT / _cfg["parquet"]
FARE_COL = _cfg["columns"]["fare"]
PICKUP_DATETIME_COL = _cfg["columns"]["pickup_datetime"]
DROPOFF_DATETIME_COL = _cfg["columns"]["dropoff_datetime"]
MIN_FARE = float(_cfg["min_fare"])
MAX_DURATION_MINUTES = float(_cfg["max_duration_minutes"])


def time_bucket(hour: int) -> str:
    """Surcharge-style bucket from config boundaries (else = default)."""
    for label, bounds in _cfg["time_buckets"].items():
        if bounds["start"] <= hour < bounds["end"]:
            return label
    return _cfg["time_bucket_default"]


def load_working() -> pd.DataFrame:
    """Working dataset with dataset-level filters applied."""
    df = pd.read_parquet(WORKING_DATASET)
    return df[df[FARE_COL] > MIN_FARE]


def load_metered() -> pd.DataFrame:
    """Working dataset + duration derived; duration filters applied."""
    df = load_working()
    df["trip_duration"] = (
        df[DROPOFF_DATETIME_COL] - df[PICKUP_DATETIME_COL]
    ).dt.total_seconds()
    df["duration_minutes"] = df["trip_duration"] / 60
    return df[(df["trip_duration"] > 0) & (df["duration_minutes"] < MAX_DURATION_MINUTES)]
