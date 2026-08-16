"""Shared setup for the multivariate experiment (categorical breakdown).

The working dataset loader and its constants are OWNED by the univariate
experiment's `_common.py` (single source of truth) — loaded here under an
explicit module name (importlib) rather than duplicated or shadowed. The
zone-lookup path/columns are owned by `project.data` and reused here. This
module is deliberately NOT named `_common` because the univariate modules
import `_common` by name; a same-named module here would shadow them.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import yaml

from project.data import LOOKUP_PATH, ZONE_BOROUGH_COL, ZONE_ID_COL

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results" / "multivariate"

UNIVARIATE = HERE.parents[0] / "univariate" / "fare_amount_trip_distance"


def _load_univariate_module(module_name: str, filename: str) -> types.ModuleType:
    """Load a univariate experiment module under a unique name (no shadowing)."""
    spec = importlib.util.spec_from_file_location(
        module_name, UNIVARIATE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_uni_common = _load_univariate_module("_uni_common", "_common.py")
sys.modules["_common"] = _uni_common  # univariate _ols_bp imports `_common` by name
_uni_ols_bp = _load_univariate_module("_uni_ols_bp", "_ols_bp.py")

load_metered = _uni_common.load_metered
WORKING_DATASET = _uni_common.WORKING_DATASET
time_bucket = _uni_ols_bp.time_bucket


def load_config() -> dict:
    """Analysis policy from config.yaml (never hardcoded in code)."""
    with open(HERE / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_zones() -> pd.DataFrame:
    """LocationID -> Borough lookup (path/columns owned by project.data)."""
    return pd.read_csv(LOOKUP_PATH, usecols=[ZONE_ID_COL, ZONE_BOROUGH_COL])


def add_borough(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join pickup borough onto the metered rows (join_on/column from config)."""
    spec = cfg["borough"]
    return df.merge(load_zones(), left_on=spec["join_on"],
                    right_on=ZONE_ID_COL, how="left").rename(
                        columns={ZONE_BOROUGH_COL: spec["column"]})


def load_metered_categorical(cfg: dict) -> pd.DataFrame:
    """Metered rows + derived categoricals (pickup_hour, time_bucket, borough)."""
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    return add_borough(df, cfg)
