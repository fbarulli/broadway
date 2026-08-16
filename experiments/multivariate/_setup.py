"""Shared setup for the multivariate experiment (categorical breakdown).

The working dataset loader and its constants are OWNED by the univariate
experiment's `_common.py` (single source of truth) — loaded here under an
explicit module name (importlib) rather than duplicated or shadowed. This
module is deliberately NOT named `_common` because the univariate modules
import `_common` by name; a same-named module here would shadow them.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results" / "multivariate"

UNIVARIATE = HERE.parents[0] / "univariate" / "fare_amount_trip_distance"


def _load_univariate_module(module_name: str, filename: str):
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


def load_metered_categorical() -> pd.DataFrame:
    """Metered rows + derived categoricals (pickup_hour, time_bucket)."""
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    return df
