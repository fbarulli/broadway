"""Fare-series temporal recipe: the shared builders resolve in the registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED = {
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
}


def _common():
    """Load the experiment's shared module by path (it is not a package)."""
    path = _ROOT / "project" / "experiments" / "fare_prediction" / "_common.py"
    spec = importlib.util.spec_from_file_location("fare_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fare_common"] = module
    spec.loader.exec_module(module)
    return module


def test_temporal_recipe_builds_every_declared_feature() -> None:
    frame = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(["2024-01-01 08:30", "2024-01-07 21:15"]),
            "fare_amount": [12.5, 30.0],
        }
    )
    out = _common().build_temporal_features(frame)

    assert _EXPECTED <= set(out.columns)
    assert out["pickup_hour"].tolist() == [8, 21]
    assert out["pickup_day_of_week"].tolist() == [0, 6]
