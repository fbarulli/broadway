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

import numpy as np
import pandas as pd
import yaml

from project.data import LOOKUP_PATH, ZONE_BOROUGH_COL, ZONE_ID_COL

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results" / "multivariate"

UNIVARIATE = HERE.parents[0] / "univariate" / "fare_amount_trip_distance"

CONFIG_KEYS = ["target", "value_counts_head", "sample_role", "categorical",
               "borough", "sample", "borough_dummies", "labels"]


def require_keys(config: dict, keys: list[str], context: str) -> None:
    """Fail loudly when config is missing keys (no silent defaults)."""
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"{context}: config missing required key(s): {missing}")


def require_finite(frame: pd.DataFrame, context: str) -> None:
    """Fail loudly on NaN/Inf — a silent fit on dirty input is worse than an error."""
    if frame.isna().any().any():
        raise ValueError(f"{context}: contains NaN — aborting instead of "
                         "fitting on misaligned/dirty input")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"{context}: contains Inf — aborting instead of "
                         "fitting on misaligned/dirty input")


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
        config = yaml.safe_load(fh)
    require_keys(config, CONFIG_KEYS, "config.yaml")
    return config


def load_zones() -> pd.DataFrame:
    """LocationID -> Borough lookup (path/columns owned by project.data)."""
    return pd.read_csv(LOOKUP_PATH, usecols=[ZONE_ID_COL, ZONE_BOROUGH_COL])


def add_boroughs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join pickup + dropoff boroughs; fail loudly on broken joins."""
    zones = load_zones()
    out = df
    for key in ("pickup", "dropoff"):
        spec = cfg["borough"][key]
        if spec["join_on"] not in out.columns:
            raise ValueError(f"borough join: column '{spec['join_on']}' not in data")
        before = len(out)
        out = out.merge(zones, left_on=spec["join_on"], right_on=ZONE_ID_COL,
                        how="left").rename(
                            columns={ZONE_BOROUGH_COL: spec["column"]})
        if len(out) != before:
            raise ValueError(
                f"borough join on '{spec['join_on']}' changed row count "
                f"({before} -> {len(out)}) — duplicate LocationIDs in lookup?")
        unmapped = int(out[spec["column"]].isna().sum())
        if unmapped:
            print(f"borough join '{spec['column']}': {unmapped} rows unmapped "
                  f"(kept as NaN — callers must handle or drop them)")
    return out


def load_manhattan_sample(cfg: dict) -> pd.DataFrame:
    """Metered rows restricted to the config pickup borough (the 'manhattan_sample')."""
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    df = add_boroughs(df, cfg)
    pickup_col = cfg["borough"]["pickup"]["column"]
    keep = cfg["sample"]["pickup_borough"]
    total = len(df)
    sample = df[df[pickup_col] == keep]
    print(f"sample filter '{pickup_col}' == '{keep}': {len(sample)} of {total} "
          f"({len(sample) / total:.1%})")
    if sample.empty:
        raise ValueError(f"sample filter yielded 0 rows "
                         f"(pickup borough '{keep}' absent from data)")
    return sample


def build_borough_dummies(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One-hot dummies for the config borough column (config reference dropped).

    Wraps the Categorical in a Series so get_dummies keeps df's index — a
    bare Categorical would emit a RangeIndex and silently misalign a concat.
    Callers decide missing-value handling (drop or fill) before calling.
    """
    spec = cfg["borough_dummies"]
    col = spec["column"]
    categories = [spec["reference"]] + sorted(
        v for v in df[col].dropna().unique() if v != spec["reference"])
    coded = pd.Series(pd.Categorical(df[col], categories=categories),
                      index=df.index)
    return pd.get_dummies(coded, prefix="borough",
                          drop_first=True, dtype=float)
