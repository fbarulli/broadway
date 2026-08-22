"""Feature derivation — compute derived columns from source data.

``BUILDERS`` is the shared registry of pure column transforms. Both the
feature-engineering pipeline (``features.derived[].func`` in experiment
configs) and the named-sample pipeline (``SampleSpec.derived[].formula`` in
sample configs) resolve their functions here. It is an *implementation*
registry only — which features exist, their names, and their dtypes are
declared in the dataset layer (``project/features.py`` for model-facing
features; sample configs declare their own derived columns).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from broadway.config.schema import DerivedFeature


def _same_group(df: pd.DataFrame, group_col: str, lookup_col: str) -> pd.Series:
    if group_col not in df.columns:
        raise ValueError(f"same_group requires column {group_col!r}.")
    if lookup_col not in df.columns:
        raise ValueError(f"same_group requires column {lookup_col!r}")
    pu = df[group_col]
    do = df[lookup_col]
    return (pu == do).astype(int)


def _in_hour_window(df: pd.DataFrame, source: str, hours: list[int]) -> pd.Series:
    return pd.to_datetime(df[source]).dt.hour.isin(hours).astype(int)


def _rate_per_hour(df: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    """distance / (duration_minutes / 60). Column names come from the
    dataset's role mapping (``columns``), falling back to generic names."""
    distance = columns.get("distance", "distance")
    duration = columns.get("duration_minutes", "duration_minutes")
    return df[distance] / (df[duration] / 60)


# Shared transform registry. Contract: callable(df, src=None, **kw); a
# multi-input transform reads its input columns by role from kw["columns"].
BUILDERS = {
    "datetime_hour": lambda df, src, **kw: pd.to_datetime(df[src]).dt.hour.astype(int),
    "datetime_dayofweek": lambda df, src, **kw: pd.to_datetime(df[src]).dt.dayofweek.astype(int),
    "datetime_month": lambda df, src, **kw: pd.to_datetime(df[src]).dt.month.astype(int),
    "is_weekend": lambda df, src, **kw: pd.to_datetime(df[src]).dt.dayofweek.isin([5, 6]).astype(int),
    "in_hour_window": lambda df, src, **kw: _in_hour_window(df, src, kw.get("hour_window", [7, 8, 9, 17, 18, 19])),
    "is_night": lambda df, src, **kw: pd.to_datetime(df[src]).dt.hour.isin([0, 1, 2, 3, 4, 5, 21, 22, 23]).astype(int),
    "log_distance": lambda df, src, **kw: np.log1p(df[src]),
    "same_group": lambda df, src, **kw: _same_group(df, kw.get("group_col", "group"), kw.get("lookup_col", "group_lookup")),
    "rate_per_hour": lambda df, src, **kw: _rate_per_hour(df, kw.get("columns", {})),
    # source_copy casts to the declared dtype: copy-source dtype varies with the
    # source, so _BUILDER_DTYPES is the SSOT ("derived dtype == declared dtype
    # regardless of source"); downstream numeric feature selection takes float64.
    "source_copy": lambda df, src, **kw: df[src].astype(_BUILDER_DTYPES["source_copy"]),
}

_BUILDER_DTYPES: dict[str, str] = {
    "datetime_hour": "int64",
    "datetime_dayofweek": "int64",
    "datetime_month": "int64",
    "is_weekend": "int64",
    "in_hour_window": "int64",
    "is_night": "int64",
    "same_group": "int64",
    "rate_per_hour": "float64",
    "log_distance": "float64",
    "source_copy": "float64",
}


def load_custom_builders(builder_module: str | None) -> dict:
    if not builder_module:
        return {}
    try:
        mod = importlib.import_module(builder_module)
    except ImportError as exc:
        raise ValueError(f"builder module '{builder_module}' could not be imported: {exc}") from exc
    builders = getattr(mod, "BUILDERS", None)
    if not isinstance(builders, dict):
        raise ValueError(f"builder module '{builder_module}' must define a BUILDERS dict of name -> callable")
    collisions = sorted(set(builders) & set(BUILDERS))
    if collisions:
        raise ValueError(
            f"builder module '{builder_module}' collides with generic builder name(s): {collisions}"
        )
    return builders


def builder_dtype(func: str) -> str:
    return _BUILDER_DTYPES.get(func, "float64")


def build_derived(df: pd.DataFrame, features: list[DerivedFeature], target: str,
                  extra_builders: dict | None = None,
                  group_col: str = "group",
                  lookup_col: str = "group_lookup",
                  hour_window: list[int] | None = None) -> pd.DataFrame:
    _window = hour_window if hour_window is not None else [7, 8, 9, 17, 18, 19]
    builder_kwargs = {"group_col": group_col, "lookup_col": lookup_col, "hour_window": _window}
    registry = dict(BUILDERS)
    if extra_builders:
        registry.update(extra_builders)
    result = df.copy()
    for feat in features:
        if feat.func not in registry:
            raise ValueError(f"unknown builder function '{feat.func}' for derived feature '{feat.name}'")
        if feat.source not in df.columns:
            raise ValueError(f"derived feature '{feat.name}' references missing source column '{feat.source}'")
        result[feat.name] = registry[feat.func](result, feat.source, **builder_kwargs)
    return result
