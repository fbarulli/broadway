"""Feature derivation — compute derived columns from source data."""

from __future__ import annotations

import importlib

import pandas as pd

from broadway.config.schema import DerivedFeature


_BUILDERS = {
    "datetime_hour": lambda df, src, **kw: pd.to_datetime(df[src]).dt.hour.astype(int),
    "datetime_dayofweek": lambda df, src, **kw: pd.to_datetime(df[src]).dt.dayofweek.astype(int),
    "datetime_month": lambda df, src, **kw: pd.to_datetime(df[src]).dt.month.astype(int),
}

_BUILDER_DTYPES: dict[str, str] = {
    "datetime_hour": "int64",
    "datetime_dayofweek": "int64",
    "datetime_month": "int64",
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
    collisions = sorted(set(builders) & set(_BUILDERS))
    if collisions:
        raise ValueError(
            f"builder module '{builder_module}' collides with generic builder name(s): {collisions}"
        )
    return builders


def builder_dtype(func: str) -> str:
    return _BUILDER_DTYPES.get(func, "float64")


def build_derived(df: pd.DataFrame, features: list[DerivedFeature], target: str,
                  extra_builders: dict | None = None) -> pd.DataFrame:
    registry = dict(_BUILDERS)
    if extra_builders:
        registry.update(extra_builders)
    result = df.copy()
    for feat in features:
        if feat.func not in registry:
            raise ValueError(f"unknown builder function '{feat.func}' for derived feature '{feat.name}'")
        if feat.source not in df.columns:
            raise ValueError(f"derived feature '{feat.name}' references missing source column '{feat.source}'")
        result[feat.name] = registry[feat.func](result, feat.source)
    return result
