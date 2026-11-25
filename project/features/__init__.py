"""Dataset-specific column plan for the taxi project.

The plan is the single keep-set for the model surface: source columns,
deterministic derived features, and the read-side encoder outputs, in the
producer's write order. ``project.ml_pipeline`` produces the derived/encoded
columns and projects through this plan; the platform experiment config
references the same callable as ``feature_plan``.
"""

from __future__ import annotations

import yaml

from broadway.config.loader import config_path
from broadway.config.schema import DatasetContract
from broadway.features.plan import ColumnOp, ColumnPlan

ROUTE_KEYS: tuple[str, str] = ("pickup_location_id", "dropoff_location_id")


def load_contract() -> DatasetContract:
    """The taxi dataset contract this plan resolves against (overlay-aware)."""
    data = yaml.safe_load(config_path("dataset/taxi.yaml").read_text(encoding="utf-8"))
    return DatasetContract.model_validate(data)


def _source(name: str, dtype: str) -> ColumnOp:
    return ColumnOp(op="source", name=name, dtype=dtype)


def _derive(name: str, func: str, source: str, dtype: str) -> ColumnOp:
    return ColumnOp(op="derive", name=name, dtype=dtype, sources=(source,), func=func)


def _encode(
    name: str, sources: tuple[str, ...], encoder: str, dtype: str, **params: object
) -> ColumnOp:
    return ColumnOp(
        op="encode",
        name=name,
        dtype=dtype,
        sources=sources,
        encoder=encoder,
        params=dict(params),
    )


def taxi(dataset: DatasetContract) -> ColumnPlan:
    """The taxi feature surface, in ``project.ml_pipeline`` write order."""
    datetime_column = dataset.datetime_column or "pickup_datetime"
    columns = (
        _derive("pickup_hour", "datetime_hour", datetime_column, "int32"),
        _derive("pickup_day_of_week", "datetime_dayofweek", datetime_column, "int32"),
        _derive("pickup_month", "datetime_month", datetime_column, "int32"),
        _source("passenger_count", "int64"),
        _source("trip_distance", "float64"),
        _source("pickup_location_id", "int32"),
        _source("dropoff_location_id", "int32"),
        _derive("is_weekend", "is_weekend", datetime_column, "int8"),
        _derive("rush_hour", "in_hour_window", datetime_column, "int8"),
        _derive("is_night", "is_night", datetime_column, "int8"),
        _derive("log_distance", "log_distance", "trip_distance", "float64"),
        _derive("same_borough", "same_group", "pickup_location_id", "int8"),
        _encode("route_avg_duration", ROUTE_KEYS, "target", "float64", smoothing=20),
        _encode(
            "route_frequency",
            ROUTE_KEYS,
            "frequency",
            "int32",
            normalize=False,
        ),
    )
    return ColumnPlan(
        target=dataset.target,
        datetime_column=dataset.datetime_column,
        columns=columns,
        builder_params={"group_col": "Borough", "lookup_col": "Borough_lookup"},
    )


CONTRACT = load_contract()
PLAN = taxi(CONTRACT)
