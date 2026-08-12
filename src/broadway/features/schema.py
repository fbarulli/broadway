"""Feature definitions and schema generation.

``FeatureSpec`` is the generic feature contract; ``build_engineered_schema``
generates a Pandera schema from a registry of specs. Dataset-specific feature
registries (e.g. the taxi features) live in ``project/features.py`` and derive
their name/dtype/schema views from a single ``FEATURE_SPECS`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandera as pa

from broadway.contracts.pandera import pandera_dtype

RAW_FEATURES = [
    "pickup_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
]

RAW_FEATURE_TYPES = {
    "pickup_datetime": datetime,
    "passenger_count": float,
    "trip_distance": float,
    "pickup_location_id": int,
    "dropoff_location_id": int,
}

STREAM_FEATURE_TYPES = {
    "pickup_location_id": int,
    "dropoff_location_id": int,
    "trip_distance": float,
    "passenger_count": float,
    "pickup_hour": int,
    "pickup_day_of_week": int,
    "pickup_month": int,
}

TARGET = "trip_duration_minutes"

ROUTE_KEYS = [
    "pickup_location_id",
    "dropoff_location_id",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Structure-only contract for a single engineered feature."""

    name: str
    dtype: str
    nullable: bool = False
    coerce: bool = False


def build_engineered_schema(specs: dict[str, FeatureSpec]) -> pa.DataFrameSchema:
    """Build a structure-only Pandera schema from a ``FeatureSpec`` registry."""
    return pa.DataFrameSchema(
        {
            spec.name: pa.Column(
                pandera_dtype(spec.dtype),
                nullable=spec.nullable,
                coerce=spec.coerce,
            )
            for spec in specs.values()
        }
    )
