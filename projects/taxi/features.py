"""Dataset-specific feature registry for the taxi project.

Single source of truth for the engineered feature set. The generic core
(``broadway.features``) derives names, dtypes, and the Pandera schema from
this registry — there is no parallel hand-maintained ``ENGINEERED_FEATURES``
list anywhere else.
"""

from __future__ import annotations

from broadway.features.schema import FeatureSpec, build_engineered_schema

FEATURE_SPECS: dict[str, FeatureSpec] = {
    "pickup_hour": FeatureSpec("pickup_hour", "int32"),
    "pickup_day_of_week": FeatureSpec("pickup_day_of_week", "int32"),
    "pickup_month": FeatureSpec("pickup_month", "int32"),
    "passenger_count": FeatureSpec("passenger_count", "float32", coerce=True),
    "trip_distance": FeatureSpec("trip_distance", "float32", coerce=True),
    "pickup_location_id": FeatureSpec("pickup_location_id", "int32", coerce=True),
    "dropoff_location_id": FeatureSpec("dropoff_location_id", "int32", coerce=True),
    "is_weekend": FeatureSpec("is_weekend", "int8"),
    "rush_hour": FeatureSpec("rush_hour", "int8"),
    "is_night": FeatureSpec("is_night", "int8"),
    "log_distance": FeatureSpec("log_distance", "float32", coerce=True),
    "same_borough": FeatureSpec("same_borough", "int8"),
    "route_avg_duration": FeatureSpec("route_avg_duration", "float64"),
    "route_frequency": FeatureSpec("route_frequency", "int32"),
}

ENGINEERED_FEATURES: tuple[str, ...] = tuple(FEATURE_SPECS)

ENGINEERED_FEATURE_TYPES: dict[str, str] = {
    name: spec.dtype for name, spec in FEATURE_SPECS.items()
}

ENGINEERED_SCHEMA = build_engineered_schema(FEATURE_SPECS)
