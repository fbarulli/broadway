"""
Canonical feature definitions.

This file is the single source of truth for every feature used by
training and inference.
"""

from datetime import datetime

import pandera as pa
from pandera.typing import Series

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

ENGINEERED_FEATURES = [
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "is_weekend",
    "rush_hour",
    "is_night",
    "log_distance",
    "same_borough",
    "route_avg_duration",
    "route_frequency",
]


class EngineeredFeaturesSchema(pa.DataFrameModel):
    """Structure of the DataFrame returned by ``FeaturePipeline.transform()``.

    Exactly the ``ENGINEERED_FEATURES`` columns. Columns the pipeline computes
    directly (time/boolean/frequency/target-encoding) are checked strictly; the
    columns that merely pass through the input DataFrame inherit the input's
    width, so they are coerced to the data layer's downcast widths (int32 /
    float32). Structure only, no range checks.
    """

    pickup_hour: Series[pa.Int32]
    pickup_day_of_week: Series[pa.Int32]
    pickup_month: Series[pa.Int32]
    passenger_count: Series[pa.Float32] = pa.Field(coerce=True)
    trip_distance: Series[pa.Float32] = pa.Field(coerce=True)
    pickup_location_id: Series[pa.Int32] = pa.Field(coerce=True)
    dropoff_location_id: Series[pa.Int32] = pa.Field(coerce=True)
    is_weekend: Series[pa.Int8]
    rush_hour: Series[pa.Int8]
    is_night: Series[pa.Int8]
    log_distance: Series[pa.Float32] = pa.Field(coerce=True)
    same_borough: Series[pa.Int8]
    route_avg_duration: Series[pa.Float64]
    route_frequency: Series[pa.Int32]


TARGET = "trip_duration_minutes"

ROUTE_KEYS = [
    "pickup_location_id",
    "dropoff_location_id",
]
