from datetime import datetime

"""
Canonical feature definitions.

This file is the single source of truth for every feature used by
training and inference.
"""

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

TARGET = "trip_duration_minutes"

ROUTE_KEYS = [
    "pickup_location_id",
    "dropoff_location_id",
]
