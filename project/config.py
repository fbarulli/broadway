"""Dataset-specific project config — the taxi demo layer's ingest knobs.

Lives in ``project/`` (not ``src/broadway/``) because it is dataset-specific:
the generic platform never needs to know about borough lookups or rush-hour
windows. The generic core reads the shared steps from ``configs/step/*.yaml``;
this model parses ``project/config/project/taxi.yaml`` through project composition.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProjectConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    processed_file: str
    min_trip_distance: float
    max_trip_distance: float
    min_trip_duration_minutes: float
    max_trip_duration_minutes: float
    min_pickup_datetime: str
    min_passenger_count: int
    max_passenger_count: int
    rename_map: dict[str, str]
    borough_column: str
    borough_lookup_column: str
    lookup_path: str
    rush_hour_morning_start: int
    rush_hour_morning_end: int
    rush_hour_evening_start: int
    rush_hour_evening_end: int
    night_start: int
    night_end: int
