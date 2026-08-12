"""Pandera DataFrame contracts for the NYC taxi project.

Structure-only schemas (columns, dtypes, nullability). Range/value checks
live in ``broadway/etl/process.py`` and ``broadway/contracts/checks.py``.

Column names mirror the single-source constants in ``projects.taxi.data``.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series


class TaxiRawSchema(pa.DataFrameModel):
    """Structure of the DataFrame returned by ``data.load_stratified_sample()``
    and ``data.load_time_slice()``.

    Dtypes reflect the data layer's downcast (int32/float32) and the zone
    lookup join (``LocationID`` stays int64, ``pickup_borough`` is nullable).
    """

    pickup_datetime: Series[pa.DateTime]
    passenger_count: Series[pa.Float32]
    trip_distance: Series[pa.Float32]
    pickup_location_id: Series[pa.Int32]
    dropoff_location_id: Series[pa.Int32]
    trip_duration_minutes: Series[pa.Float32]
    LocationID: Series[pa.Int64]
    pickup_borough: Series[pa.String] = pa.Field(nullable=True)
