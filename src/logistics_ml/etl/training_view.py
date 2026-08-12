# src/logistics_ml/etl/training_view.py
from sqlalchemy import text

from logistics_ml.db import engine
from logistics_ml.config.data import data


SQL = f"""
DROP VIEW IF EXISTS training_data;
CREATE VIEW training_data AS
SELECT * FROM (
    SELECT
        pickup_datetime,
        dropoff_datetime,
        EXTRACT(HOUR FROM pickup_datetime) AS pickup_hour,
        EXTRACT(DOW FROM pickup_datetime) AS pickup_day_of_week,
        EXTRACT(MONTH FROM pickup_datetime) AS pickup_month,
        passenger_count,
        trip_distance,
        pickup_location_id,
        dropoff_location_id,
        fare_amount,
        tip_amount,
        total_amount,
        EXTRACT(EPOCH FROM (
            dropoff_datetime - pickup_datetime
        )) / 60.0 AS trip_duration_minutes
    FROM taxi_trips
) sub
WHERE trip_duration_minutes >= {data.min_trip_duration_minutes}
  AND trip_duration_minutes <= {data.max_trip_duration_minutes};
"""


def create_training_view():
    with engine.begin() as conn:
        conn.execute(text(SQL))

    print("training_data view created.")
