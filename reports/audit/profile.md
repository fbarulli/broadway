# Dataset Profile

## Answer

The dataset has 8545833 rows and 8 columns.

## Key evidence

### Variables

| Variable | Type | Missing | Unique | Min | Max |
| --- | --- | --- | --- | --- | --- |
| pickup_datetime | datetime64[ns] | 0 | 4584079 | 2024-01-01T00:00:00 | 2024-04-01T00:34:55 |
| passenger_count | float64 | 0 | 6 | 1.0 | 6.0 |
| trip_distance | float64 | 0 | 4658 | 0.01 | 49.99 |
| pickup_location_id | int32 | 0 | 257 | 1 | 265 |
| dropoff_location_id | int32 | 0 | 261 | 1 | 265 |
| total_amount | float64 | 0 | 23975 | -1000.0 | 1021.99 |
| airport_fee | float64 | 0 | 3 | -1.75 | 1.75 |
| trip_duration_minutes | float64 | 0 | 8721 | 1.0 | 179.71666666666667 |

### Potentially important observations

none

## Profile evidence

![passenger_count, trip_distance, pickup_location_id, dropoff_location_id, total_amount, airport_fee, trip_duration_minutes](../figures/numeric_qq_1.png)

Traces are per-feature z-score.

How to read: traces hugging the diagonal are approximately normal; S-curves indicate tail behavior; curvature indicates skew.

## Why this may matter

High-cardinality/identifier-like columns can inflate group counts or leak identity if used as grouping features.

## What Broadway did not decide

Broadway did not exclude or transform any column based on this profile.

## Technical details

artifacts/discover/profile.json
