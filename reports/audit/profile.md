# Dataset Profile

## Answer

The dataset has 8545833 rows and 7 columns.

## Key evidence

### Variables

| Variable | Type | Missing | Unique | Min | Max |
| --- | --- | --- | --- | --- | --- |
| dropoff_location_id | int32 | 0 | 261 | 1 | 265 |
| fare_amount | float64 | 0 | 3277 | -999.0 | 999.0 |
| passenger_count | float64 | 0 | 6 | 1.0 | 6.0 |
| pickup_datetime | datetime64[ns] | 0 | 4584079 | 2024-01-01T00:00:00 | 2024-04-01T00:34:55 |
| pickup_location_id | int32 | 0 | 257 | 1 | 265 |
| trip_distance | float64 | 0 | 4658 | 0.01 | 49.99 |
| trip_duration_minutes | float64 | 0 | 8721 | 1.0 | 179.71666666666667 |

### Potentially important observations

none

## Profile evidence

Traces are per-feature z-score.
Sample size: n = 10,000

![Per-feature Q-Q plots — figure 1 of 1](../figures/numeric_qq_1.png)

In this figure: fare_amount, trip_distance, trip_duration_minutes. Chunk 1 of 1; the trailing `_1` in the filename is the chunk number.

How to read (Q-Q): points should follow the fitted reference line; S-curves indicate tail behavior; curvature indicates skew. Shaded bands mark the middle 50% (centre) and the ±1.96σ tails; the red dashed horizontal line is the zero-mass shelf (a flat clump of dots = a spike of exact zeros).

![Per-feature Q-Q (raw vs log) — figure 1 of 1](../figures/numeric_qq_log_1.png)

In this figure: trip_distance, trip_duration_minutes. Chunk 1 of 1; the trailing `_1` in the filename is the chunk number.

How to read (raw vs log): the skewed, strictly-positive features re-plotted after a log transform; compare raw (left) to log (right) — if the right tail straightens toward the reference line, logging is a viable remediation.

Histograms are in raw units.

![Per-feature distributions — figure 1 of 1](../figures/numeric_dist_1.png)

In this figure: fare_amount, passenger_count, trip_distance, trip_duration_minutes. Chunk 1 of 1; the trailing `_1` in the filename is the chunk number.

How to read (distribution): actual spread and skew in original units; look for heavy tails, multimodality, and gaps.

### Distribution diagnostics

| Variable | n | mean | std | skew | excess_kurtosis | zero_rate | p99/median | max/median | log_skew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fare_amount | 10000 | 18.1 | 17.5 | 2.41 | 14.5 | 0.000 | 5.48 | 20.7 | - |
| passenger_count | 10000 | 1.35 | 0.832 | 3.17 | 11.2 | 0.000 | 5 | 6 | 2.05 |
| trip_distance | 10000 | 3.31 | 4.35 | 2.82 | 9.04 | 0.000 | 11.8 | 26.8 | 0.528 |
| trip_duration_minutes | 10000 | 15.5 | 12.6 | 2.51 | 10.6 | 0.000 | 5.31 | 13.6 | -0.0499 |

![Per-feature distribution diagnostics — figure 1 of 1](../figures/numeric_diagnostics.png)

In this figure: fare_amount, passenger_count, trip_distance, trip_duration_minutes. Chunk 1 of 1; the trailing `_1` in the filename is the chunk number.

How to read (diagnostics): colors are per-column z-scores; cell text is the raw value.

### Decision flags

- fare_amount: skew 2.41 exceeds 1.0
- fare_amount: kurtosis 14.54 exceeds 3.0
- passenger_count: skew 3.17 exceeds 1.0
- passenger_count: kurtosis 11.17 exceeds 3.0
- trip_distance: skew 2.82 exceeds 1.0
- trip_distance: kurtosis 9.04 exceeds 3.0
- trip_duration_minutes: skew 2.51 exceeds 1.0
- trip_duration_minutes: kurtosis 10.63 exceeds 3.0


Notes:
- dropoff_location_id: declared id
- pickup_location_id: declared id
- passenger_count: discrete (6 unique values) (excluded from Q-Q, kept as a bar chart in the distribution grid)
- not profiled (non-numeric): pickup_datetime

## Why this may matter

High-cardinality/identifier-like columns can inflate group counts or leak identity if used as grouping features.

## What Broadway did not decide

Broadway did not exclude or transform any column based on this profile.

## Technical details

artifacts/discover/profile.json
