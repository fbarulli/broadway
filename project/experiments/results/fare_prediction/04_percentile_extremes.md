# Percentile extremes audit (bottom/top 1% of fares)

Bottom 1%: fares <= $4.40 — 12,946 rows (0 flagged)
Top 1%: fares >= $70.90 — 10,053 rows (4 flagged)

## Bottom 1% describe

| stat  | fare_amount | trip_distance | trip_duration_minutes |
|-------|-------------|---------------|-----------------------|
| count |       12946 |         12946 |                 12946 |
| mean  |        4.24 |          0.33 |                  2.09 |
| std   |        0.31 |          0.13 |                  0.96 |
| min   |        2.80 |          0.01 |                  1.00 |
| 1%    |        3.00 |          0.03 |                  1.03 |
| 25%   |        4.40 |          0.26 |                  1.63 |
| 50%   |        4.40 |          0.34 |                  2.02 |
| 75%   |        4.40 |          0.41 |                  2.38 |
| 99%   |        4.40 |          0.57 |                  4.56 |
| max   |        4.40 |          2.40 |                 27.85 |

## Top 1% describe

| stat  | fare_amount | trip_distance | trip_duration_minutes |
|-------|-------------|---------------|-----------------------|
| count |       10053 |         10053 |                 10053 |
| mean  |       96.43 |         20.58 |                 46.62 |
| std   |       31.38 |          6.27 |                 17.75 |
| min   |       70.90 |          0.10 |                  2.15 |
| 1%    |       70.90 |          5.94 |                 16.91 |
| 25%   |       78.30 |         17.20 |                 34.17 |
| 50%   |       87.40 |         19.30 |                 42.72 |
| 75%   |      100.00 |         23.14 |                 55.72 |
| 99%   |      228.40 |         42.56 |                101.67 |
| max   |      400.00 |         49.90 |                179.07 |

Flagged rows: bottom 0, top 4.

## Bottom flagged rows (head 10)

| # | fare_amount | trip_distance | trip_duration_minutes | speed_mph |
|---|-------------|---------------|-----------------------|-----------|

## Top flagged rows (head 10)

| # | fare_amount | trip_distance | trip_duration_minutes | speed_mph |
|---|-------------|---------------|-----------------------|-----------|
| 1 |      $71.00 |       0.19 mi |               4.2 min |   2.7 mph |
| 2 |      $71.06 |       1.08 mi |              17.9 min |   3.6 mph |
| 3 |      $71.00 |       0.10 mi |               2.1 min |   2.8 mph |
| 4 |      $70.90 |       2.80 mi |              99.0 min |   1.7 mph |
