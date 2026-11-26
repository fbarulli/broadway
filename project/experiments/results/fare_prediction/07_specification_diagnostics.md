# Specification diagnostics (prepared train)

Ramsey RESET · Pearson-vs-Spearman discrepancies · MI-vs-R² audit.

- Rows: 790,463 (prepared train, NaN-dropped); target: fare_amount.
- RESET baseline: fare_amount ~ pickup_hour + route_id_encoded + pu_rush_interaction (plain OLS).
- RESET power=2 (yhat²): F=149,867.3, p=0.000e+00 → specification error (p<0.05).
- RESET power=3 (yhat²+yhat³): F=77,641.6, p=0.000e+00 → specification error (p<0.05).

## RESET variants (non-linear monotonic trap — duration curvature)

trip_duration_minutes is monotonic but curved; log or quadratic terms should collapse the specification error.

| feature            |     r2 |     reset_f | reset_p |
|--------------------|--------|-------------|---------|
| linear (safe)      | 0.6943 | 149867.2541 |  0.0000 |
| quadratic route_id | 0.7430 |   3771.8650 |  0.0000 |

## Evidence table

| feature                     | pearson | spearman |  delta |     r2 |     mi | mi_r2_ratio |
|-----------------------------|---------|----------|--------|--------|--------|-------------|
| pickup_hour                 |  0.0001 |   0.0105 | 0.0104 | 0.0000 | 0.0122 |      1.2160 |
| pickup_day_of_week          | -0.0193 |  -0.0184 | 0.0009 | 0.0004 | 0.0020 |      0.1967 |
| pickup_month                |  0.0084 |   0.0165 | 0.0082 | 0.0001 | 0.0007 |      0.0720 |
| is_weekend                  | -0.0181 |  -0.0266 | 0.0086 | 0.0003 | 0.0006 |      0.0610 |
| is_rush_hour                | -0.0245 |  -0.0173 | 0.0072 | 0.0006 | 0.0026 |      0.2599 |
| is_night                    |  0.0085 |   0.0101 | 0.0016 | 0.0001 | 0.0021 |      0.2137 |
| hour_sin                    |  0.0052 |   0.0032 | 0.0020 | 0.0000 | 0.0064 |      0.6370 |
| hour_cos                    | -0.0011 |  -0.0058 | 0.0047 | 0.0000 | 0.0056 |      0.5639 |
| dayofweek_sin               |  0.0144 |   0.0258 | 0.0114 | 0.0002 | 0.0034 |      0.3351 |
| dayofweek_cos               |  0.0180 |  -0.0008 | 0.0188 | 0.0003 | 0.0014 |      0.1425 |
| route_id_encoded            |  0.8330 |   0.8670 | 0.0341 | 0.6938 | 1.0017 |      1.4437 |
| dropoff_location_id_encoded |  0.5885 |   0.3566 | 0.2319 | 0.3463 | 0.2393 |      0.6911 |
| pu_rush_interaction         |  0.0368 |   0.0288 | 0.0080 | 0.0014 | 0.0823 |      8.2264 |
| pu_night_interaction        |  0.0705 |   0.0420 | 0.0285 | 0.0050 | 0.0682 |      6.8175 |
| pu_weekend_interaction      |  0.0364 |   0.0061 | 0.0303 | 0.0013 | 0.0613 |      6.1297 |

## Flagged: top-3 |Pearson − Spearman| (threshold Δ ≥ 0.05)

Features over threshold: dropoff_location_id_encoded.

| feature                     | pearson | spearman |  delta |
|-----------------------------|---------|----------|--------|
| dropoff_location_id_encoded |  0.5885 |   0.3566 | 0.2319 |
| route_id_encoded            |  0.8330 |   0.8670 | 0.0341 |
| pu_weekend_interaction      |  0.0364 |   0.0061 | 0.0303 |

## Flagged: top-3 MI/R² ratio (nonlinearity candidates)

| feature                |     mi |     r2 | mi_r2_ratio |
|------------------------|--------|--------|-------------|
| pu_rush_interaction    | 0.0823 | 0.0014 |      8.2264 |
| pu_night_interaction   | 0.0682 | 0.0050 |      6.8175 |
| pu_weekend_interaction | 0.0613 | 0.0013 |      6.1297 |
