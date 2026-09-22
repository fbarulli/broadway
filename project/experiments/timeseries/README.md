# Timeseries experiments

Demand/volume dynamics on contiguous NYC taxi time — hourly/daily
aggregates, seasonality, gaps, and naive forecast baselines. This series
is TEMPORAL; cross-sectional fare modeling lives in `fare_prediction/`,
`multivariate/`, and `more_modeling/` (note: `more_modeling/22_demand_forecasting.py`
already predicts zone-hour volume with a RandomForest — start there before
duplicating it; this series owns the series-first analysis: stationarity,
autocorrelation, seasonality decomposition, and time-based validation).

## Dataset decision

| Candidate | Rows | Coverage | Verdict |
|---|---|---|---|
| `data/processed/taxi_canonical.parquet` | 8.5M | 2024-01-01 → 2024-04-01, cleaned, zone-enriched | **primary** — contiguous + clean |
| `data/raw/yellow_tripdata_2024-*.parquet` | ~3M/mo | Jan/Feb/Mar 2024, uncleaned | provenance checks only |
| `data/samples/fare_prediction_1m@v3.parquet` | 988k | random draw | rejected — random sampling breaks continuity |

## Series plan

1. `01_series_profile.py` — temporal coverage, frequency, gaps (read-only).
2. Hourly/daily demand aggregates + seasonality plots.
3. Stationarity (ADF) + autocorrelation (ACF) per aggregate.
4. Naive baselines (persistence, seasonal-naive) with time-based splits.
5. Model comparison only after 2-4 hold.

Step scripts follow the `NN_name.py` convention. Run from the repo root, e.g.
`bash scripts/uv.sh run python project/experiments/timeseries/01_series_profile.py`.
Results land in `project/experiments/results/timeseries/`. Artifact names begin
with their step stem so the experiment dashboard can group and preview them
(CSVs tracked, PNGs ignored).
