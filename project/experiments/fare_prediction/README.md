# Fare prediction experiments

Exploring baseline models for predicting `fare_amount` on the metered
working dataset.








Target: fare_amount
Allowed features: information known before the trip starts
Evaluation: MAE and RMSE on future held-out trips
Split: train on earlier dates, validate/test on later dates












Sample policy lives in `project/config/sample/fare_prediction_1m.yaml`
(seed/size/columns/filters/schema), consumed by name via `broadway.samples`;
feature and leakage policy (`SAFE_FEATURES`, `TEMPORAL_FEATURES`) is declared
once in `_common.py`.

Series map (each script owns one question; results land in
`project/experiments/results/fare_prediction/`):

| Step | Owns |
|---|---|
| `01_generate_sample.py` | build the immutable `fare_prediction_1m` sample artifact (no analysis) |
| `02_filtered_profile.py` | filtered-sample profile |
| `03_short_trips.py` | short-trip slice profile |
| `04_percentile_extremes_audit.py` | percentile extremes audit |
| `05_temporal_features.py` | temporal feature profile |
| `06_prepare_model_data.py` | train/val split for modeling |
| `07_specification_diagnostics.py` | specification diagnostics |
| `08_lightgbm_baseline.py` | the LightGBM baseline model |

Step scripts follow the `NN_name.py` convention used by the
`project/experiments/univariate/fare_amount_trip_distance` series.

This series lives under `project/experiments/`, alongside the
`univariate/` and `multivariate/` series.
