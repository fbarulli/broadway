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

Step scripts follow the `NN_name.py` convention used by the
`project/experiments/univariate/fare_amount_trip_distance` series.

This series lives under `project/experiments/`, alongside the
`univariate/` and `multivariate/` series.
