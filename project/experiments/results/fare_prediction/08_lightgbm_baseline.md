# LightGBM baseline (pre-trip features, log-dollar target)

LightGBM regressor on strictly pre-trip features; target `fare_amount_log`; MAE/RMSE/R² computed on real dollars via `np.expm1`.

## Dollar-space metrics

| split | mae | rmse | r2 |
| --- | --- | --- | --- |
| val | 3.02 | 6.49 | 0.86 |
| test | 3.09 | 6.76 | 0.85 |

## Top-10 gain importance

| feature | gain (% of total) |
| --- | --- |
| route_id_encoded | 90.2% |
| dropoff_location_id | 3.0% |
| dropoff_location_id_encoded | 2.8% |
| pickup_location_id | 2.7% |
| hour_cos | 0.4% |
| pickup_hour | 0.3% |
| dayofweek_sin | 0.2% |
| pickup_day_of_week | 0.1% |
| hour_sin | 0.1% |
| dayofweek_cos | 0.1% |
