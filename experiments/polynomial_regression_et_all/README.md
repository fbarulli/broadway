# Polynomial regression experiments

Exploring whether polynomial degree beats a linear fit for
`fare_amount ~ trip_distance` on the metered working dataset.

All analysis policy lives in `config.yaml` (no hardcoded values, no env vars).

Step scripts follow the `NN_name.py` convention used by the
`experiments/univariate/fare_amount_trip_distance` series.

This series lives at the top level of `experiments/`, alongside the
`univariate/` and `multivariate/` series.
