# Polynomial regression experiments

Exploring whether polynomial degree beats a linear fit for
`fare_amount ~ trip_distance` on the metered working dataset.

There is no experiment-local config file: the working-dataset binding (parquet
path, filters, time buckets) is owned by `project.working` from
`configs/experiments/working.yaml`, re-exported through `_common.py`.

Step scripts follow the `NN_name.py` convention used by the
`experiments/univariate/fare_amount_trip_distance` series.

This series lives at the top level of `experiments/`, alongside the
`univariate/` and `multivariate/` series.
