# Forecast model comparison — hourly demand

- train: 1848 complete hours, 2024-01-01 00:00:00 .. 2024-03-17 23:00:00
- holdout: 336 hours, 2024-03-18 00:00:00 .. 2024-03-31 23:00:00 (last 14 days)
- preprocessing: partial terminal hour dropped; the DST spring-forward hole (2024-03-10 02:00) is linearly interpolated for contiguous lags — train only.
- baseline convention: EVERY model is a multi-step forecast of the 336-hour horizon. persistence tiles the last train value; seasonal-naive-24h tiles the last 24 train hours. Neither reads the holdout, so the ranking is like-for-like.
- repeatability: DeepAR and WaveNet are stochastic and retrained on seeds (0, 1, 2); std is across seeds. Every other model is a deterministic single run.
- intervals: SARIMAX (`get_forecast(...).conf_int`), AutoARIMA (statsforecast native level) and quantile LightGBM (2.5/50/97.5%) report a 95% holdout interval; coverage/width come from the platform calibration metrics.
- verdict: champion lgbm_quantile on MAE = 264.1 trips [230.5, 298.5] 95%; +2266.2 trips versus persistence, runner-up sarima.

| model | repeats | MAE | RMSE | MAPE % |
|---|---|---|---|---|
| lgbm_quantile | deterministic (single) | 264.1 | 369.8 | 8.0 |
| sarima | deterministic (single) | 778.7 | 977.1 | 60.1 |
| auto_arima | deterministic (single) | 801.5 | 1009.9 | 44.0 |
| seasonal_naive_24h | deterministic (single) | 1206.9 | 1624.0 | 90.8 |
| train_mean | deterministic (single) | 1988.6 | 2256.6 | 204.4 |
| persistence | deterministic (single) | 2530.4 | 2846.9 | 133.3 |

| interval model | nominal | holdout coverage | mean width (trips) |
|---|---|---|---|
| sarima | 0.95 | 0.961 | 4070.8 |
| auto_arima | 0.95 | 1.000 | 8094.8 |
| lgbm_quantile | 0.95 | 0.970 | 1965.9 |

- context: 02 found the hour-of-day/weekday shape and 03 the 24h/168h ACF dominance for a series stationary at 5% hourly — hence the 24h naive lag and daily+weekly seasonality in the learned models (`02_seasonality.md`, `03_stationarity_acf.md`).
- plot: stochastic models are drawn as their seed-mean path.

- skipped: `prophet` skipped — step 04 needs prophet; install the forecast extra:
  bash scripts/uv.sh sync --extra forecast; `timesfm` skipped — step 04 needs timesfm[torch]; install the forecast extra:
  bash scripts/uv.sh sync --extra forecast; `gluonts` skipped — step 04 needs gluonts[torch]; install the forecast extra:
  bash scripts/uv.sh sync --extra forecast

Artifacts: 04_forecast_models_metrics.csv, 04_forecast_models_aggregate.csv, 04_forecast_models_intervals.csv, 04_forecast_models.md, 04_forecast_models.png, one 04_forecast_models_<metric>.png leaderboard per metric, 04_forecast_models_champion.json, 04_forecast_models_decision.json (this directory).
