# Forecast model comparison — hourly demand

- train: 1848 complete hours, 2024-01-01 00:00:00 .. 2024-03-17 23:00:00
- holdout: 336 hours, 2024-03-18 00:00:00 .. 2024-03-31 23:00:00 (last 14 days)
- preprocessing: partial terminal hour dropped; the DST spring-forward hole (2024-03-10 02:00) is linearly interpolated for contiguous lags — train only.
- baseline convention: persistence/seasonal-naive are one-step lags of the observed series (t-1, t-24); the learned models forecast all 336 hours in one shot.
- repeatability: DeepAR and WaveNet are stochastic and retrained on seeds (0, 1, 2); std is across seeds. Every other model is a deterministic single run.
- intervals: SARIMAX (`get_forecast(...).conf_int`), AutoARIMA (statsforecast native level) and quantile LightGBM (2.5/50/97.5%) report a 95% holdout interval; coverage/width come from the platform calibration metrics.
- verdict: lgbm_quantile wins on mean MAE: +325.7 trips versus persistence.

| model | repeats | MAE | RMSE | MAPE % |
|---|---|---|---|---|
| lgbm_quantile | deterministic (single) | 264.1 | 369.8 | 8.0 |
| persistence | deterministic (single) | 589.8 | 761.8 | 28.9 |
| seasonal_naive_24h | deterministic (single) | 626.1 | 921.8 | 33.1 |
| sarima | deterministic (single) | 778.7 | 977.1 | 60.1 |
| auto_arima | deterministic (single) | 801.5 | 1009.9 | 44.0 |
| train_mean | deterministic (single) | 1988.6 | 2256.6 | 204.5 |

| interval model | nominal | holdout coverage | mean width (trips) |
|---|---|---|---|
| sarima | 0.95 | 0.961 | 4070.8 |
| auto_arima | 0.95 | 1.000 | 8094.8 |
| lgbm_quantile | 0.95 | 0.970 | 1965.9 |

- context: 02 found the hour-of-day/weekday shape and 03 the 24h/168h ACF dominance for a series stationary at 5% hourly — hence the 24h naive lag and daily+weekly seasonality in the learned models (`02_seasonality.md`, `03_stationarity_acf.md`).
- TimesFM: not run in this environment (see the skipped line).
- GluonTS: `gluonts_mqcnn` is unavailable in GluonTS >= 0.16 (MQCNN removed with the MXNet API); `gluonts_wavenet` is the torch CNN used as the second model.
- plot: stochastic models are drawn as their seed-mean path.

- skipped: `prophet` skipped — step 04 needs prophet; install the forecast extra:
  bash scripts/uv.sh sync --extra forecast; `timesfm` skipped — step 04 needs timesfm[torch]; install the forecast extra:
  bash scripts/uv.sh sync --extra forecast; `gluonts` skipped — step 04 needs gluonts[torch]; install the forecast extra:
  bash scripts/uv.sh sync --extra forecast

Artifacts: 04_forecast_models_metrics.csv, 04_forecast_models_aggregate.csv, 04_forecast_models_intervals.csv, 04_forecast_models.md, 04_forecast_models.png (this directory).
