# Stationarity, autocorrelation, partial autocorrelation, cross-correlation

- hourly ADF: stat -5.527, p 1.82e-06 (stationary @5%)
- daily ADF: stat -2.077, p 0.254 (NON-stationary @5%)
- top hourly ACF lags (24+): 168h (0.89), 24h (0.88), 167h (0.85)
- top daily ACF lags (1+): 7d (0.67), 14d (0.56), 6d (0.50)
- hourly PACF outside the 95% band: 1h (+0.94), 2h (-0.70), 4h (-0.35)
- daily PACF outside the 95% band: 1d (+0.46), 6d (+0.42), 7d (+0.39)
- CCF demand vs mean trip duration: peak at lag -4h (+0.566); positive lag = demand leads duration, 95% band ±0.042

Artifacts: 03_stationarity_acf_adf.csv, 03_stationarity_acf_acf_hourly.csv, 03_stationarity_acf_acf_daily.csv, 03_stationarity_acf_pacf_hourly.csv, 03_stationarity_acf_pacf_daily.csv, 03_stationarity_acf_ccf.csv, 03_stationarity_acf.png, 03_stationarity_acf_ccf.png (this directory).
