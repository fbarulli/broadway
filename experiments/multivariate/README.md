# Multivariate experiment — categorical breakdown, geography premiums, ML baselines

All analysis policy lives in `config.yaml` (no hardcoded values, no env vars).
The dataset loader is owned by the univariate experiment's `_common.py`
(single source of truth); the zone lookup is owned by `project.data`.

## Steps

| step | what | key outputs |
|---|---|---|
| `01_categorical_breakdown.py` | auto-detect categoricals; describe-style figures (boxplot + group sizes) + per-category stats | `01_categorical_breakdown_<col>.{png,csv}` |
| `02_borough_dummies_kurtosis.py` | borough dummies vs residual heavy tails (kurtosis/skew/JB/BP/R2) | `02_borough_dummies_kurtosis.{csv}` |
| `03_geography_premium.py` | per-borough dollar premium vs Manhattan (HC3, 95% CI) | `03_geography_premium.{csv,png}`, `..._diagnostics.csv` |
| `04_baseline_sample.py` | ML baseline samples A/B/C (target `total_amount`, chronological 80/20 split) | `<sample>_train/test.parquet`, `04_baseline_sample_split.csv` |
| `05_lightgbm_baselines.py` | LightGBM on each population; performance dashboard (MAE/RMSE/R2/tail MAE, residuals) | `05_lightgbm_baselines*.{csv,png}`, `baseline_lightgbm.json` |
| `06_model_verdict.py` | scorecard: role + verdict + metrics per model | `06_model_verdict.{csv,png}`, `model_verdict.json` |

Sample: `manhattan_sample` (Population A) — metered trips with Manhattan
pickups (93% of the working set). Population B stratifies by pickup borough;
C adds outer-borough error weighting.

## The answers (evidence-backed)

- **Heavy tails are not a geography story** — kurtosis 194.8 → 199.6 with
  dropoff borough dummies; premiums are real but small (Staten Island +$7.22,
  Queens +$0.87) and don't explain the tails (see `02`/`03`).
- **Model A (Revenue Engine):** MAE $1.20, tail $4.18, R2 0.963 on the future
  holdout — reliable for Manhattan-grid pricing/payouts/forecasting.
- **Model B (Risk Simulator):** MAE $2.84, tail $8.51 — outer-borough/airport
  trips are structurally harder; use for risk, not point estimates.
- **Model C (Cheat Code, failed):** weighting outer-borough errors barely
  moves the metrics — penalties don't fix the fare physics.

Metrics are evidence (`baseline_lightgbm.json`); roles/verdicts are analyst
interpretation (`config.yaml → model_verdicts`).

## Run order

```
uv run python experiments/multivariate/04_baseline_sample.py
uv run python experiments/multivariate/05_lightgbm_baselines.py
uv run python experiments/multivariate/06_model_verdict.py
```

`01`–`03` are independent; `04` → `05` → `06` chain on the samples.

## Fail-loud rules

Config keys are validated (`require_keys`), joins are integrity-checked,
exog matrices are checked for NaN/Inf before fitting, and missing split
parquets raise clear errors — never silent wrong results.
