# dataflow

Architecture map for the taxi stats learning project. LLM-friendly: read
top-to-bottom, use the tables to locate code.

## Directory tree

```
broadway/
  src/broadway/
    config/schema.py        # Pydantic models (StatsStep, TrainStep, FeaturesStep, ...)
    stats/                  # pandas/numpy stats library (no Spark)
      base.py               # stratified_sample
      plan.py               # AnalysisPlan dataclass + save/load
      effect_size.py        # eta², omega², Cohen's d, Hedges' g, group_imbalance
      assumptions.py        # Levene, skew/kurtosis/Shapiro
      anova.py              # run_anova, run_welch, run_kruskal
      post_hoc.py           # games_howell
      regression.py         # fit_ols, fit_robust, bp_jb
      diagnostics.py        # bp_test, jb_test, durbin_watson, plot_residuals
      time_series.py        # durbin_watson_test, plot_acf
      baseline.py           # train_lgbm, evaluate
      module.py             # pipeline step run(cfg)
  projects/taxi/
    data.py                 # loaders, constants, mode system, streaming cache
    scripts/                # numbered experiment scripts (01..12)
  configs/step/
    stats.yaml              # stats SSOT
    train.yaml              # model hyperparams SSOT
    features.yaml           # feature-engineer params SSOT
  tests/                    # test_base.py, test_anova.py, ... (pytest)
  results/                  # mode-keyed caches + reports (gitignored artifacts)
```

## Module → function → file

| Call site (script) | Function | File |
|---|---|---|
| `data.load_stratified_sample()` | `load_stratified_sample` | `projects/taxi/data.py` |
| `data.load_time_slice()` | `load_time_slice` | `projects/taxi/data.py` |
| `data.load_borough_durations()` | `load_borough_durations` | `projects/taxi/data.py` |
| `data.generate_sample_cache()` | `generate_sample_cache` (streaming) | `projects/taxi/data.py` |
| `data.inspect_schema()` | `inspect_schema` | `projects/taxi/data.py` |
| `data.write_quality_report()` | `write_quality_report` | `projects/taxi/data.py` |
| `anova.run_anova(groups)` | `run_anova` | `src/broadway/stats/anova.py` |
| `anova.run_welch(groups)` | `run_welch` | `src/broadway/stats/anova.py` |
| `anova.run_kruskal(groups)` | `run_kruskal` | `src/broadway/stats/anova.py` |
| `assumptions.run_levene(groups)` | `run_levene` | `src/broadway/stats/assumptions.py` |
| `assumptions.check_normality(groups)` | `check_normality` | `src/broadway/stats/assumptions.py` |
| `post_hoc.games_howell(df, ...)` | `games_howell` | `src/broadway/stats/post_hoc.py` |
| `regression.fit_ols(df, formula)` | `fit_ols` | `src/broadway/stats/regression.py` |
| `regression.bp_jb(model)` | `bp_jb` | `src/broadway/stats/regression.py` |
| `diagnostics.durbin_watson(resid)` | `durbin_watson` | `src/broadway/stats/diagnostics.py` |
| `time_series.plot_acf(resid, ...)` | `plot_acf` | `src/broadway/stats/time_series.py` |
| `baseline.train_lgbm(X, y, ...)` | `train_lgbm` | `src/broadway/stats/baseline.py` |
| `baseline.evaluate(model, ...)` | `evaluate` | `src/broadway/stats/baseline.py` |

## Data flow

```
data/processed/training_data.parquet   (raw, 8.6M rows)
        │  pyarrow ParquetFile.iter_batches(batch_size=100_000)
        ▼
generate_sample_cache()   ── downcast int64→int32 / float64→float32
        │                     merge zone lookup (pickup_borough)
        │                     incremental per-borough stratified sample
        ├──▶ results/joined_sample_{MODE}.parquet   (≈ SAMPLE_SIZE rows)
        ├──▶ results/sample_meta_{MODE}.json        (params_hash)
        └──▶ results/quality_report.json            (exact group sizes/means)
        │
        ▼
scripts (01..12)
        │  load_stratified_sample()  → random stratified groups
        │  load_time_slice()         → contiguous, time-sorted slice (filter pushdown)
        ▼
results/*.json / *.png  (AnalysisPlan JSON, residual plots, ACF plot)
```

## Config SSOT

| Value | Owned by | Consumer |
|---|---|---|
| `sample_size_dev` / `sample_size_live` | `configs/step/stats.yaml` → `StatsStep` | `data.SAMPLE_SIZE` |
| `time_slice_start_{mode}` / `time_slice_end_{mode}` | `configs/step/stats.yaml` → `StatsStep` | `data.TIME_SLICE_START/END` |
| `time_split_cutoff` | `configs/step/stats.yaml` → `StatsStep` | `data.TIME_SPLIT_CUTOFF` |
| `min_rows_for_sampling`, `per_group_sample_fraction`, `group_values` | `configs/step/stats.yaml` | `data.MIN_ROWS_FOR_SAMPLING`, `data.BOROUGHS` |
| `data_path`, `lookup_path` | `configs/step/stats.yaml` | `data.DATA_PATH`, `data.LOOKUP_PATH` |
| `n_estimators`, `learning_rate`, `num_leaves`, ... | `configs/step/train.yaml` → `TrainStep` | `data.N_ESTIMATORS`, ... |
| rush-hour/night/passenger params | `configs/step/features.yaml` → `FeaturesStep` | `data.FEATURE_*` |
| column names | module constants in `data.py` | scripts |

## Mode system

| Env var | Default | `SAMPLE_SIZE` | time slice |
|---|---|---|---|
| `DATA_MODE=dev` | ✓ | `sample_size_dev` (2000) | `time_slice_start_dev` → `time_slice_end_dev` (1 day) |
| `DATA_MODE=live` | | `sample_size_live` (200000) | `time_slice_start_live` → `time_slice_end_live` (1 month) |

- Cache files are mode-keyed: `joined_sample_{MODE}.parquet`, `sample_meta_{MODE}.json`.
- `DATA_MODE` is read once at import; any value other than `dev`/`live` raises.

## Sampling strategies

| Strategy | Loader | Guarantees | Used by |
|---|---|---|---|
| Stratified random | `load_stratified_sample` | per-borough proportions preserved; deterministic (`RANDOM_STATE`) | 08, 09, 11, 07 (games-howell), 04–06 (ANOVA groups via `load_borough_durations`) |
| Contiguous time slice | `load_time_slice` | rows sorted by `pickup_datetime`, no random sampling (filter pushdown) | 10 (Durbin-Watson / ACF) |

## Where to make changes

| Goal | Change |
|---|---|
| New config knob | `configs/step/stats.yaml` + matching field in `StatsStep` (`schema.py`) + constant in `data.py` |
| New loader | add function in `projects/taxi/data.py`; reuse `read_training_data` / `_downcast` / `_join_boroughs` |
| New statistical test | add function in `src/broadway/stats/` (pandas/numpy only) + document in `API.md` |
| New experiment script | add `projects/taxi/scripts/NN_*.py`; import from `projects.taxi.data` and `broadway.stats` |
| Change sample behavior | edit `generate_sample_cache` / `_params_hash` in `data.py`; bump stale `params_hash` by regenerating |
