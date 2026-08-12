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
      plan.py               # AnalysisPlan (Pydantic model) + save/load
      effect_size.py        # eta², omega², Cohen's d, Hedges' g, group_imbalance
      assumptions.py        # Levene, skew/kurtosis/Shapiro
      anova.py              # run_anova, run_welch, run_kruskal
      post_hoc.py           # games_howell
      regression.py         # fit_ols, fit_robust, bp_jb
      diagnostics.py        # bp_test, jb_test, durbin_watson, plot_residuals
      time_series.py        # durbin_watson_test, plot_acf
      baseline.py           # train_lgbm, evaluate
      module.py             # pipeline step: build groups → run_anova → save_plan
  projects/taxi/
    data.py                 # loaders, constants, mode system, streaming cache
    STATS.md                # script index (what each numbered script does)
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
| `data_path`, `lookup_path` (from `path` / `lookup_tables`) | `configs/dataset/taxi.yaml` → `DatasetContract` | `data.DATA_PATH`, `data.LOOKUP_PATH` (`projects/taxi/data.py`) |
| `n_estimators`, `learning_rate`, `num_leaves`, ... | `configs/step/train.yaml` → `TrainStep` | `data.N_ESTIMATORS`, ... |
| rush-hour/night/passenger params | `configs/step/features.yaml` → `FeaturesStep` | `data.FEATURE_*` |
| column names | module constants in `data.py` | scripts |

## Mode system

| Env var | Default | `SAMPLE_SIZE` | time slice |
|---|---|---|---|
| `DATA_MODE=dev` | ✓ | `sample_size_dev` (2000) | `time_slice_start_dev` → `time_slice_end_dev` (1 day) |
| `DATA_MODE=live` | | `sample_size_live` (200000) | `time_slice_start_live` → `time_slice_end_live` (1 month) |

- Cache files are mode-keyed: `joined_sample_{MODE}.parquet`, `sample_meta_{MODE}.json`.
- `mode` is a per-call parameter on the loaders (`load_stratified_sample(mode=None)`, `generate_sample_cache(mode=None)`, `load_time_slice(mode=None)`, `load_borough_durations(mode=None)`). `mode=None` falls back to `os.getenv("DATA_MODE", "dev")` via `_resolve_mode`; any value other than `dev`/`live` raises.
- The module constants `MODE`, `SAMPLE_SIZE`, `TIME_SLICE_START`, `TIME_SLICE_END`, `SAMPLE_CACHE`, `SAMPLE_META` are still resolved at import (from `_resolve_mode()`) as defaults, because scripts read `data.TIME_SLICE_START`/`TIME_SLICE_END` and `data.SAMPLE_SIZE`.

## Sampling strategies

| Strategy | Loader | Guarantees | Used by |
|---|---|---|---|
| Stratified random | `load_stratified_sample` | per-borough proportions preserved; deterministic (`RANDOM_STATE`) | 08, 09, 11, 07 (games-howell), 04–06 (ANOVA groups via `load_borough_durations`) |
| Contiguous time slice | `load_time_slice` | rows sorted by `pickup_datetime`, no random sampling (filter pushdown) | 10 (Durbin-Watson / ACF) |

## Contracts

| Contract | Tool | Where |
|---|---|---|
| Configuration | Pydantic | `broadway/config/schema.py` |
| AnalysisPlan | Pydantic | `broadway/stats/plan.py` |
| Taxi raw DataFrame | Pandera | `projects/taxi/schemas.py` |
| Engineered features | Pandera | `broadway/features/schema.py` |
| Python interfaces | type hints | throughout |

- Pandera `DataFrameModel`s are structure-only (columns, dtypes, nullability). Range/value checks stay in `etl/process.py` + `contracts/checks.py`.
- Enforcement points: `load_stratified_sample()` / `load_time_slice()` validate against `TaxiRawSchema`; `FeaturePipeline.transform()` validates against `EngineeredFeaturesSchema`.

## Where to make changes

| Goal | Change |
|---|---|
| New config knob | `configs/step/stats.yaml` + matching field in `StatsStep` (`schema.py`) + constant in `data.py` |
| New DataFrame contract | add a Pandera `DataFrameModel` (`projects/taxi/schemas.py` or `broadway/features/schema.py`) and call `Schema.validate(df)` at the stage boundary |
| New loader | add function in `projects/taxi/data.py`; reuse `read_training_data` / `_downcast` / `_join_boroughs` |
| New statistical test | add function in `src/broadway/stats/` (pandas/numpy only) + document in `API.md` |
| New experiment script | add `projects/taxi/scripts/NN_*.py`; import from `projects.taxi.data` and `broadway.stats` |
| Change sample behavior | edit `generate_sample_cache` / `_params_hash` in `data.py`; bump stale `params_hash` by regenerating |
