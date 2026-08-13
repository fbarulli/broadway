# dataflow

Architecture map for the taxi stats learning project. LLM-friendly: read
top-to-bottom, use the tables to locate code.

## Lifecycle

One coherent platform flow (pipeline CLI), from dataset contract to champion
model:

```
DatasetContract → FeatureSpec → TrainingConfig → Optuna → TrainingResult
  → MLflow model/artifacts → EvaluationResult → promotion decision
  → champion model → prediction
```

## Artifacts

Typed execution outputs live under `artifacts/<step>/` (training/, evaluation/,
stats/, causal/), human-facing HTML reports live under `artifacts/reports/`,
and processed data stays under `data/processed/`.

`causal` is a separate analysis mode, not part of this flow and not part of
`full`. `configs/step/full.yaml` = discover, etl, contracts, eda, features,
stats, train, evaluate. Run causal explicitly:

```
ds-pipeline causal --dataset <d> --experiment <e>
```

## Directory tree

```
broadway/
  src/broadway/
    config/schema.py        # Pydantic models (DatasetContract, StatsStep, TrainStep, FeaturesStep, ...)
    discover/               # read CSV/parquet → infer contract + observed profile
      module.py             # run(): writes configs/dataset/<name>.yaml + artifacts/discover/profile.json
      profile.py            # DatasetProfile / ColumnProfile (observed facts; identifier_score is descriptive only)
    contracts/              # contract-generated schema + role selectors
      pandera.py            # build_raw_schema(contract) -> pa.DataFrameSchema (generated)
      selectors.py          # feature/datetime/target column selectors over DatasetContract
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
    causal/                 # experiment design + causal analysis (statsmodels/scipy)
      contracts.py          # ExperimentDesign, ExperimentResult (Pydantic) + save/load
      design.py             # design_experiment, minimum_detectable_effect (TTestIndPower)
      assignment.py         # assign_randomly, assign_stratified
      analysis.py           # analyze_two_groups (Welch's t-test, Cohen's d, 95% CI)
      multiple.py           # correct_pvalues (bonferroni, fdr_bh)
      sequential.py, hte.py # out-of-scope docstring stubs
      module.py             # pipeline step: reads cfg.causal, persists ExperimentDesign to artifacts/causal/
    features/               # generic feature machinery + the config-driven pipeline step
      schema.py             # FeatureSpec, build_engineered_schema
      ml_encodings.py, frequency.py   # generic target/frequency encodings
      pipeline.py, builders.py, encodings.py, module.py, contracts.py  # ds-pipeline features step
    evaluate/               # model evaluation + promotion decision
      contracts.py          # EvaluationResult (Pydantic model)
      metrics.py            # compute_metrics (mae/rmse/r2)
      comparison.py         # compare_models (candidate vs champion)
      validation.py         # cross_validate, residual_summary
      promotion.py          # should_promote
      module.py             # pipeline step: load model → evaluate → promotion
    training/               # model training + HPO + MLflow tracking
      contracts.py          # TrainingResult (Pydantic model)
      trainer.py            # train(model_type, X, y, **params) -> (model, elapsed)
      module.py             # pipeline step: load data → train → pickle to disk
      optuna.py             # run_study(objective, n_trials, ...) -> best_params
      mlflow_utils.py       # setup_mlflow, log_metrics, log_model
      models/               # model factories + ABC
        base.py             # BaseModel ABC (fit/predict/feature_importance/get_params/set_params)
        linear.py           # LinearRegression factory
        random_forest.py    # RandomForestRegressor factory
        xgboost.py          # XGBRegressor factory
        lightgbm.py         # LGBMRegressor factory
        registry.py         # get_model(name, **params)
        pyfunc_wrapper.py   # ModelPyFunc (MLflow PyFunc over a pickled model)
  project/
    features.py             # FEATURE_SPECS registry → ENGINEERED_FEATURES/types/schema
    ml_pipeline.py          # FeaturePipeline (taxi orchestration)
    basic.py, boroughs.py   # taxi datetime features + zone join
    data.py                 # loaders, constants, mode system, streaming cache
    STATS.md                # script index (what each numbered script does)
    scripts/                # numbered experiment scripts (01..12)
  configs/step/
    stats.yaml              # stats SSOT
    train.yaml              # model hyperparams SSOT
    features.yaml           # feature-engineer params SSOT
  configs/analysis/<name>.yaml  # authored analytical intent (AnalysisContract)
  tests/                    # test_base.py, test_anova.py, ... (pytest)
  results/                  # mode-keyed caches + reports (gitignored artifacts)
```

## Module → function → file

| Call site (script) | Function | File |
|---|---|---|
| `data.load_stratified_sample()` | `load_stratified_sample` | `project/data.py` |
| `data.load_time_slice()` | `load_time_slice` | `project/data.py` |
| `data.load_borough_durations()` | `load_borough_durations` | `project/data.py` |
| `data.generate_sample_cache()` | `generate_sample_cache` (streaming) | `project/data.py` |
| `data.inspect_schema()` | `inspect_schema` | `project/data.py` |
| `data.write_quality_report()` | `write_quality_report` | `project/data.py` |
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
generate_sample_cache()   ── merge zone lookup (pickup_borough)
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
| `data_path`, `lookup_path` (from `path` / `lookup_tables`) | `configs/dataset/taxi.yaml` → `DatasetContract` | `data.DATA_PATH`, `data.LOOKUP_PATH` (`project/data.py`) |
| `n_estimators`, `learning_rate`, `num_leaves`, ... | `configs/step/train.yaml` → `TrainStep` | `data.N_ESTIMATORS`, ... |
| rush-hour/night/passenger params | `configs/step/features.yaml` → `FeaturesStep` | `data.FEATURE_*` |
| column names | module constants in `data.py` | scripts |

Analysis intent is authored separately via `configs/analysis/<name>.yaml` → `AnalysisContract` (mode, goal, row_definition, decision_moment, available_info, leakage_notes, success_criterion), wired through the `--analysis <name>` CLI flag.

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
| AnalysisContract | Pydantic | `broadway/analysis/contracts.py` |
| AnalysisPlan | Pydantic | `broadway/stats/plan.py` |
| ExperimentDesign | Pydantic | `broadway/causal/contracts.py` |
| ExperimentResult | Pydantic | `broadway/causal/contracts.py` |
| EvaluationResult | Pydantic | `broadway/evaluate/contracts.py` |
| TrainingResult | Pydantic | `broadway/training/contracts.py` |
| DatasetProfile / ColumnProfile | Pydantic | `broadway/discover/profile.py` |
| Raw DataFrame | Pandera | `broadway/contracts/pandera.py::build_raw_schema(contract)` (generated) |
| Engineered features | Pandera | `project/features.py` (`FEATURE_SPECS`) → `broadway/features/schema.py::build_engineered_schema` |
| Python interfaces | type hints | throughout |

- The raw schema is generated at runtime from `DatasetContract.columns` — one `pa.Column` per contract entry (the raw 6 columns, not join-derived `pickup_borough`/`LocationID`). Dtypes are checked strictly (`coerce=False`); `null_count` is observed, not an invariant, so nullability is left at Pandera's default.
- Role-based column selection is `broadway/contracts/selectors.py` (`feature_columns`, `datetime_columns`, `target_columns`) — pure functions over the contract, no hardcoded names.
- Engineered features are defined ONCE in `project/features.py::FEATURE_SPECS`; `ENGINEERED_FEATURES`, `ENGINEERED_FEATURE_TYPES`, and `ENGINEERED_SCHEMA` are all derived from that registry (no parallel hand-maintained list).
- Enforcement points: `read_training_data()` validates the raw frame via `build_raw_schema`; `FeaturePipeline.transform()` validates against `ENGINEERED_SCHEMA`.
- `DatasetContract` is the accepted schema (authored/authoritative); `DatasetProfile` / `ColumnProfile` describe observed facts computed at discover time. `identifier_score` is purely descriptive — discover only logs a recommendation, it never mutates roles or the contract.

## Where to make changes

| Goal | Change |
|---|---|
| New config knob | `configs/step/stats.yaml` + matching field in `StatsStep` (`schema.py`) + constant in `data.py` |
| New DataFrame contract | add the column to `configs/dataset/taxi.yaml` — `build_raw_schema` regenerates the raw schema; call `Schema.validate(df)` at the stage boundary |
| New loader | add function in `project/data.py`; reuse `read_training_data` / `_join_boroughs` |
| New statistical test | add function in `src/broadway/stats/` (pandas/numpy only) + document in `API.md` |
| New experiment script | add `project/scripts/NN_*.py`; import from `project.data` and `broadway.stats` |
| Change sample behavior | edit `generate_sample_cache` / `_params_hash` in `data.py`; bump stale `params_hash` by regenerating |
