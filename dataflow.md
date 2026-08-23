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

Inference: new-path artifacts are sklearn Pipelines logged with an explicit
signature (`Pipeline([("pre", …), ("model", …)])` via `infer_signature(X, y)`,
cloudpickle); they load through MLflow's native pyfunc flavor
(`evaluate/module.py::_load_candidate/_load_champion`) and predict on RAW
input frames — the pre-preprocessing feature frame — with MLflow enforcing
the logged signature at predict time. Previously logged bare-model artifacts
remain loadable via `ModelPyFunc` (`training/models/pyfunc_wrapper.py`).

## Artifacts

Typed execution outputs live under `artifacts/<step>/` (training/, evaluation/,
stats/, causal/), human-facing reports (markdown + figures) live under
`reports/`, and processed data stays under `data/processed/`.

`artifacts/` holds machine-readable evidence/provenance; `reports/` holds
human-facing derived views (regenerable from records + configs).

`reports/` is the human-facing product surface, owned by
`src/broadway/reports/`. The surface is split by owner: `walkthrough` owns
`index.md` (progress dashboard) and `results/` (per-step pages + results
index); `audit` owns `audit/`; `lineage` owns `lineage/`;
`reports/timeline.md` is the analysis timeline. The legacy `report` command is
a thin wrapper that renders `reports/results/index.md` (and errors with "run
the walkthrough first" if no timeline state exists). Outputs are split by kind:
machine JSON → `artifacts/`, markdown → `reports/`, images →
`reports/figures/`.

Per-step figures are declared as `FigureRef` (a `path` relative to `reports/`
plus a one-line "How to read" caption) and embedded as `![caption](figures/...)`
in `timeline.md` and `![caption](../figures/...)` in `results/` pages (one link
depth deeper). The surface is guarded by `tests/test_surface_integrity.py`,
which verifies every relative markdown link in tracked `reports/` files resolves
on disk and that tracked `.html` / `.png` files stay under size caps (5 MB / 2
MB).

The `audit` command (`ds-pipeline audit --dataset <d> [--analysis <a>]`) is a
separate on-demand, question-oriented surface. It reads the persisted typed
evidence — `StructuralCleanResult` (`data/processed/<name>_clean.json`),
`JoinAuditReport` (`<name>_join_audit.json`), `LookupValueAuditReport`
(`<name>_lookup_value_audit.json`), and `DatasetProfile`
(`artifacts/discover/profile.json`) — and renders one page per question
(`profile.md`, `transform.md`, `join.md`, `lookup_values.md`) plus an
`index.md` (data used, dataset status, what changed, enrichment quality,
things to consider). It is pure rendering: it never re-runs ingest/etl/stats
and never reads parquet.

`causal` is a separate analysis mode, not part of this flow and not part of
`full`. `full` is a thin dispatcher: it reads `AnalysisContract.mode` and
resolves one of `configs/flow/{prediction,hypothesis,causal}.yaml`. Run causal
explicitly:

```
ds-pipeline causal --dataset <d> --experiment <e>
```

### Timeline / walkthrough

The "analysis timeline" is an analyst-led hypothesis walkthrough driven by two
commands. `ds-pipeline walkthrough --analysis <a> --dataset <d> [--sample <s>]
[--force]` advances an eight-step sequence — `describe_groups → normality →
variance → decide_omnibus → omnibus → decide_posthoc → posthoc → conclusion` —
authored in `configs/flow/hypothesis_walkthrough.yaml` (thresholds in
`configs/step/walkthrough.yaml`), stopping at each decision gate. It is
idempotent (existing steps are skipped on resume); `--force` recomputes steps
but never overwrites recorded decisions. `ds-pipeline decide --analysis <a>
--method <m> --reason "..." [--kind omnibus|posthoc]` records an
`AnalysisDecision` (omnibus: `welch`/`anova`/`kruskal`; post-hoc:
`games_howell`) that gates the walkthrough.

Step state is a plain-text vocabulary — `completed`, `completed with note`,
`awaiting decision`, `failed`, `warning` — persisted as `AnalysisStep` JSON
under `artifacts/timeline/<analysis>/steps/`. A `failed` step captures the
exception at the step boundary, writes the traceback to
`artifacts/timeline/<analysis>/failures/<step>.log`, and shows a one-line
message on the report page. Report pages are humanized: human step labels,
three significant figures, p-values floored at "< 0.001", and no dict literals
or machine paths.

Evidence steps attach figures via `FigureRef` (`path` relative to `reports/` +
one-line "How to read" caption) on `AnalysisStep.figures`; `timeline.md` embeds
them as `![caption](figures/...)` and `results/` pages as
`![caption](../figures/...)` (one link depth deeper).

Two Q-Q surfaces answer "is this normal?" at different scopes and converge on
small multiples:

- **Features Q-Q** (`src/broadway/discover/qq.py::plot_numeric_qq`) is the
  profile surface: **small multiples** — one subplot per numeric feature
  (per-feature z-score) plus a matching per-feature **distribution (histogram)
  grid in raw units** — because 7+ features don't read overlaid. Non-finite and
  zero-variance features are recorded, not plotted; the grid chunks into
  multiple PNGs beyond 12 features per figure.
- **Groups Q-Q** (`src/broadway/timeline/runners.py::run_normality`) is the
  normality gate: **small multiples**, one subplot per group, per-group z-score
  (capped at 12 groups).

The `audit` command renders the feature grids in a "Profile evidence" section
on `reports/audit/profile.md` from the `QqOverview` record
(`artifacts/discover/qq_overview.json`), with how-to-read lines and
standardization notes (Q-Q = per-feature z-score, distribution = raw units).
The discover Q-Q/distribution figures downsample the input to a configured
sample size (`qq_sample_size`, 10,000) once per figure and show a single
`n = …` in the suptitle; discrete distributions use midpoint bin edges so bars
center on the observed unique values (min/max auto-derived from the data).

The features Q-Q also draws config-driven **diagnostic zones** (`qq_zones` in
`configs/step/viz.yaml`): shaded tail bands and a central quantile band plus a
dashed zero-mass shelf. These are **visual references only** — they carry no
statistical verdict and are not part of any normality decision; the groups Q-Q
draws none of them. The shelf is keyed off `QqFeature.zero_rate`, the fraction
of zeros in the **downsampled plotted sample** (`finite`), so it reflects the
points actually shown, not the full column.

The profile surface also carries a **per-feature distribution diagnostics**
surface (`diagnostics` in `configs/step/viz.yaml`): a single heatmap
(`numeric_diagnostics.png`) over the `[skew, kurtosis, zero_rate]` columns,
z-normalized per column (cell text is the raw value), plus a matching
`mean`/`std`/`skew`/`kurtosis`/`zero_rate` table on `reports/audit/profile.md`.
Skew/kurtosis are computed on the same downsampled `finite` sample as
`zero_rate`/`mean`/`std`; the surface is a visual reference only and carries no
statistical verdict or threshold.
Suggestions are de-prescribed: `suggest.py` emits
`ds-pipeline decide --analysis <a> --method <method> --reason "..."` (never a
pre-filled method) and adds `--kind posthoc` at the post-hoc gate.

### Mode-specific pipelines

The three flows share the prefix `discover → etl → contracts → baseline`
and diverge on the mode tail: prediction appends `features → train → evaluate`,
hypothesis appends `stats`, and causal appends `causal`.

`baseline` is guidance, not a hard gate: it dispatches on
`AnalysisContract.mode` (prediction/hypothesis/causal), persists a
`BaselineResult` to `artifacts/baseline/`, and is included in each mode flow.
Run it explicitly:

```
ds-pipeline baseline --dataset <d> --analysis <a>
```

## Directory tree

```
broadway/
  src/broadway/
    config/schema.py        # Pydantic models (DatasetContract, StatsStep, TrainStep, FeaturesStep, FullStep, FlowConfig, ...)
    analysis/               # authored analytical intent + mode enforcement
      contracts.py          # AnalysisContract / HypothesisConfig / AnalysisMode + require_mode
    discover/               # read CSV/parquet → infer contract + observed profile
      module.py             # run(): writes configs/dataset/<name>.yaml + artifacts/discover/profile.json
      columns.py            # run(): prints `name: dtype` per source column (read-only)
      profile.py            # DatasetProfile / ColumnProfile (observed facts; identifier_score is descriptive only)
      qq.py                 # plot_numeric_qq: per-feature Q-Q small multiples + per-feature distribution grid (QqOverview evidence)
    onboard/                # onboarding/scaffolding (ds-pipeline init) + semantic inference hints
      infer.py              # dtype/null/cardinality/identifier/datetime hints (evidence only)
      models.py             # InferenceReport (typed hints)
      module.py             # init(): writes configs/{dataset,analysis,experiment} + profile sidecar
    contracts/              # contract-generated schema + role selectors
      pandera.py            # build_raw_schema(contract) -> pa.DataFrameSchema (generated)
      selectors.py          # feature/datetime/target column selectors over DatasetContract
    data/                   # data layer: format detection, joins, canonicalization, splits
      loader.py             # load() / load_with_audit(): csv/parquet/excel → left-join lookups + audits
      join_audit.py         # JoinAudit / audit_join(): key-match completeness
      lookup_value_audit.py # LookupValueAudit / audit_lookup_values(): matched-value quality (sentinel/na_values)
      cleaner.py            # clean() / canonicalize(): duplicates, missing encodings, datetime/numeric parsing
      splitter.py           # split(): time/random/stratified train/val split
    cleaning/               # deterministic structural cleaning (representation only, no domain cleaning)
      models.py             # ParseFailure / StructuralCleanResult (typed evidence)
      structural.py         # standardize_missing / parse_datetime / parse_numeric
    etl/                    # config-driven etl pipeline step
      module.py             # etl step: load_with_audit → canonicalize → validate → split → join/lookup_value lineage
    stats/                  # pandas/numpy stats library (no Spark)
      base.py               # stratified_sample
      plan.py               # AnalysisPlan (Pydantic model) + save/load
      effect_size.py        # eta², omega², epsilon_squared, Cohen's d, Hedges' g, group_imbalance
      assumptions.py        # Levene, skew/kurtosis/Shapiro
      anova.py              # run_anova, run_welch, run_kruskal
      post_hoc.py           # games_howell
      regression.py         # fit_ols, fit_robust, bp_jb
      diagnostic_models.py  # DiagnosticResult (question → evidence → ramification)
      diagnostics.py        # bp_test, jb_test, durbin_watson, plot_residuals, plot_residuals_vs_fitted, mean_specification_diagnostic
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
    baseline/               # guidance baseline, dispatched on AnalysisContract.mode
      contracts.py          # BaselineResult (Pydantic) + save/load
      prediction.py         # majority-class / mean baselines (sklearn accuracy/MAE)
      hypothesis.py         # naive effect = range of group means
      causal.py             # power-analysis sample size (reuses design_experiment)
      module.py             # pipeline step: dispatches on mode, persists BaselineResult to artifacts/baseline/
    features/               # generic feature machinery + the config-driven pipeline step
      schema.py             # FeatureSpec, build_engineered_schema
      pipeline.py, builders.py, transformers.py, module.py, contracts.py  # ds-pipeline features step
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
        pyfunc_wrapper.py   # ModelPyFunc — backward-compat loader for previously logged bare-model artifacts
    lineage/                # decision + lineage graph (sidecar records, Mermaid, run state)
      models.py             # DatasetRef, DatasetSlice, DecisionRecord, LineageRecord, LineageNode/Edge, LineageGraph, RunState
      ids.py                # node_id(kind, name) -> "kind:name"
      records.py            # write_record() sidecars under artifacts/lineage/records/
      graph.py              # build_graph(configs_dir, lineage_dir) -> LineageGraph
      mermaid.py            # to_mermaid(graph) -> mermaid source
      state.py              # current_state(graph, mode, goal, decisions) -> RunState
      module.py             # ds-pipeline lineage command
    samples/                # named-sample registry: definition → immutable artifact → validated consumption
      generate.py           # generate_sample(name): spec → data/samples/<name>@v<N>.parquet + provenance sidecar
      loader.py             # read_named_sample(name): integrity/digest/row-count/schema validation → Sample
      models.py             # Sample (df + spec + provenance)
    timeline/               # hypothesis walkthrough: step sequence + decision gates (the "analysis timeline")
      models.py             # StepStatus / AnalysisStep / AnalysisDecision / Suggestion / Alternative (Pydantic)
      sequence.py           # WalkthroughSequence / DecisionSpec / WalkthroughConfig.decisions / WalkthroughStepConfig.action + loaders (configs/flow/hypothesis_walkthrough.yaml, configs/step/walkthrough.yaml)
      module.py             # steps_dir/decisions_dir + save/load AnalysisStep / AnalysisDecision (artifacts/timeline/<analysis>/)
      evidence.py           # typed evidence: NormalityEvidence / VarianceEvidence / PosthocEvidence / ConclusionEvidence
      runners.py            # run_describe/run_normality/run_variance/run_omnibus/run_posthoc/run_conclusion (one runner per evidence step)
      walkthrough.py        # ds-pipeline walkthrough: _STEP_RUNNERS executor registry, advances the sequence, stops at decision gates, captures failures
      decide.py             # ds-pipeline decide: record() an AnalysisDecision (methods + parents from config-driven decision registry)
      suggest.py            # suggest_after / suggest_next via _DECIDE_SUGGESTERS / _EVIDENCE_SUGGESTERS registries
    reports/                # human-facing product surface (index + timeline + per-step markdown + figures)
      paths.py              # REPORTS_DIR / RESULTS_DIR / FIGURES_DIR / AUDIT_DIR / TIMELINE_PATH / INDEX_PATH (owns the surface paths)
      markdown.py           # render_result(title, sections) -> markdown
      sequence.py           # StatsSequence (configs/flow/stats_sequence.yaml)
      describe.py           # load_artifact / render / headline for the describe result
      registry.py           # RESULT_RENDERERS = {"describe": describe, ...}
      index.py              # render_index(question, stats_dir) + render_dashboard(...) -> reports/index.md
      results.py            # render_results/write_results -> reports/results/ (per-step pages + results index; humanize_* helpers)
      timeline.py           # render_timeline(...) -> reports/timeline.md (status table + per-step details)
      audit.py              # ds-pipeline audit: typed renderers -> reports/audit/{index,profile,transform,join,lookup_values}.md
  project/
    etl/
      process.py            # process_data(dataset): yellow_tripdata_*.parquet → training_data.parquet (contract-driven ingest step)
      process_config.py     # reads configs/project/taxi.yaml + configs/step/etl.yaml
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
  configs/flow/<mode>.yaml  # mode-specific step lists (prediction/hypothesis/causal)
  configs/flow/stats_sequence.yaml  # ordered stats-step list for reports/index.md (StatsSequence)
  configs/sample/<name>.yaml  # SampleSpec for `stats --sample`; named samples declare version/source/seed/size/columns/filters/schema
  configs/analysis/<name>.yaml  # authored analytical intent (AnalysisContract)
  tests/                    # test_base.py, test_anova.py, ... (pytest)
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
| `loader.load(dataset)` | `load` | `src/broadway/data/loader.py` |
| `loader.load_with_audit(dataset)` | `load_with_audit` | `src/broadway/data/loader.py` |
| `join_audit.audit_join(df, ...)` | `audit_join` | `src/broadway/data/join_audit.py` |
| `lookup_value_audit.audit_lookup_values(...)` | `audit_lookup_values` | `src/broadway/data/lookup_value_audit.py` |
| `sample.load_sample(name)` | `load_sample` | `src/broadway/lineage/sample.py` |
| `samples.generate_sample(name)` | `generate_sample` | `src/broadway/samples/generate.py` |
| `samples.read_named_sample(name)` | `read_named_sample` | `src/broadway/samples/loader.py` |
| `anova.run_anova(groups)` | `run_anova` | `src/broadway/stats/anova.py` |
| `anova.run_welch(groups)` | `run_welch` | `src/broadway/stats/anova.py` |
| `anova.run_kruskal(groups)` | `run_kruskal` | `src/broadway/stats/anova.py` |
| `assumptions.run_levene(groups)` | `run_levene` | `src/broadway/stats/assumptions.py` |
| `assumptions.check_normality(groups)` | `check_normality` | `src/broadway/stats/assumptions.py` |
| `post_hoc.games_howell(df, ...)` | `games_howell` | `src/broadway/stats/post_hoc.py` |
| `regression.fit_ols(df, formula)` | `fit_ols` | `src/broadway/stats/regression.py` |
| `regression.bp_jb(model)` | `bp_jb` | `src/broadway/stats/regression.py` |
| `diagnostics.durbin_watson(resid)` | `durbin_watson` | `src/broadway/stats/diagnostics.py` |
| `diagnostics.plot_residuals_vs_fitted(model, out)` | `plot_residuals_vs_fitted` | `src/broadway/stats/diagnostics.py` |
| `diagnostics.mean_specification_diagnostic(model, out)` | `mean_specification_diagnostic` | `src/broadway/stats/diagnostics.py` |
| `time_series.plot_acf(resid, ...)` | `plot_acf` | `src/broadway/stats/time_series.py` |
| `baseline.train_lgbm(X, y, ...)` | `train_lgbm` | `src/broadway/stats/baseline.py` |
| `baseline.evaluate(model, ...)` | `evaluate` | `src/broadway/stats/baseline.py` |
| `etl.process.process_data(dataset)` | `process_data` | `project/etl/process.py` |
| `reports.describe.render(summary)` | `render` | `src/broadway/reports/describe.py` |
| `reports.describe.headline(summary)` | `headline` | `src/broadway/reports/describe.py` |
| `reports.index.render_index(question, stats_dir)` | `render_index` | `src/broadway/reports/index.py` |

## Data flow

```
data/processed/training_data.parquet   (raw, 8.6M rows)
        │  pyarrow ParquetFile.iter_batches(batch_size=100_000)
        ▼
generate_sample_cache()   ── merge zone lookup (pickup_borough)
        │                     incremental per-borough stratified sample
        ├──▶ data/processed/joined_sample_{MODE}.parquet   (≈ SAMPLE_SIZE rows)
        ├──▶ data/processed/sample_meta_{MODE}.json        (params_hash)
        └──▶ data/processed/quality_report.json            (exact group sizes/means)
        │
        ▼
scripts (01..12)
        │  load_stratified_sample()  → random stratified groups
        │  load_time_slice()         → contiguous, time-sorted slice (filter pushdown)
        ▼
data/processed/*.json / *.png  (AnalysisPlan JSON, residual plots, ACF plot)
```

The pipeline CLI (`ds-pipeline`) runs a separate data flow, from raw parquet
through ingest and the etl step:

```
data/raw/yellow_tripdata_*.parquet
        │  ingest (process_data: Polars scan → CI-gated sample → clean → contract-validate → save)
        ▼
data/processed/training_data.parquet
        │  etl (load_with_audit → canonicalize → validate → split)
        ├──▶ data/processed/<name>_join_audit.json           (JoinAudit)
        ├──▶ data/processed/<name>_lookup_value_audit.json   (LookupValueAudit)
        ├──▶ data/processed/<name>_canonical.parquet         (canonical dataset)
        └──▶ data/processed/train.parquet / val.parquet      (or training_data.parquet)
```

`ingest` and `etl` sample only when `CI=true` (`ci_sample_size`, gated via
`sample_for_ci` / the etl step); local runs canonicalize the full dataset.

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

## Named sample registry

Steps never own sample paths, filtering, or sampling logic — they declare a
sample name, and the registry resolves it to an immutable versioned artifact:

```
configs/sample/<name>.yaml  (SampleSpec: version/source/seed/size/columns/filters/schema)
        ▼  generate_sample (broadway/samples/generate.py)  — fails if the version exists
data/samples/<name>@v<N>.parquet  +  <name>@v<N>.json  (provenance sidecar)
        ▼  read_named_sample (broadway/samples/loader.py)
validates: artifact sha256 == provenance.artifact_sha256  (integrity)
           definition sha256 == provenance.definition_sha256  (definition immutability)
           row count == provenance.row_count
           pandera schema built from spec.schema (dtype + nullable + op checks)
        ▼
Sample(df, spec, provenance)  →  steps consume by name (e.g. experiments/fare_prediction)
```

Regenerating a changed definition requires bumping `version` in the config —
the artifact is immutable; the loader rejects any drift between definition,
artifact, and provenance.

## Contracts

| Contract | Tool | Where |
|---|---|---|
| Configuration | Pydantic | `broadway/config/schema.py` |
| FullStep / FlowConfig | Pydantic | `broadway/config/schema.py` |
| AnalysisContract | Pydantic | `broadway/analysis/contracts.py` |
| DatasetSlice | Pydantic | `broadway/lineage/models.py` |
| DecisionRecord | Pydantic | `broadway/lineage/models.py` |
| LineageRecord | Pydantic | `broadway/lineage/models.py` |
| LineageGraph | Pydantic | `broadway/lineage/models.py` |
| RunState | Pydantic | `broadway/lineage/models.py` |
| AnalysisPlan | Pydantic | `broadway/stats/plan.py` |
| ExperimentDesign | Pydantic | `broadway/causal/contracts.py` |
| ExperimentResult | Pydantic | `broadway/causal/contracts.py` |
| BaselineResult | Pydantic | `broadway/baseline/contracts.py` |
| ArtifactTrace | Pydantic | `broadway/trace.py` |
| BaselineComparison | Pydantic | `broadway/evaluate/contracts.py` |
| EvaluationResult | Pydantic | `broadway/evaluate/contracts.py` |
| TrainingResult | Pydantic | `broadway/training/contracts.py` |
| DatasetProfile / ColumnProfile | Pydantic | `broadway/discover/profile.py` |
| Raw DataFrame | Pandera | `broadway/contracts/pandera.py::build_raw_schema(contract)` (generated) |
| Engineered features | Pandera | `project/features.py` (`FEATURE_SPECS`) → `broadway/features/schema.py::build_engineered_schema` |
| Python interfaces | type hints | throughout |

- The raw schema is generated at runtime from `DatasetContract.columns` — one `pa.Column` per contract entry (the raw 7 columns, not join-derived `pickup_borough`/`LocationID`). Dtypes are checked strictly (`coerce=False`); `null_count` is observed, not an invariant, so nullability is left at Pandera's default.
- Role-based column selection is `broadway/contracts/selectors.py` (`feature_columns`, `datetime_columns`, `target_columns`) — pure functions over the contract, no hardcoded names.
- Engineered-feature definitions: see `SKLEARN_PIPELINES.md` Decision 5 — the generic config-driven path is the SSOT; `FEATURE_SPECS` is demoted legacy.
- `DatasetContract` is the accepted schema (authored/authoritative); `DatasetProfile` / `ColumnProfile` describe observed facts computed at discover time. `identifier_score` is purely descriptive — discover only logs a recommendation, it never mutates roles or the contract.
- `DatasetContract` carries no `row_count` — observed counts live in `DatasetProfile` (discover) and `TransformAudit` (etl lineage). Datetime dtypes are normalized to canonical `datetime64` (`schema.py::normalize_dtype`).
- Lookup ingestion reads lookups with `keep_default_na=False` plus a per-lookup `na_values` policy, so nulls are attributable to the authored config. `JoinAudit` measures key completeness, while `LookupValueAudit` measures matched-value quality and records the `na_values` evidence.
- `broadway/data/loader.py::read_sample(dataset, sample, seed, columns, *, full)` is a seeded random draw of the dataset's raw parquet (lazy scan → optional column pruning → sample) — a fast experiment path that is NOT `DATA_MODE`-aware; the caller supplies `seed`. `project/data.py::read_training_sample` wraps it with the taxi `_contract` and `RANDOM_STATE`.
- Experiment layout: scripts live under `experiments/<category>/<name>/`; `experiments/results/` CSV outputs are tracked (`.gitignore` negates `!experiments/results/**/*.csv`), non-CSV outputs stay ignored. Root-level `experiment_*.py` / `*_experiment/` dirs predating this convention are grandfathered (forward-only).

## Mode enforcement

The declared analytical intent (`AnalysisContract.mode`) determines which steps are valid: `stats` requires `mode == "hypothesis"`, `causal` requires `"causal"`, and `train`/`evaluate` require `"prediction"`. `baseline` dispatches on mode (prediction/hypothesis/causal). Mismatches fail early via `broadway/analysis/contracts.py::require_mode`, which also errors when the `--analysis` contract is missing. `full` resolves the mode-specific flow (`configs/flow/{prediction,hypothesis,causal}.yaml`) via `resolve_full_steps`, so each mode runs only its valid tail.

`train` and `evaluate` read the persisted `BaselineResult` (from `artifacts/baseline/`) and report improvement over the baseline (`improvement_vs_baseline` in `broadway/baseline/improvement.py`). `BaselineResult` carries an `ArtifactTrace` (commit/dataset/analysis_goal) for lineage.

## Decision + Lineage

Broadway builds a run graph from persisted artifacts + decisions rather than a
hand-maintained diagram. Step modules write a `LineageRecord` sidecar after
saving their result; the `lineage` command assembles them into a graph and a
run-state summary.

The node chain begins `dataset → ingest → join → {etl, lookup_value}`, then
`analysis → baseline → … → decision`, with `describe`/`stats`/`causal`/
`training`/`evaluation` nodes joining as their sidecar records are produced.

- Sidecars: after saving its artifact, each step module calls `write_record`
  (`broadway/lineage/records.py`) to write a `LineageRecord {node_id, kind,
  artifact, parents}` under `artifacts/lineage/records/`. Domain contracts
  carry no lineage fields.
- Node ids are `kind:name` (`broadway/lineage/ids.py::node_id`), e.g.
  `baseline:taxi`, keyed by `AnalysisContract.name`.
- `DatasetSlice` is authored config (`configs/slice/<name>.yaml`);
  `DecisionRecord` is a runtime event in `artifacts/lineage/decisions/<id>.json`.
- `ds-pipeline lineage --analysis <n> --dataset <d>` builds the graph
  (`broadway/lineage/graph.py::build_graph`), writes `reports/lineage/graph.json`
  + `graph.md` (Mermaid), and prints a run-state summary (goal, stage,
  open/resolved decisions, `not_yet_run`, `ran_but_output_missing`).
- `not_yet_run` is derived only from lineage-emitting steps for the active
  mode flow (`broadway/lineage/state.py::LINEAGE_STEPS`), e.g. hypothesis =
  `profile`/`etl`/`baseline`/`stats`; `contracts` is never listed.
- `ran_but_output_missing` flags steps whose sidecar record exists but whose
  artifact file is absent — an integrity error, not a normal "not yet produced".

## Where to make changes

| Goal | Change |
|---|---|
| New config knob | `configs/step/stats.yaml` + matching field in `StatsStep` (`schema.py`) + constant in `data.py` |
| New raw feature | edit `configs/dataset/<name>.yaml` (dtype via `ds-pipeline columns --csv <path>`), then re-run `ingest` + `profile` — no code change |
| New loader | add function in `project/data.py`; reuse `read_training_data` / `_join_boroughs` |
| New statistical test | add function in `src/broadway/stats/` (pandas/numpy only) + document in `API.md` |
| New experiment script | add `project/scripts/NN_*.py`; import from `project.data` and `broadway.stats` |
| Change sample behavior | edit `generate_sample_cache` / `_params_hash` in `data.py`; bump stale `params_hash` by regenerating |
