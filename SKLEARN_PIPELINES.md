# SKLEARN_PIPELINES.md — migration plan: sklearn Pipelines throughout

Goal: one composable, cloneable, loggable preprocessing+model object per
experiment path. Today the platform has **zero** `sklearn.pipeline.Pipeline`
usage in `src/broadway/`; preprocessing is hand-rolled in two custom
`FeaturePipeline` classes and free functions, models are fit bare, and MLflow
logs the bare model — so preprocessing does not travel with the artifact.

Reference pattern already in-repo: `experiments/mlflow/_common.py::make_pipeline`
(`ColumnTransformer` passthrough/one-hot + model) and `_pca_pipeline`. The
platform adopts this pattern; experiments stop being ahead of it.

## Principles (unchanged)

- Config-driven: every preprocessing step and its params come from YAML via
  `src/broadway/config/schema.py`. No hardcoded step lists, no hardcoded
  thresholds. A pipeline recipe is authored config like a search space.
- Data-agnostic `src/`: no column names in platform code — column groups are
  config lists passed into generic transformers.
- Derive, don't maintain: no persisted fitted-preproc state outside the
  artifact itself; the fitted pipeline IS the state.
- Backward compatibility: existing pickled bare models and logged MLflow
  artifacts stay loadable; evidence JSON shapes unchanged unless stated.
- Immutable worker rules apply (`AGENT_WORKER_CONTRACT.md`). Full suite gate
  for every slice (`uv run pytest -q`, direct exit code), ruff + mypy,
  docs updated in the same commit, parity sync to `main` after push
  (`scripts/check_branch_parity.sh`).

## Inventory — what exists today

| Site | File | Current state |
|---|---|---|
| Trainer | `src/broadway/training/trainer.py::train` | fits bare registry model |
| HPO | `src/broadway/training/hpo.py` (`make_objective`) | fits bare model per trial |
| Model log/serve | `src/broadway/training/mlflow_utils.py::log_model`; `src/broadway/training/models/pyfunc_wrapper.py::ModelPyFunc` | logs/pickles bare model; caller must re-apply FeaturePipeline manually at inference |
| CV | `src/broadway/evaluate/validation.py::cross_validate` | manual KFold loop, `clone(model)` only — preprocessing outside folds would leak |
| Platform features | `src/broadway/features/pipeline.py::FeaturePipeline` | dict-based `_target_mappings` / `_freq_mappings`, free functions in `features/encodings.py` |
| Taxi binding | `project/ml_pipeline.py::FeaturePipeline` | deterministic features + merge-based target/frequency encodings + row-count guard |
| Script consumer | `project/scripts/12_lgbm_baseline.py` | carries FeaturePipeline + model as two objects |
| Experiments (hazard) | `experiments/causal_inference/05_joint_model_top10_zones.py`, `06_joint_model_time_of_day.py` | `pd.get_dummies` before split → train/test column-alignment hazard |
| Experiments (correct) | `experiments/mlflow/_common.py` | Pipeline + ColumnTransformer — the reference |

## Target design

1. **sklearn-compatible transformers** for the existing encodings:
   `TargetEncoding` and `FrequencyEncoding` as
   `BaseEstimator`/`TransformerMixin` classes wrapping the current math
   (`fit_target_encoding` / `fit_frequency_encoding` logic). Same smoothed
   formulas, same outputs — no statistical change.
2. **Config-declared pipeline recipes**: `configs/experiment/<name>.yaml`
   gains an optional `preprocessing:` block (ordered steps, each with type +
   columns + params), parsed into Pydantic (`schema.py`) and built by a
   generic builder into a `sklearn.pipeline.Pipeline`. Absent block = bare
   model wrapped in a trivial passthrough Pipeline (behavior identical).
3. **Trainer/HPO/CV operate on the whole Pipeline**: fit, clone, tune, log —
   never the bare model alone.
4. **MLflow logs the Pipeline** (`mlflow.sklearn.log_model` already used by
   `log_model` — it just receives a Pipeline now); pyfunc predict takes raw
   feature-frame input.
5. **The loader path is a required, typed, validated config field — not a
   convention.** `ExperimentConfig.data_source: DataSourceRef` (no default)
   declares `loader` (`canonical | joined | named_sample | pinned`), the
   sample `version` (required for `named_sample`), and the bound
   `schema_contract`. Train and predict both resolve through the same ref,
   so "wrong loader at predict" has no compilable code path. Provenance
   logging to MLflow is serializing a field that already exists — no
   parallel bookkeeping.

## Change list (ordered slices)

### Slice 2 — preprocessing schema + recipe builder (DataSourceRef landed, `6d8bf00`)

LANDED in 2a (`6d8bf00`): `DataSourceRef` required on `ExperimentConfig`
(`loader` Literal + version-iff-`named_sample` validator + `schema_contract`),
all four experiment YAMLs bound (canonical ×3, joined for taxi), etl
step-eligibility dispatch consumes the ref at etl start — git history is
the record. Full ref-driven resolution in train/predict arrives with Slices
3–4; the column-cross-check validator is a named gap closed below.

2a enforcement scope — exact, so "fails loud" is never assumed beyond what
exists:

- **The only loader dispatch in 2a is the etl step-eligibility guard**
  (`src/broadway/etl/module.py::_assert_data_source_supported`, called before
  any data is read). An experiment declaring a pre-built loader
  (`named_sample`/`pinned`) fails etl with
  `ValueError: etl cannot run for data_source.loader 'named_sample'
  (supported: canonical, joined)` — the message names the field, the
  offending value, and the supported set; there is no KeyError or
  FileNotFoundError from inside a loader. It fires at etl start, not at
  config-load: `load_config` accepts any loader value, and train/predict do
  not read the ref until Slices 3–4.
- **`schema_contract` is required but NOT cross-checked in 2a.** A wrong
  value (e.g. `schema_contract: raw` on a dataset whose contract is not raw)
  passes config-load silently — same status as `preprocessing:`, which is
  still absent from `ExperimentConfig` until 2b. The chosen values (`raw` for
  test-canonical and taxi-joined) are truthful bindings to the raw-boundary
  contract (`build_raw_schema`), not enforced invariants; the cross-check
  against the referenced schema module lands in 2b with the recipe builder.

REMAINING in this slice:

- `src/broadway/config/schema.py`: `PreprocessingStepConfig` (type, columns,
  params) + `ExperimentConfig.preprocessing: list[PreprocessingStepConfig] = []`.
- New versioned schema modules under `schemas/` (e.g.
  `schemas/named_sample_v3.py`) reusing `build_raw_schema` /
  engineered-schema builders — reviewable code diffs + explicit version
  tags, NOT serialized JSON snapshots (stored derived state is forbidden by
  `AGENT_WORKER_CONTRACT.md`; pinned artifacts like `ratecode1_sample.json`
  remain the precedent for deliberate pins).
- New `src/broadway/features/recipe.py::build_pipeline(cfg) -> sklearn.pipeline.Pipeline`
  — generic registry of step types (`target_encoding`, `frequency_encoding`,
  `one_hot`, `passthrough`, scaler types as needed later). Unknown type fails
  loud. Column lists are name-driven, enforced against the schema contract
  referenced by `data_source` (closes the cross-check gap named above).
- Tests: builder round-trip (YAML → Pipeline → get_params), unknown-step
  failure, empty-block passthrough identity, cross-check validator.
- Acceptance: full suite green; CI parse-all passes over every experiment
  YAML; `ds-pipeline train --experiment baseline` produces identical metrics
  to pre-change run (evidence: pasted metric lines).

### Slice 3 — trainer + HPO on the whole Pipeline

- `src/broadway/training/trainer.py::train`: build Pipeline from
  experiment config, fit Pipeline. `TrainingResult` unchanged (model_type +
  params remain the record; the recipe is config-derived, not stored).
- `src/broadway/training/hpo.py`: objective fits the Pipeline; search spaces
  may address preprocessing params as `pre__<step>__<param>` when a recipe is
  declared. Registry validation (`MODEL_META.allowed_params`) applies to the
  model segment only — document the boundary in `HPO_TRAINING.md`.
- `mlflow_utils.log_model` receives the fitted Pipeline **plus an explicit
  signature**: `signature=infer_signature(X, y)` on the log call (MLflow's
  native schema-enforcement mechanism — no custom hash comparison anywhere;
  that idea is dropped). MLflow owns the fit/predict shape contract; Pandera
  owns within-column dtype/nullability depth.
- Provenance: serialize `cfg.experiment.data_source` into the MLflow run
  (params/tags) — the provenance IS the config; no parallel bookkeeping.
- Tests: HPO trial refits preprocessing per trial (leakage guard test with a
  stateful counter-transformer), trainer returns Pipeline, logged artifact
  reloads via `mlflow.sklearn.load_model`, carries the signature, and
  predicts on a raw frame.
- Acceptance: full suite green; `train --experiment hyperopt` smoke run;
  `reports/lineage/graph.*` regenerated by the lineage command only if its
  records change (they should not).

### Slice 4 — inference surface

- `pyfunc_wrapper.ModelPyFunc`: keep for previously logged bare-model
  artifacts (backward compat); new-path models load through
  `mlflow.sklearn.pyfunc` which carries preprocessing inside the Pipeline
  and enforces the logged signature at predict time automatically.
  Mark the wrapper's scope in its docstring.
- **Retirement is a checked condition, not a vibe** (RESOLVED — decision 3):
  a manifest check (champion-promotion path or CI script) lists deployed
  champions by logging path — bare-model vs Pipeline+signature. When the
  bare-model list is empty, retirement becomes a mechanical PR.
- `evaluate/module.py` consumes transformed feature parquets — unchanged.
- Tests: champion predict path on raw input frame; signature mismatch at
  predict fails loud (wrong column set / wrong dtype); manifest check returns
  the champion/path listing (asserted structure).
- Acceptance: full suite green; README + dataflow.md updated in same commit
  (inference section); manifest-check output pasted.

## Decision points — all resolved

1. **RESOLVED — CV implementation**: `sklearn.model_selection.cross_validate`
   replaces the hand loop. Standard-library mechanism over a hand-maintained
   loop doing the same job; the `_mean_metrics` decimals contract is a test
   concern, handled by the parity-test-first sequencing used when the swap
   landed (`aee531b`). No dual path survived the swap commit.
2. **RESOLVED — encoder origin**: custom sklearn-compatible wrappers around
   the existing smoothed math. Correctness/compatibility fork, not an
   enforcement question: `sklearn.preprocessing.TargetEncoder` silently
   changes smoothing semantics and drops multi-column keys, violating the
   "no statistical change" principle at the top of this doc.
3. **RESOLVED — ModelPyFunc retirement**: keep-until-unused with "unused" as
   a checked condition — a manifest check listing deployed champions by
   logging path (bare-model vs Pipeline+signature), wired into Slice 4.
   Empty list = mechanical retirement PR; nobody has to remember to notice.
4. **RESOLVED — recipe location**: `configs/experiment/<name>.yaml`.
   `DataSourceRef` and `preprocessing:` are tightly coupled — a schema
   contract is meaningless without the loader/version that produced the data.
   Splitting them across files recreates the "two things that must agree but
   live apart" problem this doc eliminates elsewhere; co-locating them lets
   the Slice 2 validator cross-check column lists against the schema
   contract at config-load time (as conflict-2's resolution promises).
5. **RESOLVED — column selection philosophy**: name-driven selection,
   enforced by the schema contract referenced in `DataSourceRef` (not a
   separate dtype-vs-name debate). The dtype-driven `feature_columns`
   selector is retired when the last caller switches; until then both exist
   only inside the transition slices.

## Test-suite impact

Full suite gate applies to every slice touching `src/` or `tests/`
(`uv run pytest -q`, direct exit code). Platform tests stay synthetic-only —
`tests/test_platform_hygiene.py` enforces no taxi coupling and will fail any
violating new test. Existing coverage gate: CI runs
`pytest --cov=src/broadway --cov-fail-under=85`, so every new module ships
with its tests in the same slice.

Existing test files that touch migration sites (verified by import grep):

| Test file | Site it pins | Slice |
|---|---|---|
| `tests/test_config.py` | config schema/loader round-trip | 2 |
| `tests/test_contracts.py` | engineered schema / contract fixtures | 2 |
| `tests/test_hpo.py` | `training/hpo.py` objective/study | 3 |
| `tests/test_optuna_extended.py` | HPO determinism/resume | 3 |
| `tests/test_registry.py` | model registry params/allowed_params | 3 |
| `tests/test_training_contracts.py` | `TrainingResult` shape | 3 |
| `tests/test_mlflow_utils_extended.py` | `log_model` / artifact URIs | 3, 4 |
| `tests/test_integration.py` | end-to-end pipeline steps | 3 |
| `tests/test_loud_failures.py` | failure-path loudness | 3 |
| `tests/test_explain.py` | explainability over fitted model | 4 |

New tests per slice:

- Slice 2: `tests/test_recipe.py` — YAML → Pipeline round-trip, unknown step
  fails loud, empty block = passthrough identity.
- Slice 3: extend `tests/test_hpo.py` — preprocessing refits per trial
  (counting-transformer leakage guard); trainer returns Pipeline; logged
  artifact reloads and predicts on raw frame.
- Slice 4: champion predict path test on raw input frame.

## GitHub Actions impact

`.github/workflows/ci.yml` changes:

1. **Branch triggers** — `on.push.branches` and `on.pull_request.branches`
   gain `sklearn` so the branch gets CI from the first push. Without this the
   branch runs ungated.
2. **Coverage gate** (`--cov-fail-under=85`) — unchanged command; new modules
   (`features/recipe.py`, versioned `schemas/`) must land with tests to
   hold 85%.
3. **Parity job** — unchanged: it diffs `origin/main` vs `origin/taxi`
   regardless of the branch CI runs on, and keeps passing while `sklearn`
   work is unmerged. NOTE: `.github/` is on the parity shared-surface list,
   so when this workflow change merges to `taxi`, `main` must be synced
   (`scripts/check_branch_parity.sh --sync`) in the same flow.
4. **CD job** — deliberately NOT extended to `sklearn`: images publish on
   push to `main`/`taxi` only. A feature branch does not ship images.
5. **Experiment smoke** (`experiments.py verify`) is taxi-ref-gated and stays
   so; it reappears when slices merge back to `taxi`.

## Loader ↔ sklearn boundary — conflicts and mitigations

sklearn loads nothing; it consumes one in-memory DataFrame per fit/transform.
All conflicts live where the custom loaders hand data to the Pipeline.

1. **Streaming vs in-memory.** `project/data.py::generate_sample_cache`
   streams 8.6M rows via `pq.ParquetFile.iter_batches` with incremental
   per-borough sampling; `data/loader.py::read_sample` uses a lazy
   `pl.scan_parquet`. A Pipeline cannot sit inside the streaming loop.
   Mitigation: pipeline stays downstream of load; loaders remain the owners
   of materialization and sampling. Fine for dev/live sample sizes; the
   `full=True` path keeps its memory assumption — document it, don't hide it.
2. **Silent dtype-driven column drop.** `utils.feature_columns` =
   `df.select_dtypes(include="number").drop(target)` — non-numeric columns
   vanish silently before `.fit`. Harmless for bare tree models; with a
   name-based `ColumnTransformer` it becomes a KeyError or silent feature
   loss. Resolution rides on decision 5 above: switch to name-driven
   selection from config lists in the same slice that introduces
   ColumnTransformer (Slice 2/3), never before both exist.
3. **Feature-name contract at predict.** Pipelines validate feature names
   fit-vs-predict; the loaders emit different schemas by path (canonical
   parquet, joined cache, named samples `@vN`, pinned `ratecode1_sample`).
   Resolution (structural, not conventional): `DataSourceRef` is a required,
   typed field on `ExperimentConfig` — train and predict resolve loaders
   through the same declared ref, so an undeclared loader path cannot be
   reached. MLflow's logged signature enforces the shape contract at predict.
   No "same path" convention to remember — invalid configs fail at
   config-validation time.
4. **Index alignment.** Lookup joins (`load_with_audit`, `_join_boroughs`)
   and both target/frequency encoders merge → non-default, possibly
   non-unique indexes; sklearn transforms are positional. Resolution: the
   landed transformers (Slice 1, commit `7ad52e7`) apply mappings
   by key-map (no merge) so row order and index are preserved; the taxi
   row-count guard stays.
5. **dtype drift through parquet round-trips.** etl writes `index=False`
   parquet; re-read can shift datetime units / nullable dtypes. Resolution
   (structural): versioned schema modules under `schemas/` bound by
   `DataSourceRef.schema_contract` — a schema change is a reviewed code diff
   plus an explicit version bump, never a side-effect of editing a loader.
   The named-sample registry keeps its load-time validation; the schema
   module is what it validates against.
6. **Double-sampling / split ownership.** Loaders own sampling+seed
   (`read_sample(seed)`, `DATA_MODE` caches, registry version bumps);
   sklearn owns resampling too. Mitigation: rule — sampling and splitting
   stay in `data/splitter.py` / loaders; Pipeline and CV never sample. The
   Slice 3 leakage-guard test asserts preprocessing refits per fold but
   never resamples.
7. **Time-ordering vs CV shuffle.** `evaluate/validation.py::cross_validate`
   uses `KFold(shuffle=True)` — shuffled folds destroy time structure (the
   script-10 DW lesson). Resolution: config-driven `cv_kind`
   (`kfold` vs `time_series_split`) landed in place of the hardcoded shuffle;
   time-split experiments route to `TimeSeriesSplit`.
8. **Polars→pandas edge.** `read_sample` ends `df.to_pandas()`, which can
   yield Arrow-backed `str` dtype; some sklearn selector paths expect
   `object`. Mitigation: explicit dtype normalization at the loader boundary
   (single cast site in `read_sample`), covered by a boundary test.
9. **Storytelling/CV disjointness.** walkthrough/timeline/reporting steps
   compute their own statistics with zero imports of `broadway.evaluate`
   (sole production caller: `evaluate/module.py`), verified 2026-08-22 —
   pinned by the `tests/test_platform_hygiene.py` import guard.

## Non-goals

- No new statistics, no formula changes, no resampling changes.
- No Spark/Pyspark pipeline migration.
- No automatic hyperparameter search over preprocessing (only enabling it via
  `pre__` paths when authored).
- No change to evidence JSON contracts or report surfaces (except none).

## Sequencing & gates

Slices 2→3 are strictly sequential (each builds on the previous commit).
4, 5 independent after 3. 6, 7 landed on the Slice 1 foundation (`99acdba`,
`c593c88`). Detailed contracts authored just-in-time per
`CONTRACT_TEMPLATE.md`, against the just-committed state. Every slice: green
suite (direct exit code), ruff, mypy `src/broadway`, one logical commit,
push `taxi`, then parity check/sync to `main`.

### End-to-end verification criteria

Standing bar for any end-to-end/dogfood verification: numerics must be
identical across identical invocations — all metric values, `cv_metrics`
values, and params. Only volatile-by-design fields, compared by name, are
excluded from byte-equality: `trace.created_at`
(`artifacts/baseline/baseline.json`); the MLflow-generated `artifact_path`
`models:/m-<id>` (`artifacts/training/training_result.json`); the
champion-registry state fields `promote`, `reason`,
`comparison.metrics.*.champion`, `warnings`
(`artifacts/evaluation/metrics.json`). Timestamps, registry-assigned URIs, and
mutable champion state are external-world coupling, not pipeline
non-determinism.

Known gaps (accepted as-is, no pipeline defect): the MLflow integer-column
schema hint (`Inferred schema contains integer column(s)…`, moot once Slice 2b
lands explicit `infer_signature` on logged models) and the MLflow ambiguous
dataset-source UserWarning are third-party emissions from `mlflow.types` /
`dataset_source_registry`, not sklearn pipeline issues. The optuna
`ExperimentalWarning: heartbeat_interval` is likewise a deliberate opt-in to a
used experimental feature — RDB dead-trial recovery depends on the heartbeat.
