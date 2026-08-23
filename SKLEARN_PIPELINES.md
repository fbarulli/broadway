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
- Immutable worker rules apply (`WORKER_CONTRACT.md`). Full suite gate
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
| Platform features | `src/broadway/features/pipeline.py::FeaturePipeline` | fits/transforms sklearn `TargetEncoding` / `FrequencyEncoding` transformers (`features/transformers.py`), output value-pinned in tests |
| Taxi binding | `project/ml_pipeline.py::FeaturePipeline` | deterministic features + merge-based target/frequency encodings + row-count guard |
| Script consumer | `project/scripts/12_lgbm_baseline.py` | carries FeaturePipeline + model as two objects |
| Experiments (hazard) | `experiments/more_modeling/05_joint_model_top10_zones.py`, `06_joint_model_time_of_day.py` | `pd.get_dummies` before split → train/test column-alignment hazard |
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

### Slice 4 — inference surface

- `pyfunc_wrapper.ModelPyFunc`: keep for previously logged bare-model
  artifacts (backward compat); new-path models load through
  `mlflow.sklearn.pyfunc` which carries preprocessing inside the Pipeline
  and enforces the logged signature at predict time automatically.
  Mark the wrapper's scope in its docstring.
- **Retirement is a checked condition, not a vibe**:
  a manifest check (champion-promotion path or CI script) lists deployed
  champions by logging path — bare-model vs Pipeline+signature. When the
  bare-model list is empty, retirement becomes a mechanical PR.
- `evaluate/module.py` consumes transformed feature parquets — unchanged.
- Tests: champion predict path on raw input frame; signature mismatch at
  predict fails loud (wrong column set / wrong dtype); manifest check returns
  the champion/path listing (asserted structure).
- Acceptance: full suite green; README + dataflow.md updated in same commit
  (inference section); manifest-check output pasted.

## Decision 5 — column selection philosophy (RATIFIED 2026-08-22 — Contract C/D spec)

Human-ratified this date; portable, do not re-litigate. Two corrections to the
earlier text: (a) **frame column order is preserved, NOT sorted** — the retired
dtype selector kept frame order and eligible-column output stays byte-identical,
including MLflow logged-signature column stability; (b) **decision-6 extension**:
the schema contract may name features-step output — `schema_contract:
"engineered"` resolves through `build_generic_feature_specs(dataset, features)`
(base ∪ derived ∪ encodings), so an experiment's declared surface is its model
input contract.

Name-driven selection, enforced by the schema contract referenced in
`DataSourceRef` (not a separate dtype-vs-name debate). The dtype-driven
`feature_columns` selector is retired when the last caller switches; until
then both exist only inside the transition slices.

*Implementation resolution (ratified at Slice-4 close, refined by the
Contract C census):* option (b) — dtype demoted from **selector** to
**assertion**. Eligible columns derive from the experiment's schema
contract; numeric-only becomes a fail-loud assertion guarding the
passthrough-only case canonical experiments use, naming the offending
column ("categorical column X has no preprocessing step and no
numeric-selector fallback applies"). Explicit recipe columns bypass the
assertion. (a) rejected — making empty preprocessing illegal breaks
"absent block = passthrough identity" for zero functional gain;
(c) rejected — deferral leaves Decision 5 open in practice forever (same
shape as the `fit(X, y)` deferral).

*Refined resolution (Contract C — census-verified, read-only):* the
numeric-post-engineering assumption does NOT hold for any shipped
experiment; the assertion will fire on current configs until each leak is
repointed to its single declaration home. Leak census (post-engineering
frames, exact pipeline path):
- test / `raw` — covers `baseline`, `engineered`, `hyperopt`: three
  distinct shipped configs sharing the dataset config
  `configs/dataset/test.yaml`, with an identical exclusion-side leak (they
  diverge only on the inclusion side, per the repoint-mapping bullet). The
  leak: `feature_3` (object) — declared `object` in
  `configs/dataset/test.yaml` and stays object post-engineering (the
  features step keeps all raw columns; include/derived/encodings only add).
- taxi / `joined`: `pickup_datetime` (datetime64[ns]) and
  `Borough`/`Zone`/`service_zone` plus their `_lookup` duplicates (object)
  in the joined frame.
Repoint mapping — every leak resolves to exactly one declaration home; no
schema-module extension needed:
- `engineered_feature_1` (float64, builder `source_copy`) and
  `feature_3_target_enc` (float64, target-encoding output) →
  `features.derived`/`features.encodings` → the generic engineered schema
  (`build_generic_feature_specs`, validated in `features/module.py`) — the
  only engineered-column declaration mechanism the sklearn path sees.
- taxi derived (`pickup_hour`…`same_borough`) and
  `pickup_location_id_target_enc` → the generic engineered schema. The
  legacy `project/features.py` `FEATURE_SPECS` is NOT a candidate home:
  `src` has zero `project.*` imports (the generic path is deliberately
  legacy-free), and legacy declares `passenger_count` int64 in direct
  contradiction of the dataset contract float64 — a dtype-correctness
  failure on its own terms, independent of reachability. The taxi home is
  settled twice over: reachability + dtype-correctness.
Resolved (CONTRACT S): multi-input builder inputs are config-declared —
`FeatureConfig.builder_params` (`group_col`, `lookup_col`) forwards to
`build_derived`, whose signature stays the single source of the generic
defaults (`group`/`group_lookup`). Taxi declares
`group_col: Borough` / `lookup_col: Borough_lookup` against the joined
frame, so `same_borough` executes. `same_group` output is int64 (vs legacy
int8) — a deliberate re-baseline; sentinel semantics are inherited
unchanged (observed evidence, not normalized): `NaN==NaN -> 0`,
`"Unknown"=="Unknown" -> 1`.
Category-match semantics pinned to observed variance: numeric-category
matching over runtime dtypes {int32, int64, float64} — int32 vs int64 is
legitimate observed variance (taxi/base). No float32, no nullable
Int64/Float64, no bool in any shipped post-engineering frame; the
float32→float64 widening example is not census-observed and rests on
external authority only — dropped from the justification. Non-numeric
context excluded by the assertion: object, datetime64[ns].
Acceptance pins: after repoint, the inverse census (numeric runtime columns
− schema-module-declared − target) is empty for every shipped experiment;
the assertion fires with the exact named-column message for
categorical-in-schema-without-step; explicit recipe columns bypass.
Parity contract (FIX_2): an experiment's `features.include` order must equal the features-step producer write order — enforced by `tests/test_engineered_order_parity.py` (the config aligns to the producer; never the reverse).

## Test-suite impact

Full suite gate applies to every slice touching `src/` or `tests/`
(`uv run pytest -q`, direct exit code). Platform tests stay synthetic-only —
`tests/test_platform_hygiene.py` enforces no taxi coupling and will fail any
violating new test. Existing coverage gate: CI runs
`pytest --cov=src/broadway --cov-fail-under=85`, so every new module ships
with its tests in the same slice. A boundary-contract suite exists —
`tests/test_boundary_contracts.py` parametrically flips one column's dtype at
every DataFrame writer→reader boundary and requires a loud failure; known-leaky
boundaries stay in it as strict-xfail tripwires. As of 2026-08-23 the READ side
is enforced: `features/generic.py` builds an ORDERED pandera schema and the
training/evaluate loaders validate every engineered frame on read (Contract H,
`3db7b4b`); `taxi.yaml`'s include order matches the producer write order under
a standing parity test (`tests/test_engineered_order_parity.py`, FIX_2,
`3ee1ef5`); two gap classes remain strict-xfail pending the ratified Option-E
closure contract (FIX_4: evidence-tagged coercion + unique-label merge guard) —
live status in `FIXES.md`'s ledger.

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
`models:/m-<id>` and the wall-clock `train_time_seconds` (training duration;
machine-dependent by nature) (`artifacts/training/training_result.json`); the
champion-registry state fields `promote`, `reason`,
`comparison.metrics.*.{champion,delta,delta_pct}`, `warnings`
(`artifacts/evaluation/metrics.json`; the delta pair is champion-derived —
null on a first run with no champion, computed once a champion exists —
enforced by `scripts/check_e2e_determinism.sh`). Timestamps, registry-assigned URIs, and
mutable champion state are external-world coupling, not pipeline
non-determinism. Whitelist SSOT: the EXACT/PATTERN table inside
`scripts/check_e2e_determinism.sh` — the list above only points there.

Known gaps (as of 2026-08-23): the MLflow warning classes are RESOLVED
(`c324583`, FIX_1) — the duplicate `LocalArtifactDatasetSource` registration is
bypassed structurally via a pre-built DatasetSource through the public
`get_registered_sources()` accessor, model logging uses `name=`, and the
integer-column hint is suppressed ONLY at the `infer_signature` call site under
an evidence comment (the hinted int columns are provably null-free; see the
comment block in `training/module.py`). The optuna
`ExperimentalWarning: heartbeat_interval` remains a deliberate opt-in to a
used experimental feature — RDB dead-trial recovery depends on the heartbeat.
