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

## Change list (ordered slices)

### Slice 1 — platform transformers (foundation)

- New `src/broadway/features/transformers.py`:
  - `TargetEncoding(BaseEstimator, TransformerMixin)` — columns, target,
    smoothing; `fit` computes mappings, `transform` applies with global-mean
    fill. Delegates to (or absorbs) `features/encodings.py` math.
  - `FrequencyEncoding(BaseEstimator, TransformerMixin)` — same shape.
  - Both must survive `sklearn.base.clone` and refit-per-fold cleanly
    (`get_params`/`set_params` correct — no state leaked through constructor).
- `src/broadway/features/pipeline.py::FeaturePipeline` re-expressed over the
  transformers (public behavior unchanged: `fit(df, target, smoothing)` /
  `transform(df, cfg, target, freq_fill)` signatures preserved this slice).
- Tests: fit/transform equivalence vs old encodings (golden numbers),
  clone-refit independence, unseen-category fill behavior.
- Acceptance: full suite green; no renderer/evidence changes.

### Slice 2 — config schema + pipeline builder

- `src/broadway/config/schema.py`: `PreprocessingStepConfig` (type, columns,
  params) + `ExperimentConfig.preprocessing: list[PreprocessingStepConfig] = []`.
- New `src/broadway/features/recipe.py::build_pipeline(cfg) -> sklearn.pipeline.Pipeline`
  — generic registry of step types (`target_encoding`, `frequency_encoding`,
  `one_hot`, `passthrough`, scaler types as needed later). Unknown type fails
  loud. No dataset terms anywhere.
- `configs/experiment/*.yaml` unchanged until a recipe is authored (absent =
  passthrough; zero behavior drift).
- Tests: builder round-trip (YAML → Pipeline → get_params), unknown-step
  failure, empty-block passthrough identity.
- Acceptance: full suite green; `ds-pipeline train --experiment baseline`
  produces identical metrics to pre-change run (evidence: pasted metric lines).

### Slice 3 — trainer + HPO on the whole Pipeline

- `src/broadway/training/trainer.py::train`: build Pipeline from
  experiment config, fit Pipeline. `TrainingResult` unchanged (model_type +
  params remain the record; the recipe is config-derived, not stored).
- `src/broadway/training/hpo.py`: objective fits the Pipeline; search spaces
  may address preprocessing params as `pre__<step>__<param>` when a recipe is
  declared. Registry validation (`MODEL_META.allowed_params`) applies to the
  model segment only — document the boundary in `HPO_TRAINING.md`.
- `mlflow_utils.log_model` now receives the fitted Pipeline (no signature
  change).
- Tests: HPO trial refits preprocessing per trial (leakage guard test with a
  stateful counter-transformer), trainer returns Pipeline, logged artifact
  reloads via `mlflow.sklearn.load_model` and predicts on a raw frame.
- Acceptance: full suite green; `train --experiment hyperopt` smoke run;
  `reports/lineage/graph.*` regenerated by the lineage command only if its
  records change (they should not).

### Slice 4 — inference surface

- `pyfunc_wrapper.ModelPyFunc`: keep for previously logged bare-model
  artifacts (backward compat); new-path models load through
  `mlflow.sklearn.pyfunc` which carries preprocessing inside the Pipeline.
  Mark the wrapper's scope in its docstring; retire later once no champion
  needs it (decision recorded, not silent).
- `evaluate/module.py` consumes transformed feature parquets — unchanged.
- Tests: champion predict path on raw input frame.
- Acceptance: full suite green; README + dataflow.md updated in same commit
  (inference section).

### Slice 5 — cross-validation correctness

- `src/broadway/evaluate/validation.py::cross_validate`: accept any estimator
  (model or Pipeline), keep the KFold loop but `clone(estimator)` so
  preprocessing refits per fold — or replace the loop with
  `sklearn.model_selection.cross_validate` (equivalent output dict; prefer
  whichever keeps `_mean_metrics` contract). Decision point below.
- Tests: fold-independence with a counting transformer (each fold sees fresh
  fit), score parity with previous implementation on a fixed seed.
- Acceptance: full suite green.

### Slice 6 — taxi binding

- `project/ml_pipeline.py::FeaturePipeline`: internals re-expressed as
  sklearn-compatible transformer chain (deterministic feature step + the two
  encodings from Slice 1); row-count merge guard stays in `transform`;
  `ENGINEERED_SCHEMA.validate` stays terminal. Public API
  (`fit`/`transform`/`fit_transform`) unchanged.
- `project/scripts/12_lgbm_baseline.py`: passes the single pipeline object;
  outputs identical.
- Tests under `project/tests/`: transform-equivalence golden check vs old
  implementation on a seeded sample.
- Acceptance: full suite green; script `12` rerun evidence (pasted metrics).

### Slice 7 — experiments de-hazard

- `experiments/causal_inference/05_joint_model_top10_zones.py`,
  `06_joint_model_time_of_day.py`: replace pre-split `pd.get_dummies` with
  `ColumnTransformer([OneHotEncoder(handle_unknown="ignore")])` inside a
  Pipeline fitted on train only.
- `experiments/mlflow/_common.py::make_pipeline`: leave as-is (already the
  reference); optionally note it now mirrors the platform builder.
- Acceptance: ruff + touched scripts rerun (experiments tier — pytest does
  not collect `experiments/`); pasted script output.

## Decision points (user calls, not silent)

1. **CV implementation**: hand loop with whole-pipeline clone (small diff,
   keeps `_mean_metrics` decimals contract) vs
   `sklearn.model_selection.cross_validate` (stdlib-correct, replaces ~15
   lines). Recommendation: `cross_validate`.
2. **Encoder origin**: custom sklearn-compatible wrappers around the existing
   smoothed math (recommended — preserves exact formulas and multi-column
   keys) vs adopting `sklearn.preprocessing.TargetEncoder` (different
   smoothing semantics, single-column; would change numbers).
3. **ModelPyFunc retirement timing**: keep-until-unused (recommended) vs
   remove now with a champion reload migration.
4. **Where recipes live**: `configs/experiment/<name>.yaml` (recommended —
   recipe travels with the experiment) vs `configs/step/train.yaml`.

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
| `tests/test_generic_features.py` | `features/pipeline.py` fit/transform | 1 |
| `tests/test_builders.py` | `features/builders.py` (derived steps) | 1 |
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
| `tests/test_feature_selection.py` | estimator `.fit(X, y)` contract | 5 |
| `tests/test_evaluate_contracts.py` | evaluation result shapes | 5 |
| `tests/test_metrics_extended.py` | metrics consumed by CV | 5 |

New tests per slice:

- Slice 1: `tests/test_transformers.py` — golden equivalence vs old encoding
  math, `clone` + refit independence, unseen-category fill, synthetic data
  only.
- Slice 2: `tests/test_recipe.py` — YAML → Pipeline round-trip, unknown step
  fails loud, empty block = passthrough identity.
- Slice 3: extend `tests/test_hpo.py` — preprocessing refits per trial
  (counting-transformer leakage guard); trainer returns Pipeline; logged
  artifact reloads and predicts on raw frame.
- Slice 4: champion predict path test on raw input frame.
- Slice 5: fold-independence test (fresh fit per fold) + seeded score parity.
- Slice 6: new `project/tests/test_ml_pipeline.py` — transform-equivalence
  golden check vs old implementation on a seeded sample (taxi-tier tests may
  couple to taxi).

## GitHub Actions impact

`.github/workflows/ci.yml` changes:

1. **Branch triggers** — `on.push.branches` and `on.pull_request.branches`
   gain `sklearn` so the branch gets CI from the first push. Without this the
   branch runs ungated.
2. **Coverage gate** (`--cov-fail-under=85`) — unchanged command; new modules
   (`features/transformers.py`, `features/recipe.py`) must land with tests to
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

## Non-goals

- No new statistics, no formula changes, no resampling changes.
- No Spark/Pyspark pipeline migration.
- No automatic hyperparameter search over preprocessing (only enabling it via
  `pre__` paths when authored).
- No change to evidence JSON contracts or report surfaces (except none).

## Sequencing & gates

Slices 1→2→3 are strictly sequential (each builds on the previous commit).
4, 5 independent after 3. 6, 7 independent of 3–5 (depend on 1 only) — can
run parallel to 4/5. Detailed contracts authored just-in-time per
`CONTRACT_TEMPLATE.md`, against the just-committed state. Every slice: green
suite (direct exit code), ruff, mypy `src/broadway`, one logical commit,
push `taxi`, then parity check/sync to `main`.
