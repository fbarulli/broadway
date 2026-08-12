# HANDOFF

How the broadway ML experimentation platform works — architecture, lifecycle,
contracts, and how to run it. `dataflow.md` is the living module→file map; this
file is the conceptual explanation.

## The one-sentence summary

Broadway is a **generic ML pipeline core** (`src/broadway/`) driven by **one
dataset** (`project/`), where every boundary is a typed contract: Pydantic for
config and results, Pandera for DataFrames, `FeatureSpec` for features.

## Architecture: three layers

```
project/                 dataset-specific (knows "taxi")
    data.py              loaders, mode system, streaming cache
    features.py          FEATURE_SPECS registry (the engineered feature set)
    ml_pipeline.py       FeaturePipeline (fit/transform) + basic.py/boroughs.py
    scripts/01..12       numbered analysis narrative (ANOVA → OLS → LGBM)

src/broadway/            generic (never knows "taxi")
    config/              Pydantic models + YAML loader (load_config)
    contracts/           selectors + build_raw_schema(contract) → Pandera
    stats/               statistical library (anova, assumptions, post_hoc, …)
    features/            FeatureSpec, build_engineered_schema, encodings
    training/            trainer, optuna, mlflow_utils, models
    evaluate/            metrics, comparison, validation, promotion
    causal/              experiment design + analysis (separate mode)

configs/                 single source of truth (dataset/experiment/environment/step YAML)
```

The litmus test: **if you swapped the dataset YAML tomorrow, would you have to
edit `src/broadway/**`?** The answer should be no; only `project/` and the
config YAML change.

## The contract system

Every boundary carries a typed contract — a value is either valid (typed) or it
fails loudly. Nothing is a bare dict or a magic number.

| Layer | Tool | Where |
|-------|------|-------|
| Configuration | Pydantic | `config/schema.py` (`DatasetContract`, `StatsStep`, `TrainStep`, `ExperimentConfig`, …) |
| Raw DataFrame | Pandera | `contracts/pandera.py::build_raw_schema(contract)` — generated from `DatasetContract` |
| Engineered DataFrame | Pandera | `features/schema.py::build_engineered_schema(specs)` — generated from `FEATURE_SPECS` |
| Feature definitions | `FeatureSpec` | `project/features.py::FEATURE_SPECS` (single source; names/types/schema all derive from it) |
| Analysis result | Pydantic `AnalysisPlan` | `stats/plan.py` |
| Experiment design/result | Pydantic | `causal/contracts.py` (`ExperimentDesign`, `ExperimentResult`) |
| Training result | Pydantic `TrainingResult` | `training/contracts.py` |
| Evaluation result | Pydantic `EvaluationResult` | `evaluate/contracts.py` (+ `ModelComparison`) |

Principles: no hardcoded values, no defaults, no `get(key, default)`. A missing
or wrong value raises at load/validation time, not silently later.

## The lifecycle

```
DatasetContract  →  FeatureSpec  →  TrainingConfig  →  Optuna
      →  TrainingResult  →  MLflow model/artifacts  →  EvaluationResult
      →  promotion decision  →  champion model  →  prediction
```

- **Training** (`training/module.py`): if `ExperimentConfig.hpo` is set, `run_study`
  searches the space (from `HPOConfig.search_space`, validated against the model
  type); else it uses `model.params`. The best model is fit, `params`/`metrics`/
  the artifact are logged to MLflow, and a `TrainingResult` is written.
- **Evaluation** (`evaluate/module.py`): loads the candidate from
  `TrainingResult.artifact_path` via MLflow, loads the champion if one exists
  (`get_champion`), computes `compare_models` + `cross_validate` +
  `residual_summary`, and emits an `EvaluationResult` with the promotion decision.
- **Promotion**: `should_promote` + `promote_candidate` (champion alias in the
  MLflow registry). Champion is compared only when one exists; otherwise the
  candidate promotes unconditionally.

`causal` is **not** part of this flow — it is a separate analysis mode run on
its own (`ds-pipeline causal …`), not in `configs/step/full.yaml`.

## Data flow (the taxi project)

```
data/processed/training_data.parquet  (8.6M rows)
    │  pyarrow ParquetFile.iter_batches  (streaming — never holds all rows)
    ▼
generate_sample_cache()  →  results/joined_sample_{MODE}.parquet
                          →  results/quality_report.json  (exact group sizes/means)
    │
    ▼
load_stratified_sample()   random stratified (08/09/11/12)
load_time_slice()          contiguous time-ordered slice, filter pushdown (10)
    │
    ▼
scripts 01..12  →  AnalysisPlan JSON, residual/ACF plots
```

Two sampling strategies, both correct in every mode:
- **Stratified random** — preserves per-borough proportions, deterministic seed.
- **Contiguous time slice** — never randomly sampled (contiguity + ordering
  matter for Durbin-Watson/ACF). Dev mode shrinks the *window*, not the ordering.

Small groups (Staten Island 84, EWR 77) are always kept **in full** — never
sampled away.

## Modes

`DATA_MODE=dev` (default, 2000 rows / 1-day window) vs `DATA_MODE=live` (200K +
small groups in full / 1-month window). Set via env var, per-call parameter on
the loaders.

## How to run

```bash
uv sync                                   # install deps
uv run pytest                             # 129 tests

# build the sample cache once, then run analysis scripts
DATA_MODE=dev  uv run python -m project.scripts.04_anova_boroughs
DATA_MODE=live uv run python -m project.scripts.12_lgbm_baseline

# the pipeline CLI (causal is a separate mode, not in `full`)
uv run ds-pipeline full --dataset taxi --experiment taxi
uv run ds-pipeline causal --dataset taxi --experiment taxi
```

## What was deliberately dropped

- **Spark** — 8.6M rows is pandas-sized; `pyspark` stays an optional extra.
- **Kafka** — not needed for parallel HPO (Optuna + Postgres is the queue).
- **dtype downcast** — removed; `DatasetContract` dtypes are the single truth.

## Not built yet (stubs)

- **trust/** (drift, leakage, fairness, sensitivity, interpretability, uncertainty)
- **monitoring/**, **selection/**, **unsupervised/** — stubs.
- **inference/api.py** — stub; `pyfunc_wrapper` is defined but not wired (no
  serving contract yet).
- **K8s/CD** — infra scaffolding exists (`docker/`, `k8s/`, `.github/`);
  promotion → deploy → serve isn't wired.

## Conventions (enforced)

1. No hardcoded values — config YAML / `schema.py` / env var only.
2. Shared functions live in one place and are imported, never duplicated.
3. All coding changes are made by agents; agents work only on assigned files,
   report (don't change) out-of-scope findings, and review only recent changes
   (`git diff HEAD`).
4. The agent making a change updates `dataflow.md` in the same commit.
