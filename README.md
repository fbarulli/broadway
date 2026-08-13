# broadway

Generalized ML experimentation platform. Two surfaces: a pipeline CLI
(`ds-pipeline`) and a set of numbered analysis scripts for the taxi dataset
(`project/scripts/`). Full architecture map in `dataflow.md`; status
snapshot in `HANDOFF.md`.

## Install

```bash
uv sync                    # install deps (add --extra spark only for genuinely large datasets)
docker compose up -d       # mlflow + postgres (optional; training logs runs + artifacts here)
```

## Quick start (taxi)

```bash
# 1. build the mode-keyed sample cache (streams 8.6M rows, keeps small groups in full)
DATA_MODE=dev uv run python -c "from project import data; data.generate_sample_cache()"

# 2. run an analysis script
DATA_MODE=dev uv run python -m project.scripts.04_anova_boroughs
```

`dev` mode is the default (small sample, fast). Prefix any command with
`DATA_MODE=live` for full-size results.

---

## Lifecycle

One coherent flow, from dataset contract to champion model:

```
DatasetContract → FeatureSpec → TrainingConfig → Optuna → TrainingResult
  → MLflow model/artifacts → EvaluationResult → promotion decision
  → champion model → prediction
```

| Stage | What it is | Feeds |
|-------|------------|-------|
| `AnalysisContract` | authored intent (`configs/analysis/<name>.yaml`) | config |
| `DatasetContract` | raw schema + target/task (`configs/dataset/<name>.yaml`) | etl, features, stats |
| `FeatureSpec` | engineered schema + fitted pipeline | train |
| `TrainingConfig` | model type + params (`configs/experiment/<name>.yaml`) | optuna, train |
| `Optuna` | HPO → best params | train |
| `TrainingResult` | trained model + params + artifact path | MLflow |
| `MLflow model/artifacts` | logged run + model | evaluate |
| `EvaluationResult` | holdout metrics | promotion decision |
| promotion decision | candidate vs champion verdict | champion model |
| champion model | promoted artifact | prediction |

---

## 1. Pipeline CLI — `ds-pipeline`

Every step except `discover` takes the same three flags.

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--dataset <name>` | no | none | load `configs/dataset/<name>.yaml` |
| `--experiment <name>` | no | none | load `configs/experiment/<name>.yaml` |
| `--analysis <name>` | no | none | load `configs/analysis/<name>.yaml` |
| `--environment <name>` | no | `development` | load `configs/environment/{development,staging,production}.yaml` |

`discover` has its own flags:

| Flag | Required | Meaning |
|------|----------|---------|
| `--csv <path>` | yes | raw CSV/parquet to infer schema from |
| `--target <col>` | yes | target column name |
| `--task <task>` | yes | `regression` or `classification` |
| `--datetime-column <col>` | no | datetime column name |
| `--ignore-columns <col>...` | no | columns to mark as ignored (nargs `*`) |

### Steps

| Step | Command | Produces | Status |
|------|---------|----------|--------|
| discover | `ds-pipeline discover --csv … --target … --task …` | `configs/dataset/<name>.yaml` + `artifacts/discover/profile.json` | works |
| etl | `ds-pipeline etl --dataset <d> --experiment <e>` | cleaned + split parquet | works |
| contracts | `ds-pipeline contracts …` | pass/fail validation | works |
| eda | `ds-pipeline eda …` | `artifacts/reports/eda.html` | works |
| features | `ds-pipeline features …` | fitted feature pipeline | works |
| stats | `ds-pipeline stats …` | `AnalysisPlan` JSON | works (uses stats library) |
| causal | `ds-pipeline causal --dataset <d> --experiment <e>` | `ExperimentDesign` (power analysis) | separate mode (not in `full`) |
| baseline | `ds-pipeline baseline --dataset <d> --analysis <a>` | `BaselineResult` → `artifacts/baseline/` | works |
| train | `ds-pipeline train …` | `TrainingResult` → MLflow model/artifacts | works |
| evaluate | `ds-pipeline evaluate …` | `EvaluationResult` + promotion decision | works |
| full | `ds-pipeline full …` | all steps in `configs/step/full.yaml` | works |

`causal` is a separate analysis mode, run on its own — it is not part of
`full`. `configs/step/full.yaml` runs discover, etl, contracts, eda, features,
stats, train, evaluate.

`baseline` is guidance (a naive result to beat), not a hard gate — it is not
part of `full`.

---

## 2. Stats scripts — `project/scripts/`

Numbered narrative: ANOVA → assumptions → post-hoc → OLS diagnostics →
remediation → non-linear baseline. Each is a thin wrapper over
`broadway.stats` (agnostic library) + `project/data` (dataset loaders).

Run via module form (no `sys.path` hacks needed):

```bash
uv run python -m project.scripts.NN_name
```

Build the cache first (needed by scripts 04-12):

```bash
uv run python -c "from project import data; data.generate_sample_cache()"
```

| # | Module | What it does |
|---|--------|--------------|
| 01 | `01_load_data` | inspect schema, row count, sample rows |
| 02 | `02_join_boroughs` | join zone lookup, write `results/quality_report.json` |
| 04 | `04_anova_boroughs` | one-way ANOVA: F, p, eta²/omega² |
| 05 | `05_anova_assumptions` | Levene's test + skew/kurtosis/Shapiro |
| 06 | `06_anova_comparison` | standard vs log vs Welch's vs Kruskal-Wallis |
| 07 | `07_games_howell` | Games-Howell post-hoc + Cohen's d/Hedges' g per pair |
| 08 | `08_ols_residuals_diagnostics` | BP/JB/DW + residual plots |
| 09 | `09_log_target_ols` | log-target OLS + HC3 robust SEs |
| 10 | `10_durbin_watson_time` | time-ordered DW + ACF plot |
| 11 | `11_interaction_ols` | distance × borough interaction + nested F-test |
| 12 | `12_lgbm_baseline` | LightGBM baseline, time-based split, tail MAE |

(There is no `03` — it was a superseded IQR experiment, deliberately dropped.)

---

## 3. Mode system — `DATA_MODE`

| Mode | Sample size | Time window | Purpose |
|------|-------------|-------------|---------|
| `dev` (default) | 2000 rows | 1 day | does the pipeline run |
| `live` | 200K + small groups in full | 1 month | real, accurate results |

- Cache files are mode-keyed: `results/joined_sample_{MODE}.parquet`.
- Small groups (Staten Island 84, EWR 77) are always kept in full — never sampled away.
- Two sampling strategies, both mode-aware: `load_stratified_sample()` (random, stratified — scripts 04-09, 11, 12) and `load_time_slice()` (contiguous, time-sorted, filter pushdown — script 10). Never randomly sample the time slice.

```bash
DATA_MODE=dev  uv run python -m project.scripts.08_ols_residuals_diagnostics
DATA_MODE=live uv run python -m project.scripts.12_lgbm_baseline
```

---

## 4. Tests

```bash
uv run pytest              # 82 tests: library (synthetic) + data layer (real .head(1000)/cache)
```

---

## 5. Config (single source of truth)

```
configs/
  dataset/<name>.yaml      # per-dataset schema (columns, dtypes, target, task)
  experiment/<name>.yaml   # features, model, split, metric
  environment/<name>.yaml  # development / staging / production
  step/<step>.yaml         # per-step knobs + stats/train/features SSOT
  analysis/<name>.yaml     # authored analytical intent (--analysis <name>)
```

YAML → Pydantic (`src/broadway/config/schema.py`) → `load_config()`. No
defaults, no `get(key, default)`, no hardcoded values anywhere.

Typed step outputs follow `artifacts/<step>/` and reports follow
`artifacts/reports/`.

---

## 6. Where everything lives

| Concern | Location |
|---------|----------|
| Architecture map | `dataflow.md` |
| Status / what works | `HANDOFF.md` |
| Stats library (agnostic) | `src/broadway/stats/` (+ `API.md` contract) |
| Dataset loaders + constants | `project/data.py` |
| Script index | `project/STATS.md` |
| Config schema | `src/broadway/config/schema.py` |
| Tests | `tests/` |

### Conventions (for agents and humans)

1. No hardcoded values — config YAML / `schema.py` / env var only.
2. Shared functions live in one place and are imported, never duplicated.
3. The agent making a change updates `dataflow.md` in the same commit.
