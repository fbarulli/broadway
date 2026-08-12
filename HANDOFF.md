# HANDOFF

Status of the broadway ML experimentation platform — what works, how to run it, and what's next. `dataflow.md` is the living architecture map; this file is the "is it working" snapshot.

## What works now

### Agnostic stats library — `src/broadway/stats/`
Pure pandas/numpy. Every test returns an `AnalysisPlan` (not a bare float), and every effect size is computed in pairs (eta² AND omega², Cohen's d AND Hedges' g). Signatures are the contract in `src/broadway/stats/API.md`.

| Module | Functions |
|--------|-----------|
| `effect_size.py` | eta², omega², Cohen's d, Hedges' g, group_imbalance |
| `plan.py` | `AnalysisPlan` dataclass + save/load JSON |
| `anova.py` | `run_anova`, `run_welch`, `run_kruskal` |
| `assumptions.py` | `run_levene`, `check_normality` |
| `post_hoc.py` | `games_howell` (adds d/g per pair) |
| `regression.py` | `fit_ols`, `fit_robust`, `bp_jb` |
| `diagnostics.py` | `bp_test`, `jb_test`, `durbin_watson`, `plot_residuals` |
| `time_series.py` | `durbin_watson_test`, `plot_acf` |
| `baseline.py` | `train_lgbm`, `evaluate` |
| `module.py` | pipeline step: build groups → `run_anova` → `save_plan` |

### Dataset project — `project/`
- `data.py` — dataset-specific loaders. Memory-efficient: column projection, dtype downcast, pyarrow filter pushdown, and **streaming** cache build (`iter_batches`, never holds 8.6M rows). Small groups (Staten Island 84, EWR 77) are kept in full.
- `scripts/01…12` — numbered analysis wrappers, run via `python -m project.scripts.NN_xxx`. They keep the ANOVA → assumptions → post-hoc → OLS → LGBM narrative.
- `STATS.md` — what each script does.

### Modes — `DATA_MODE` env var
| Mode | Sample | Time window | Use |
|------|--------|-------------|-----|
| `dev` (default) | 2000 rows | 1 day | does it run |
| `live` | 200K + small groups in full | 31 days | real results |

```bash
DATA_MODE=dev  uv run python -m project.scripts.04_anova_boroughs
DATA_MODE=live uv run python -m project.scripts.12_lgbm_baseline
```

### Config (SSOT)
`configs/{dataset,experiment,environment,step}/*.yaml` → Pydantic (`src/broadway/config/schema.py`) → `load_config()`. No hardcoded values, no defaults, no `get(key, default)`.

### Tests
82 passing: `uv run pytest`. Library tests are synthetic; data-layer tests use real data (`.head(1000)` or the cache).

## What was deliberately dropped / deferred

- **Spark** — dropped. 8.6M rows is pandas-sized (~200MB downcast). `pyspark` remains an optional extra for genuinely large future datasets.
- **Kafka** — not needed for parallel HPO. Optuna + Postgres (already in `docker-compose.yml`) coordinates distributed trials; the DB is the queue. Revisit only for real-time streaming.
- **`learning/stats/`** — deleted; logic migrated into `src/broadway/stats/` + `project/`.

## Not built yet (stubs remain in `src/broadway/`)

- **HPO integration** — `training/optuna.py::run_study` is implemented but not yet invoked by the training step; distributed trials via Postgres + `k8s/train-job.yaml` (`parallelism: N`) remain deferred.
- **trust/** (drift, leakage, fairness, sensitivity, interpretability, uncertainty)
- **monitoring/**, **selection/**, **unsupervised/** — stubs.
- **K8s/CD** — infra scaffolding exists (`docker/`, `k8s/`, `.github/`); promotion → deploy → serve isn't wired yet.

Training is now wired end-to-end: `train` emits `TrainingResult` and logs the
run + model to MLflow (`src/broadway/training/`); `evaluate` produces
`EvaluationResult` + a promotion decision.

`causal` is a separate analysis mode, not part of `full`
(`configs/step/full.yaml` = discover, etl, contracts, eda, features, stats,
train, evaluate). Run it on its own:
`ds-pipeline causal --dataset <d> --experiment <e>` (experiment design / power
analysis → `ExperimentDesign`).

## Conventions (enforced)

1. No hardcoded values — config YAML / `schema.py` / env var only.
2. Shared functions live in one place and are imported, never duplicated.
3. Agents: work only on assigned files; report (don't change) out-of-scope findings; review only recent changes (`git diff HEAD`).
4. The agent making a change updates `dataflow.md` in the same commit.
