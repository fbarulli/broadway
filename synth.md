## FEEDBACK Synthesis → updated plan

### Gap summary (FEEDBACK.md)

| Severity | Pillar | Gap |
|---|---|---|
| Critical | 3. Experimentation & Causal | Zero modules — no power analysis, experiment design, sequential monitoring, HTE |
| High | 1. Foundations | Missing data not standalone; governance/measurement absent |
| High | 4. Trust & Robustness | Mostly `future`; no leakage detection, no drift, no fairness |
| High | 6. Production | No runtime monitoring module |
| Medium | 2. Modeling | `validate/` name collision with model validation; no CV/calibration checks |
| Low | 5. Decisioning | Only `should_promote` — no optimization, no cost-benefit |

### Resolutions

#### 1. Rename `validate/` → `contracts/`

| Before | After | Reason |
|---|---|---|
| `src/validate/module.py` | `src/contracts/module.py` | Data contracts only |
| `src/validate/` | `src/contracts/checks.py` | Null, range, cardinality validations |

Model validation (holdout, CV, calibration) goes into `src/evaluate/validation.py` — it's an evaluation concern, not a data contract concern.

#### 2. Fill Experimentation & Causal gap (new module)

```
src/experiments/
├── module.py          # run(cfg) orchestrator
├── design.py           # Power analysis, sample size, MDE
├── assignment.py       # Randomization, stratification, blocking
├── analysis.py         # T-test, ANOVA, chi-square for experiments
├── multiple.py         # Multiple testing corrections (Bonferroni, BH, FWER)
├── sequential.py       # Sequential monitoring, early stopping rules
└── hte.py              # Heterogeneous treatment effects, uplift
```

This is NOT the same as `stats/`. `stats/` answers "does X relate to Y in observational data?" `experiments/` answers "did changing X *cause* Y, and for whom?"

#### 3. Missing data → own concern

```
src/eda/
├── missing.py          # MCAR/MAR/MNAR tests, Little's MCAR, pattern analysis
├── imputation.py       # Mean, median, mode, KNN, MICE, iterative
```

Already had `eda/quality.py` — split it: `quality.py` → outliers, duplicates, constant cols. `missing.py` → missingness analysis, imputation strategies.

#### 4. Drift detection + leakage

```
src/trust/
├── module.py           # (future) run(cfg) orchestrator
├── drift.py            # PSI, KL divergence, KS test — train vs serve distribution
├── leakage.py          # Target leakage detection (correlation with timestamps, ID-based)
├── sensitivity.py      # Robustness checks, perturbation analysis
└── fairness.py         # Subgroup disparity, equal opportunity, demographic parity
```

`trust/` replaces the scattered `future` markers under interpretability and uncertainty. Those move here too as `trust/interpretability.py` and `trust/uncertainty.py`.

#### 5. Runtime monitoring

```
src/monitoring/
├── module.py           # (future) run(cfg)
├── drift_alert.py      # Scheduled drift checks, alert thresholds
├── performance.py      # Prediction accuracy decay over time
├── latency.py          # Inference latency, throughput
└── data_quality.py     # Production data quality regression vs. training
```

Separate from `contracts/` (which validates *offline*). `monitoring/` is runtime/production.

#### 6. Missing data handling → explicit in pipeline

`data/cleaner.py` will NOT silently drop NA. It will:
1. Log counts per column to `artifacts/reports/eda.html`
2. Apply imputation strategy from experiment config
3. Fall back to dropna only if config explicitly says `strategy: drop`

No silent data loss.

### Updated module map

```
src/
├── config/                     # Pydantic models + loader + resolver
├── pipeline.py                 # Orchestrator
├── cli.py                      # CLI entry
│
├── discover/                   # Auto-generate dataset YAML
├── etl/                        # Raw → processed
├── contracts/                  # was validate/ — data schema enforcement
│   ├── module.py
│   └── checks.py               # nulls, ranges, cardinality
│
├── eda/                        # Distributions, correlations, missingness
│   ├── module.py
│   ├── summary.py
│   ├── visualize.py
│   ├── quality.py              # outliers, duplicates, constant cols
│   ├── missing.py              # NEW — MCAR tests, pattern analysis
│   ├── imputation.py           # NEW — strategy-based imputation
│   ├── compare.py              # before/after ETL comparison
│   └── report.py               # HTML report
│
├── features/                   # Builders, encodings, pipeline
│
├── stats/                      # ANOVA, assumptions, post-hoc, regression, baseline
│
├── experiments/                # NEW — experiment design, power, analysis, HTE
│   ├── module.py
│   ├── design.py
│   ├── assignment.py
│   ├── analysis.py
│   ├── multiple.py
│   ├── sequential.py
│   └── hte.py
│
├── training/                   # Model training, Optuna, MLflow
│   └── models/
│
├── evaluate/                   # Metrics, comparison, promotion, model validation
│   ├── module.py
│   ├── metrics.py
│   ├── comparison.py
│   ├── promotion.py
│   └── validation.py           # NEW — holdout, CV, calibration, residual checks
│
├── trust/                      # NEW — drift, leakage, fairness, sensitivity
│   ├── module.py               # (future)
│   ├── drift.py
│   ├── leakage.py
│   ├── sensitivity.py
│   ├── fairness.py
│   ├── interpretability.py     # was interpretability/
│   └── uncertainty.py          # was uncertainty/
│
├── monitoring/                 # NEW — runtime monitoring
│   ├── module.py               # (future)
│   ├── drift_alert.py
│   ├── performance.py
│   ├── latency.py
│   └── data_quality.py
│
├── selection/                  # (future) — nested CV, AIC/BIC, learning curves
├── unsupervised/               # (future) — PCA, clustering, anomaly
├── inference/                  # FastAPI API
└── data/                       # Shared: loader, cleaner, splitter, download, db
```

### Updated pipeline steps

```yaml
# configs/step/full.yaml
pipeline:
  steps:
    - discover
    - etl
    - contracts        # was validate
    - eda
    - experiments      # optional, only if experiment config has design section
    - features
    - stats
    - train
    - evaluate
    # future:
    # - trust
    # - selection
    # - unsupervised
```

### Step configs

| Step config | Module | Notes |
|---|---|---|
| `discover.yaml` | `src/discover/module.py` | |
| `etl.yaml` | `src/etl/module.py` | |
| `contracts.yaml` | `src/contracts/module.py` | renamed from validate |
| `eda.yaml` | `src/eda/module.py` | |
| `features.yaml` | `src/features/module.py` | |
| `stats.yaml` | `src/stats/module.py` | |
| `experiments.yaml` | `src/experiments/module.py` | optional |
| `train.yaml` | `src/training/module.py` | |
| `evaluate.yaml` | `src/evaluate/module.py` | includes model validation |
| `full.yaml` | `src/pipeline.py` | orchestrates all |

### Aligned with SENIOR.md

| Pillar | Modules | Status |
|---|---|---|
| 1. Foundations | `discover`, `contracts`, `eda` (inc. missing/imputation), `features` | ✅ now explicit about missing data |
| 2. Modeling & Inference | `training`, `evaluate` (+ validation), `stats`, `seelection`*, `unsupervised`* | ✅ validation moved to `evaluate/` |
| 3. Experimentation & Causal | `experiments/` | ✅ new module fills critical gap |
| 4. Trust & Robustness | `trust/` (drift, leakage, fairness, interpretability, uncertainty) | ✅ consolidated, explicit |
| 5. Decisioning & Impact | `evaluate/promotion.py` | ⚠️ still thin; optimization deferred |
| 6. Production & Communication | `inference`, `monitoring`, `eda/report.py`, MLflow, Docker, k8s, CI | ✅ runtime monitoring added |

### Implementation order (revised)

1. Scaffold: `pyproject.toml`, `.gitignore`, `.env.example`, `Dockerfile`, `docker-compose.yml`
2. `src/config/schema.py` + `loader.py` + `resolver.py`
3. `src/cli.py` + `src/pipeline.py`
4. `src/discover/module.py`
5. `configs/dataset/taxi.yaml`
6. `src/data/{loader,cleaner,splitter,download}.py`
7. `src/etl/module.py`
8. `src/contracts/{module,checks}.py`
9. `src/eda/{module,summary,visualize,quality,missing,imputation,compare,report}.py`
10. `src/features/{builders,pipeline,encodings,module}.py`
11. `src/training/models/*`
12. `src/training/{trainer,mlflow_utils,optuna,module}.py`
13. `src/evaluate/{metrics,comparison,promotion,validation,module}.py`
14. `src/experiments/{design,assignment,analysis,multiple,sequential,hte,module}.py`
15. `src/stats/*`
16. `src/inference/api.py`
17. `src/trust/*` + `src/monitoring/*` + `src/selection/*` + `src/unsupervised/*` (stubs)
18. Infra: `docker/`, `k8s/`, `.github/workflows/`
19. `configs/{experiment,environment,step}/*.yaml`
20. `tests/`
21. Smoke test: `uv run ds-pipeline full`

Nothing is silently dropped. Missing data has its own surface in `eda/`. Model validation is distinct from data contracts. Experimentation has a dedicated module. Monitoring exists as a separate runtime concern.
