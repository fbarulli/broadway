# project.md — the live structure map

This is a map of what exists at HEAD, not an aspiration. Anything not listed
here is either untracked scratch or lives in `agents/` (ledger/contracts).

## 1. Platform (`src/broadway/`)

```
src/broadway/
├── cli.py                     # argparse entry: step → config → module.run()
├── pipeline.py                # flow orchestrator (configs/flow/*.yaml)
├── config/                    # pydantic schema + layered YAML loader
├── data/                      # loaders, cleaner, splitter, join/lookup audits
├── etl/module.py              # ingest → canonical → validate → split  (+ etl decision)
├── contracts/                 # pandera raw/joined/engineered schemas
├── cleaning/                  # structural coercion audit
├── eda/                       # profile, quality, outliers, correlation, figures,
│                              #   tests, correction, artifacts + per-feature decisions
├── stats/                     # agnostic stats library (API.md owns the contract)
│   ├── time_series.py         #   ACF/PACF/CCF/ADF, forecasting metrics
│   ├── forecast.py            #   SARIMAX / AutoARIMA wrappers
│   └── ...                    #   anova, assumptions, diagnostics, effect sizes, …
├── features/
│   ├── transformers.py        # the single preprocessing owner: FrequencyEncoding,
│   │                          #   imputer, scaler, winsorizer, target transforms
│   ├── recipe.py              # composes sklearn Pipelines from the recipe steps
│   ├── pipeline.py            # ColumnPlan executor (derive + frequency encode)
│   ├── plan.py / plans.py     # code-owned keep-sets (single source per experiment)
│   └── module.py              # fit/transform persistence (+ features decision)
├── baseline/
│   ├── prediction.py          # naive mean/median baseline
│   ├── hypothesis.py, causal.py
│   └── module.py              # persists the baseline (+ mean-vs-median decision)
├── training/
│   ├── module.py              # train → optional HPO → MLflow (+ HPO decision)
│   ├── hpo.py                 # two-round bandit HPO (canonical study runner)
│   ├── learning_curve.py
│   └── models/                # registry: linear, RF, XGB, LGBM, lgbm_quantile
├── evaluate/
│   ├── metrics.py             # metric vocabulary + MAPE SSOT (fraction × 100)
│   ├── validation.py          # holdout scoring, CV (expanding/rolling)
│   ├── leaderboard.py         # per-metric ranked figures + ChampionDecision
│   ├── promotion.py           # point-estimate helper
│   ├── feature_selection.py   # BH-gated selection
│   ├── explain.py             # SHAP/permutation explanations
│   └── module.py              # holdout eval, intervals, interval promotion
│                              #   (+ promotion_decision.json)
├── causal/
│   ├── design.py              # power / MDE / sample size
│   ├── analysis.py, hte.py, multiple.py, sequential.py
│   └── module.py              # design persistence (+ achieved-power decision)
├── tracking/                  # manifest, baseline stats, data card, bundle
├── calibration/               # split-conformal + quantile coverage metrics
├── lineage/                   # decision records, artifact graph, Mermaid
├── reports/
│   ├── decision.py            # DecisionRuleSet, DecisionRule, Decision, Evidence
│   ├── measurement.py         # MeasurementEstimator: value+interval+null at ONE n
│   ├── step_output.py         # StepOutput: decision + figure + tables + JSON
│   ├── decision_log.py        # MLflow projection (view; the JSON is the truth)
│   ├── experiments_dashboard.py
│   └── markdown.py, results.py, timeline.py, index.py, audit.py, registry.py
├── onboarding/  discover/  selection/  unsupervised/  samples/
└── inference/api.py           # FastAPI surface (no serving bundle is produced)
```

`trust/`, `monitoring/`, `notebooks/`, `scripts/ds_pipeline.py` and
`reports/model_evaluation.html` do **not** exist; they were removed or never
shipped. Do not write against them.

## 2. Project (the taxi development layer)

```
project/
├── config/                    # the project config tree (overlay-selected)
│   ├── dataset/  analysis/  slice/
│   ├── sample/                # named-sample definitions (immutable artifacts)
│   ├── experiments/           # fare_prediction.yaml, timeseries.yaml, mlflow.yaml, …
│   └── layout.yaml            # every project path, resolved once
├── features/__init__.py       # the taxi ColumnPlan (raw ids + route_frequency)
├── ml_pipeline.py             # the dataset-specific FeaturePipeline (golden-pinned)
├── experiments/
│   ├── fare_prediction/       # 01…08, each publishing <stem>_decision.json
│   ├── timeseries/            # 01…04, same contract
│   ├── more_modeling/  multivariate/  univariate/  mlflow/  …
│   └── results/               # committed figures, reports, tables, decisions
├── scripts/                   # numbered teaching scripts (record-free by D32)
└── data.py                    # project constants (config-driven)
```

## 3. GPU lane (`gpu_training/`)

```
gpu_training/
├── memory_probe.py            # measured micro-batch / concurrency baseline
├── run_hpo.py                 # fan-out launcher (dry-run by default)
├── hpo_kernel.py              # one Optuna worker → broadway.training.hpo
├── hpo_study.py               # study config (search space, data, storage)
├── lane_contract.py           # module:function boundary resolution
├── adapters/                  # gluonts_step, prophet_step, timesfm_step, taxi, cpu_smoke
└── config/                    # memory_probe.yaml, hpo_kernel.yaml, hpo_run*.yaml
```

The lane is config-driven: the kernel carries no vendor code, and the study
targets the same `project/config/experiments/mlflow.yaml` HPO spec as the
in-process bandit.

## 4. Artifacts a run produces

| Artifact | Path | Contract |
|---|---|---|
| Step report | `project/experiments/results/<series>/<stem>.md` | `agents/contracts/EXPERIMENT_OUTPUT.md` |
| Step figure | `…/<stem>.png` | one figure answering the step question |
| Evidence tables | `…/<stem>_<name>.csv` | aggregates only |
| Decision record | `…/<stem>_decision.json` | `reports/decision.py`, never hand-written |
| Pipeline decisions | `artifacts/{etl,features,training,baseline,evaluation}/…` | `StepOutput` + figure |
| Promotion record | `artifacts/evaluation/promotion_decision.json` | interval vs threshold |
| Champion record | `…/<stem>_champion.json` | best run + 95% interval |

`MeasurementEstimator` is the one measurement owner: value, interval and null
are computed on ONE seeded draw, block-resampled by day, so a published `n`
always matches the numbers beside it.
