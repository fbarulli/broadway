# dataflow

Architecture map for the Broadway platform. LLM-friendly: read top-to-bottom,
use the tables to locate code. `main` holds the generic platform only — the
demo/dataset-specific example lives on the `taxi` branch.

## Lifecycle

One coherent flow, from dataset contract to evaluation, with a lineage sidecar
annotating every transition:

```
DatasetContract → DatasetProfile → AnalysisContract → BaselineResult
  → FeatureSpec → TrainingResult → EvaluationResult
```

Each step writes a `LineageRecord` sidecar after saving its artifact; the
`lineage` command assembles them into a graph. `ArtifactTrace` (commit, dataset,
analysis goal) stamps the baseline, and `TransformAudit` (rows in/out, dropped,
columns added/removed) annotates the transforming steps (`etl`, `features`).

`full` is a thin dispatcher: it reads `AnalysisContract.mode` and resolves one
of `configs/flow/{prediction,hypothesis,causal}.yaml`.

## Directory tree

```
broadway/
  src/broadway/
    config/schema.py        # Pydantic models (DatasetContract, ExperimentConfig, FeatureConfig, step models, ...)
    config/loader.py        # load_config, resolve_full_steps, STEP_MODELS
    contracts/              # contract-generated schema + role selectors + checks
      pandera.py            # build_raw_schema(contract) -> pa.DataFrameSchema (generated)
      selectors.py          # feature/datetime/target column selectors over DatasetContract
      checks.py, module.py  # contracts step: columns / dtypes / nulls vs contract
    data/                   # loader, cleaner, splitter, download, db
    discover/               # read CSV/parquet → infer contract + observed profile
      module.py             # discover + profile commands; writes configs/dataset/<name>.yaml + profile.json
      profile.py            # DatasetProfile / ColumnProfile (observed facts; identifier_score is descriptive only)
    onboard/                # interactive init scaffolder (CSV → contracts/config)
      module.py             # init(): infer hints → build dataset/analysis/experiment configs
      infer.py, models.py   # InferenceReport / ColumnHint
    etl/                    # load → clean → split → parquet (+ TransformAudit)
    eda/                    # missingness, quality, summary, visualization, HTML report
    features/               # generic feature machinery + config-driven step
      schema.py             # FeatureSpec, build_engineered_schema
      generic.py            # build_generic_feature_specs (numeric passthrough + derived + encodings)
      builders.py           # datetime_hour/dayofweek/month + builder_module hook
      encodings.py, ml_encodings.py, frequency.py   # target/frequency encodings
      pipeline.py, module.py   # FeaturePipeline + ds-pipeline features step
    stats/                  # pandas/numpy stats library (ANOVA, Welch, Kruskal, assumptions, post-hoc, regression)
      guards.py             # validate_groups — fail loudly on empty/non-finite/zero-variance groups
      module.py             # stats step (hypothesis mode)
    causal/                 # experiment design + power analysis (statsmodels/scipy)
      contracts.py          # ExperimentDesign, ExperimentResult
      design.py, analysis.py, multiple.py, assignment.py, module.py
    baseline/               # guidance baseline, dispatched on AnalysisContract.mode
      contracts.py          # BaselineResult (Pydantic) + save/load
      prediction.py, hypothesis.py, causal.py   # per-mode naive references
      improvement.py        # improvement_vs_baseline
      module.py             # baseline step: dispatch on mode, persist BaselineResult + ArtifactTrace
    training/               # model training + HPO + MLflow tracking
      contracts.py          # TrainingResult
      trainer.py, optuna.py, mlflow_utils.py, models/, module.py
    evaluate/               # evaluation + promotion decision
      contracts.py          # EvaluationResult, ModelComparison, BaselineComparison
      metrics.py            # compute_metrics (mae/rmse/r2; rejects non-finite)
      comparison.py, validation.py, promotion.py, module.py
    lineage/                # decision + lineage graph (sidecars, Mermaid, run state)
      models.py             # DatasetRef, DatasetSlice, DecisionRecord, LineageRecord, LineageNode/Edge, LineageGraph, RunState, TransformAudit
      ids.py, records.py, graph.py, mermaid.py, state.py, module.py
    analysis/contracts.py   # AnalysisContract, AnalysisMode, HypothesisConfig, require_mode
    trace.py                # ArtifactTrace (created_at, commit, dataset, analysis_goal)
    cli.py                  # ds-pipeline argument parsing + dispatch
    pipeline.py             # step orchestrator (imports step module, calls run(cfg))
    utils.py                # feature_columns helper
    inference/, monitoring/, selection/, trust/, unsupervised/   # docstring stubs (not wired)
  configs/
    dataset/<name>.yaml     # per-dataset schema (columns, dtypes, roles, target, task)
    analysis/<name>.yaml    # authored analytical intent (AnalysisContract)
    experiment/<name>.yaml  # features, model, split, metric, hpo
    environment/<name>.yaml # development / staging / production
    step/<step>.yaml        # per-step knobs (etl, contracts, eda, features, stats, causal, train, evaluate, baseline, full)
    flow/<mode>.yaml        # mode-specific step lists (prediction / hypothesis / causal)
  tests/                    # pytest suite (library + CLI + onboarding E2E)
```

## `ds-pipeline` commands

Every step except `discover`/`profile`/`lineage`/`init` takes the same flags.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--dataset <name>` | no | none | load `configs/dataset/<name>.yaml` |
| `--experiment <name>` | no | none | load `configs/experiment/<name>.yaml` |
| `--analysis <name>` | no | none | load `configs/analysis/<name>.yaml` |
| `--environment <name>` | no | `development` | load `configs/environment/{development,staging,production}.yaml` |

| Command | Produces |
|---|---|
| `ds-pipeline init <csv> --name … [--target --task --mode --goal --row-definition --decision-moment --success-criterion …]` | `configs/{dataset,analysis,experiment}/<name>.yaml` + profile |
| `ds-pipeline discover --csv … --target … --task … [--datetime-column --ignore-columns]` | `configs/dataset/<name>.yaml` + `artifacts/discover/profile.json` |
| `ds-pipeline profile --dataset <d>` | observed `DatasetProfile` → `artifacts/discover/profile.json` |
| `ds-pipeline etl --dataset <d> [--experiment <e>]` | cleaned + split parquet |
| `ds-pipeline contracts --dataset <d>` | pass/fail validation vs contract |
| `ds-pipeline eda …` | `artifacts/reports/eda.html` |
| `ds-pipeline baseline --dataset <d> --analysis <a>` | `BaselineResult` → `artifacts/baseline/` |
| `ds-pipeline features --dataset <d> --experiment <e>` | fitted feature pipeline + engineered parquet |
| `ds-pipeline stats --dataset <d> --analysis <a>` | `AnalysisPlan` → `artifacts/stats/` |
| `ds-pipeline causal --dataset <d> --analysis <a>` | `ExperimentDesign` → `artifacts/causal/` |
| `ds-pipeline train --dataset <d> --analysis <a>` | `TrainingResult` → MLflow model/artifacts |
| `ds-pipeline evaluate --dataset <d> --analysis <a>` | `EvaluationResult` + promotion decision |
| `ds-pipeline full --dataset <d> --analysis <a>` | dispatches to the mode flow |
| `ds-pipeline lineage --analysis <a> --dataset <d>` | `artifacts/lineage/graph.json` + `graph.md` + run-state summary |

### Mode flows

The three flows share the prefix `discover → etl → contracts → eda → baseline`
and diverge on the mode tail, resolved by `full` from
`configs/step/full.yaml` (`flows: {prediction, hypothesis, causal}`):

| Mode | Tail | Flow file |
|---|---|---|
| `prediction` | `features → train → evaluate` | `configs/flow/prediction.yaml` |
| `hypothesis` | `stats` | `configs/flow/hypothesis.yaml` |
| `causal` | `causal` | `configs/flow/causal.yaml` |

`causal` is a separate analysis mode, run on its own — it is not part of the
prediction flow. `baseline` is guidance (a naive result to beat), not a hard
gate; it dispatches on `AnalysisContract.mode` and is included in every flow.

## Contracts

| Contract | Where |
|---|---|
| `DatasetContract` / `ColumnSchema` / `ColumnRole` | `src/broadway/config/schema.py` |
| `DatasetProfile` / `ColumnProfile` | `src/broadway/discover/profile.py` |
| `AnalysisContract` / `AnalysisMode` / `HypothesisConfig` | `src/broadway/analysis/contracts.py` |
| `BaselineResult` | `src/broadway/baseline/contracts.py` |
| `FeatureSpec` | `src/broadway/features/schema.py` |
| `TrainingResult` | `src/broadway/training/contracts.py` |
| `EvaluationResult` / `ModelComparison` / `BaselineComparison` | `src/broadway/evaluate/contracts.py` |
| `ArtifactTrace` | `src/broadway/trace.py` |
| `DecisionRecord` / `LineageRecord` / `LineageNode` / `LineageEdge` / `LineageGraph` / `RunState` / `TransformAudit` | `src/broadway/lineage/models.py` |

`DatasetContract` is the accepted schema (authored/authoritative);
`DatasetProfile` describes observed facts computed at discover time — its
`identifier_score` is purely descriptive (discover logs a recommendation, never
mutates roles).

## Config SSOT

YAML is the single source of truth; it is loaded through Pydantic
(`src/broadway/config/schema.py`) with no defaults and no `get(key, default)`.
A missing or wrong value raises at load/validation time.

## Decision + Lineage

Step modules write a `LineageRecord` sidecar under `artifacts/lineage/records/`
after saving their artifact; the `lineage` command assembles them into
`graph.json` + `graph.md` (Mermaid) and prints a run state:

- node ids are `kind:name` (`src/broadway/lineage/ids.py::node_id`)
- `DatasetSlice` is authored config (`configs/slice/<name>.yaml`);
  `DecisionRecord` is a runtime event (`artifacts/lineage/decisions/<id>.json`)
- run state: `goal`, `stage`, `open_decisions`, `resolved_decisions`,
  `not_yet_run`, `ran_but_output_missing`
- `not_yet_run` is derived only from lineage-emitting steps
  (`profile`/`baseline`/`stats`/`causal`/`training`/`evaluation`) for the active
  mode (`src/broadway/lineage/state.py::LINEAGE_STEPS`)
- `ran_but_output_missing` flags a sidecar whose artifact file is absent — an
  integrity error, not "not yet produced"

## Mode enforcement

`AnalysisContract.mode` determines which steps are valid: `stats` requires
`hypothesis`, `causal` requires `causal`, `train`/`evaluate` require
`prediction`. Mismatches fail early via
`src/broadway/analysis/contracts.py::require_mode`. `train`/`evaluate` read the
persisted `BaselineResult` and report improvement over it
(`src/broadway/baseline/improvement.py::improvement_vs_baseline`).
