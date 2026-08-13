# HANDOFF

Status snapshot for the Broadway platform. `dataflow.md` is the living
module→file map; this file is the conceptual explanation. `main` holds the
generic platform only; the demo lives on the `taxi` branch.

## The one-sentence summary

Broadway is a platform for **traceable tabular data science** — a generic core
(`src/broadway/`) driven by typed contracts (Pydantic for config/results,
Pandera for DataFrames, `FeatureSpec` for features), where every artifact
carries lineage and every decision is recorded.

## The contract system

Every boundary is a typed contract — a value is valid (typed) or it fails
loudly. Nothing is a bare dict or a magic number.

| Layer | Tool | Where |
|---|---|---|
| Configuration | Pydantic | `config/schema.py` (`DatasetContract`, `ExperimentConfig`, `FeatureConfig`, step models, …) |
| Analysis intent | Pydantic `AnalysisContract` | `analysis/contracts.py` (mode, goal, row definition, decision moment, available info, leakage notes) |
| Observed facts | Pydantic `DatasetProfile` | `discover/profile.py` |
| Raw DataFrame | Pandera | `contracts/pandera.py::build_raw_schema(contract)` — generated from `DatasetContract` |
| Engineered DataFrame | Pandera | `features/schema.py::build_engineered_schema(specs)` — generated from `FeatureSpec` |
| Feature definitions | `FeatureSpec` | `features/schema.py` |
| Training result | Pydantic `TrainingResult` | `training/contracts.py` |
| Evaluation result | Pydantic `EvaluationResult` | `evaluate/contracts.py` |
| Baseline result | Pydantic `BaselineResult` | `baseline/contracts.py` |
| Lineage | Pydantic (`LineageRecord`, `DecisionRecord`, `RunState`, …) | `lineage/models.py` |

Principles: no hardcoded values, no defaults, no `get(key, default)`. A missing
or wrong value raises at load/validation time.

## What works today

- **The pipeline** — `ds-pipeline` steps (`etl → contracts → baseline →
  features → train → evaluate`) run end to end, with `full` dispatching on
  `AnalysisContract.mode` to `configs/flow/{prediction,hypothesis,causal}.yaml`.
- **Lineage & decisions** — every step writes a `LineageRecord` sidecar; the
  `lineage` command builds a run graph + run state (goal, stage, open/resolved
  decisions, `not_yet_run`, `ran_but_output_missing`).
- **`init` onboarding** — a CSV becomes `DatasetContract` + `AnalysisContract` +
  `ExperimentConfig` (interactive or via flags), wired straight into `full`.
- **Hardening** — loud failure on non-finite metrics/objectives, zero-variance
  groups, unexplained row loss, and malformed artifacts; `require_mode` mode
  enforcement.
- **Generic feature path** — contract-driven numeric passthrough + datetime
  decomposition + categorical encoding, extensible via a `builder_module` hook.
- **Non-taxi E2E proof** — the `tests/test_onboarding_e2e.py` acceptance test
  onboards a synthetic dataset and runs the flow, independent of any demo data.

## What's next

- deeper EDA / statistical recommendation flows
- richer generic feature engineering
- project extension ergonomics (builder_module, project config)
- more non-taxi dogfooding
- open-source packaging / docs / contribution polish

## Where the demo lives

A working end-to-end example (real NYC taxi data with slices, decisions,
cleaning, and lineage) lives on the `taxi` branch. `main` intentionally holds
only the generic platform.

## Not built yet (stubs)

- `trust/`, `monitoring/`, `selection/`, `unsupervised/` — docstring stubs.
- `inference/api.py` — stub; the model wrapper is defined but not wired into a
  serving contract yet.

## How to run

```bash
uv sync                                   # install deps
uv run pytest                             # 194 tests

# onboard a CSV, then run the full flow + inspect lineage
uv run ds-pipeline init my_data.csv --name my_data --target price --task regression \
    --mode prediction --goal "predict price" --row-definition "one listing" \
    --decision-moment "at listing time" --success-criterion "beat mean baseline"
uv run ds-pipeline full --dataset my_data --analysis my_data
uv run ds-pipeline lineage --analysis my_data --dataset my_data
```

## Conventions (enforced)

1. No hardcoded values — config YAML / `schema.py` / env var only.
2. Shared functions live in one place and are imported, never duplicated.
3. The agent making a change updates `dataflow.md` in the same commit.
