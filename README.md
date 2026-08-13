# broadway

Broadway is a platform for **traceable tabular data science** — statistical
analysis and predictive modeling where evidence leads to decisions, and every
result carries lineage. It is not an ML experimentation platform: it is a
generic core (`src/broadway/`) driven by typed contracts, with a pipeline CLI
(`ds-pipeline`) and a run graph you can interrogate after the fact.

## What it is

Broadway turns a raw CSV and a statement of analytical intent into a complete,
auditable run. The intent is authored once (`AnalysisContract`: mode, goal, row
definition, decision moment, available info, leakage notes); the pipeline
executes the matching steps; and every artifact records where it came from. A
working end-to-end example lives on the `taxi` branch (real data, slices,
decisions, cleaning, lineage) — `main` is the platform only.

## What we have

### Contracts

Every boundary is a typed contract — a value is valid or it fails loudly.

| Contract | Role | Where |
|---|---|---|
| `DatasetContract` | accepted raw schema (columns, dtypes, roles, target/task) | `src/broadway/config/schema.py` |
| `DatasetProfile` | observed facts computed from the data (cardinality, nulls, identifier score) | `src/broadway/discover/profile.py` |
| `AnalysisContract` | authored intent: `mode`, goal, row definition, decision moment, available info, leakage notes, success criterion | `src/broadway/analysis/contracts.py` |
| `BaselineResult` | naive reference to beat, per mode | `src/broadway/baseline/contracts.py` |
| `FeatureSpec` | structure-only contract for one engineered feature | `src/broadway/features/schema.py` |

### `ds-pipeline` CLI

`init` onboards a CSV into contracts/config; `discover` proposes a dataset
contract; `profile` records observed facts; `lineage` renders the run graph.
Steps run individually or through `full`, which dispatches on
`AnalysisContract.mode` to a mode flow (`configs/flow/{prediction,hypothesis,causal}.yaml`).

```
etl → contracts → baseline → features → train → evaluate   (prediction)
etl → contracts → baseline → stats                         (hypothesis)
etl → contracts → baseline → causal                        (causal)
```

### Lineage & decisions

`ds-pipeline lineage` builds a graph plus a run state (goal, stage, open/resolved
decisions, `not_yet_run`, `ran_but_output_missing`) from persisted artifacts and
decisions rather than a hand-maintained diagram. `DatasetSlice` +
`DecisionRecord` + `ArtifactTrace` + `TransformAudit` make the
evidence → decision → transformation chain traceable end to end.

### Hardening

Invalid inputs fail loudly instead of producing silent nonsense: non-finite
metrics/objectives, zero-variance groups, unexplained row loss, and malformed
artifacts all raise. `require_mode` enforces that a step only runs under its
declared analysis mode.

### Generic feature path

Contract-driven numeric passthrough + datetime decomposition + categorical
encoding, extensible via a `builder_module` hook that registers custom builders
without touching the platform.

## Quick start

```bash
uv sync                    # install deps (add --extra spark only for genuinely large datasets)

# onboard a CSV → contracts/config (interactive; or pass flags non-interactively)
uv run ds-pipeline init my_data.csv \
    --name my_data --target price --task regression \
    --mode prediction --goal "predict price" \
    --row-definition "one listing" --decision-moment "at listing time" \
    --success-criterion "beat mean baseline"

# run the full mode flow
uv run ds-pipeline full --dataset my_data --analysis my_data

# inspect the run graph + run state
uv run ds-pipeline lineage --analysis my_data --dataset my_data

uv run pytest               # 194 tests
```

## What's left

- deeper EDA / statistical recommendation flows
- richer generic feature engineering
- project extension ergonomics (builder_module, project config)
- more non-taxi dogfooding
- open-source packaging / docs / contribution polish

## Demo branch

A working end-to-end example lives on the `taxi` branch — real NYC taxi data
with slices, decisions, cleaning, and lineage. `main` intentionally holds only
the generic platform.

## Where everything lives

| Concern | Location |
|---|---|
| Architecture map | `dataflow.md` |
| Status / what works | `HANDOFF.md` |
| Pipeline CLI + dispatcher | `src/broadway/cli.py`, `src/broadway/pipeline.py` |
| Config schema + loader | `src/broadway/config/` |
| Contracts (raw + selectors) | `src/broadway/contracts/` |
| Discover / profile | `src/broadway/discover/` |
| Onboarding (`init`) | `src/broadway/onboard/` |
| Features (generic + encodings) | `src/broadway/features/` |
| Stats library | `src/broadway/stats/` |
| Causal design | `src/broadway/causal/` |
| Training / evaluate / baseline | `src/broadway/training/`, `src/broadway/evaluate/`, `src/broadway/baseline/` |
| Lineage + decisions | `src/broadway/lineage/`, `src/broadway/trace.py` |
| Config SSOT | `configs/` |
| Tests | `tests/` |
