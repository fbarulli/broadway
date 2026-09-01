Artifact Bundles

Broadway produces three audience-specific bundles: MLE (serving & infrastructure), Data Tracking (lineage & drift), and DS (story & logic).

1. The MLE Bundle (Serving & Infrastructure)

The MLE doesn't care about your AUC or feature importance charts. They need a deterministic, containerizable package that won't crash the API.

Artifact

Format

Purpose

Inference Pipeline

.pkl (or MLflow PyFunc)

The entire bundle: Preprocessor + Feature Engine + Model. Never hand the MLE a raw model that expects raw data.

Input/Output Signature

openapi.json or MLflow Signature

Exact JSON schema of what the API expects (e.g., {"fare": float}) and returns (e.g., {"prob": float}).

Environment Lock

conda.yaml or requirements.txt

Exact dependency tree (with hashes) to build the Docker image.

Performance Profile

latency_report.json

P50, P95, P99 inference latency (in ms) and memory footprint (in MB). The MLE needs this to configure auto-scaling and Kubernetes limits.

Fallback / Default Rules

fallback.json

What the API should return if the model times out or throws an error (e.g., {"default_prediction": 0.5}).

Where it lives: artifacts/serving/

2. The Data Tracking Bundle (Lineage & Drift Baselines)

You cannot monitor production drift if you don't save the "truth" of what the model was trained on. You need a snapshot of the training data's exact state.

Artifact

Format

Purpose

Data Manifest

data_manifest.json

Cryptographic hash of the training parquet, exact row count, min/max timestamps, and schema version.

Drift Baselines

baseline_stats.json

The exact mean, std, and distribution (PSI/CSI bins) of every feature at training time. This is what your production monitoring compares against.

Lineage Graph

graph.json + graph.md

Mermaid diagram and JSON mapping of: Raw Data → Cleaned Data → Features → Model.

Data Card / Fact Sheet

data_card.md

High-level summary: Date range, known biases, exclusions, and missingness rates.

Where it lives: artifacts/tracking/

3. The DS Bundle (The Story & Logic)

This is what you already have, but it needs to be formalized so the business can audit it.

Artifact

Format

Purpose

Executive Summary

reports/index.md

The "so what?" Business metrics, expected profit, and go/no-go recommendation.

Walkthrough Timeline

reports/timeline.md

Step-by-step hypothesis testing (the gates/decisions from your CLI).

Feature Importance

figures/shap_summary.png

Global explanations.

Error Analysis

figures/residuals.png

Where the model fails (e.g., "fails on rides > $100").

Counterfactuals

reports/recourse.md

Examples of DiCE outputs (what a user must change to flip the prediction).

Where it lives: reports/ and artifacts/evaluation/

# broadway

Generalized ML experimentation platform. The reusable surface is the pipeline

CLI (`ds-pipeline`); dataset-specific experiments live under `project/` on the

development line only. Structural mapping uses the codebase knowledge graph
(`graphify`) and the lineage graph (`reports/lineage/graph.json`); current

operational state is in `agents/ledger/STATE.md`.

Branch model lives in `agents/contracts/MAIN_AGENT_CONTRACT.md` §2 — `sklearn` is the only active development line; `taxi` is a fast-forward pass-along; `main` is frozen until declared main-day.

## Install

```bash

bash scripts/uv.sh sync --extra dev        # install deps incl. dev toolchain (ruff/mypy/pytest-cov); add --extra spark only for genuinely large datasets

docker compose up -d       # mlflow + postgres (optional; training logs runs + artifacts here)

bash scripts/uv.sh run mlflow server --backend-store-uri sqlite:///$(pwd)/.mlflow.db --artifacts-destination file://$(pwd)/mlruns   # no-docker local MLflow server (MLflow 3.x; listens on :5000; mlruns/ is gitignored)

rm -f .mlflow.db && rm -rf mlruns   # reset the demo registry when a stale champion skews promote/comparison between identical runs

```

Run the reset one-liner when a stale champion from an earlier session silently

changes promote/comparison behavior between identical runs; it removes only the

regenerable demo registry state (`.mlflow.db` + `mlruns/`) — never

`artifacts/` or `data/`.

End-to-end determinism between identical pipeline runs is enforced by

`scripts/check_e2e_determinism.sh` (whitelist + exit-code bar documented in

`SKLEARN_PIPELINES.md`, "End-to-end verification criteria").

## Quick start (generic demo)

```bash

# 1. infer a contract from generic demo input

bash scripts/uv.sh run ds-pipeline discover --csv demo/demo.csv --target target --task regression

# 2. generate and validate an immutable generic named sample (output is ignored)

bash scripts/uv.sh run python -c 'from broadway.samples import generate_sample, read_named_sample; generate_sample("demo"); read_named_sample("demo")'

```

The named-sample command fails if `data/samples/demo@v1.parquet` already

exists; remove that ignored local artifact or bump the sample version before

regenerating.

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

Declared intent (`AnalysisContract`) gates which steps are valid: stats→hypothesis, causal→causal, train/evaluate→prediction.

Inference: new-path model artifacts are sklearn Pipelines logged with an

explicit signature; they load through MLflow's native pyfunc flavor and

predict on RAW input frames (the pre-preprocessing feature frame), with

MLflow enforcing the logged signature at predict time. Previously logged

bare-model artifacts remain loadable via `ModelPyFunc`

(`src/broadway/training/models/pyfunc_wrapper.py`).

Pipelines are mode-specific; `full` is a dispatcher that reads

`AnalysisContract.mode` and resolves the matching `configs/flow/{prediction,hypothesis,causal}.yaml`.

## Deliverables

| Audience | Deliverable | File types |

|---|---|---|

| DS | The story | `.md`, `.png`, `AnalysisPlan.json` |

| Model | The weights | `model.pkl`, `.onnx` |

| MLE | The serving bundle | `inference_schema.py`, `preprocessor.pkl`, `model_card.json` |

| Data Team | The observability baseline | `reference_profile.json`, `data_contract.yaml` |

The supporting artifact guide is [`read.md`](scratch/read.md): it expands the DS,

serving, and data-tracking bundles without making the README a second source

of truth.

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

| init | `ds-pipeline init <csv> --name <n> …` | `configs/{dataset,analysis,experiment}/<n>.yaml` + `artifacts/discover/profile.json` + profile lineage sidecar | works (interactive or flag-driven) |

| profile | `ds-pipeline profile --dataset <d>` | `artifacts/discover/profile.json` (re-profile observed facts) | works |

| columns | `ds-pipeline columns --csv <path>` | prints `name: dtype` per source column (read-only) | works |

| ingest | `ds-pipeline ingest --dataset <d> --experiment <e>` | `data/processed/training_data.parquet` + `ingest:<d>` lineage record | works (generic ETL; CI-gated; contract-driven) |

| etl | `ds-pipeline etl --dataset <d> --experiment <e>` | cleaned + split parquet + `JoinAudit`/`LookupValueAudit` (`join`/`lookup_value` lineage nodes) | works |

| contracts | `ds-pipeline contracts …` | pass/fail validation | works |

| features | `ds-pipeline features …` | fitted feature pipeline | works |

| stats | `ds-pipeline stats {run,describe} --dataset <d> --analysis <a> --sample <name>` | `AnalysisPlan` JSON + `reports/results/describe.md` + figures | works (requires `--sample`) |

| causal | `ds-pipeline causal --dataset <d> --analysis <a>` | `ExperimentDesign` (power analysis) | separate mode (not in `full`) |

| baseline | `ds-pipeline baseline --dataset <d> --analysis <a>` | `BaselineResult` → `artifacts/baseline/` | works |

| train | `ds-pipeline train --dataset <d> --analysis <a>` | `TrainingResult` → MLflow model/artifacts | works |

| evaluate | `ds-pipeline evaluate --dataset <d> --analysis <a>` | `EvaluationResult` + promotion decision | works |

| full | `ds-pipeline full …` | dispatches to the mode flow (prediction/hypothesis/causal) based on `--analysis` | works |

| lineage | `ds-pipeline lineage --analysis <a> --dataset <d>` | `reports/lineage/graph.json` + `graph.md` + run-state summary | works (reporting, not a pipeline step) |

| report | `ds-pipeline report --analysis <a> --dataset <d>` | `reports/results/index.md` (thin wrapper over the walkthrough results index) | works (errors "run the walkthrough first" if no timeline state) |

| walkthrough | `ds-pipeline walkthrough --analysis <a> --dataset <d> [--sample <s>] [--force]` | hypothesis analysis timeline (`reports/index.md` dashboard + `reports/timeline.md` + `reports/results/` + `reports/figures/`) | works (idempotent resume; stops at decision gates) |

| decide | `ds-pipeline decide --analysis <a> --method <m> --reason "..." [--kind omnibus\|posthoc]` | `AnalysisDecision` (gates the walkthrough) | works (recording, not a pipeline step) |

| audit | `ds-pipeline audit --dataset <d> [--analysis <a>]` | `reports/audit/*.md` (human-readable data readiness) | works (reporting, not a pipeline step) |

`causal` is a separate analysis mode, run on its own — it is not part of

`full`. `full` is a thin dispatcher that reads `AnalysisContract.mode` and

resolves one of `configs/flow/{prediction,hypothesis,causal}.yaml`.

`baseline` is guidance (a naive result to beat), not a hard gate — it is part

of each mode flow's prefix.

`stats` takes a `run`/`describe` subcommand and requires `--sample <name>`; `causal`, `train`, and `evaluate` require `--analysis <name>`; `train`/`evaluate` report improvement over the persisted baseline.

### Decision + Lineage

Broadway generates a run graph from persisted artifacts + decisions rather than

hand-maintaining a diagram:

```bash

ds-pipeline lineage --analysis taxi --dataset taxi

# → reports/lineage/graph.json + graph.md (Mermaid) + run-state summary

```

Each step writes a `LineageRecord` sidecar under `artifacts/lineage/records/`

after saving its result; the `lineage` command assembles them into the chain

`dataset → ingest → join → {etl, lookup_value} → analysis → baseline → … → decision`.

`DatasetSlice`s are authored project config (`project/config/slice/`); `DecisionRecord`s are

runtime events (`artifacts/lineage/decisions/`).

### Results reports — `reports/`

`reports/` is the human-facing product surface: a navigable, git-tracked

hierarchy built from the machine evidence in `artifacts/`.

```bash

ds-pipeline walkthrough --analysis taxi_hypothesis --dataset taxi

# → reports/index.md (dashboard) + reports/timeline.md + reports/results/ + reports/figures/

ds-pipeline audit --dataset taxi [--analysis taxi_hypothesis]

# → reports/audit/{index,profile,transform,join,lookup_values}.md

```

```text

reports/

index.md            # walkthrough progress dashboard (status, progress count, next action, navigation)

timeline.md         # analysis timeline: one status row per step + per-step details

results/            # per-step pages + results index (owned by the walkthrough)

index.md          # step-by-step status table (links to completed step pages)

\<step>.md         # one page per completed step (question / what was run / found / why it matters)

figures/*.png       # charts rendered from per-step FigureRef (path + one-line "How to read" caption)

audit/              # human-readable data readiness (owned by audit; on-demand, typed renderers)

index.md          # data used, status, what changed, enrichment quality, caveats

profile.md        # observed column facts (dtypes, nulls, cardinality, identifiers) + "Profile evidence" (feature Q-Q + distribution grids + per-feature distribution diagnostics table/heatmap)

transform.md      # structural canonicalization: row transitions + parse failures

join.md           # lookup key-matching completeness

lookup\_values.md  # matched-value quality (nulls/sentinels per enrichment column)

lineage/graph.{md,json}  # run graph (owned by lineage; from the lineage command)

```

The `reports/` surface has four owners, one per question the surface answers:

`walkthrough` owns `index.md` (progress dashboard) and `results/` (per-step

pages + results index); `audit` owns `audit/`; `lineage` owns `lineage/`;

`reports/timeline.md` is the analysis timeline. The `audit` command is

on-demand and pure-rendering: it reads the persisted typed evidence

(`StructuralCleanResult`, `JoinAuditReport`, `LookupValueAuditReport`,

`DatasetProfile`), renders Markdown, and never re-runs ingest/etl/stats/profile.

### Timeline / walkthrough

The walkthrough is an analyst-led hypothesis analysis timeline that advances an

eight-step sequence — `describe_groups → normality → variance → decide_omnibus →

omnibus → decide_posthoc → posthoc → conclusion` — authored in

`configs/flow/hypothesis_walkthrough.yaml` (thresholds in

`configs/step/walkthrough.yaml`) and implemented in `src/broadway/timeline/`.

```bash

ds-pipeline walkthrough --analysis <a> --dataset <d> [--sample <s>] [--force]

```

Each run advances the sequence and stops at the next decision gate. The run is

idempotent (existing steps are skipped on resume); `--force` recomputes steps

but never overwrites recorded decisions. Evidence steps run automatically;

decision steps require an analyst decision, recorded via:

```bash

ds-pipeline decide --analysis <a> --method <m> --reason "..." [--kind omnibus|posthoc]

```

which persists an `AnalysisDecision` that gates the walkthrough (omnibus

methods: `welch`/`anova`/`kruskal`; post-hoc: `games_howell`). Step status is a

plain-text vocabulary — `completed`, `completed with note`, `awaiting decision`,

`failed`, `warning` — and report pages are humanized (human step labels, three

significant figures, p-values floored at "< 0.001").

Evidence steps attach figures via `FigureRef` (a `path` relative to `reports/`

plus a one-line "How to read" caption) on `AnalysisStep.figures`. `timeline.md`

embeds them as `![caption](figures/...)`; per-step pages under `reports/results/`

embed the same figures one link-depth deeper as `![caption](../figures/...)`.

Two Q-Q surfaces answer "is this normal?" at different scopes and converge on

small multiples:

- **Features Q-Q** (`src/broadway/discover/qq.py`, run by `discover`/`profile`)

uses **small multiples** — one subplot per numeric feature (per-feature

z-score) plus a matching per-feature **distribution (histogram) grid in raw

units** — because 7+ features don't read overlaid. Non-finite and zero-variance

features are recorded, not plotted; the grid chunks beyond 12 features per

figure. Config-driven **diagnostic zones** (`qq_zones` in

`configs/step/viz.yaml`) shade the left/right tails (beyond a z-score

threshold) and the central quantile band as read-only visual references, and

draw a dashed "zero-mass shelf" on features where a notable fraction of the

plotted sample is exactly zero.

- **Groups Q-Q** (`src/broadway/timeline/runners.py::run_normality`) uses

**small multiples**, one subplot per group, per-group z-score (capped at 12 groups).

The `audit` profile page renders both feature grids in a "Profile evidence"

section on `reports/audit/profile.md`, from the `QqOverview` record

(`artifacts/discover/qq_overview.json`), with how-to-read lines and

standardization notes (Q-Q = per-feature z-score, distribution = raw units).

Alongside the two grids it renders a **per-feature distribution diagnostics**

surface: a `mean`/`std`/`skew`/`kurtosis`/`zero_rate` table and a single

heatmap (`numeric_diagnostics.png`) whose columns are `[skew, kurtosis,

zero_rate]` z-normalized per column, with the raw value annotated in each cell.

This is a visual reference only — no statistical verdicts or thresholds.

The discover Q-Q/distribution figures downsample the input to a configured

sample size (`qq_sample_size`, 10,000) computed once per figure and show a

single `n = …` in the figure suptitle. Discrete (low-cardinality) distributions

use value-centered bins (midpoint bin edges) so bars center on the observed

unique values rather than an integer range.

Suggestions are de-prescribed: `suggest.py` emits

`ds-pipeline decide --analysis <a> --method <method> --reason "..."` (never a

pre-filled method), and the post-hoc gate adds `--kind posthoc`.

### Git-track policy

- Tracked: `reports/**` (index.md, results/*.md, figures/*.png, lineage/graph.md + graph.json).

- Ignored: `artifacts/`, `data/raw/`, `data/processed/` (machine evidence + caches).

---

## 2. Stats scripts — `project/scripts/`

Numbered narrative: ANOVA → assumptions → post-hoc → OLS diagnostics →

remediation → non-linear baseline. Each is a thin wrapper over

`broadway.stats` (agnostic library) + `project/data` (dataset loaders).

Run via module form (no `sys.path` hacks needed):

```bash

bash scripts/uv.sh run python -m project.scripts.NN_name

```

Build the cache first (needed by scripts 04-12):

```bash

bash scripts/uv.sh run python -c "from project import data; data.generate_sample_cache()"

```

| # | Module | What it does |

|---|--------|--------------|

| 01 | `01_load_data` | inspect schema, row count, sample rows |

| 02 | `02_join_boroughs` | join zone lookup, write `data/processed/quality_report.json` |

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

The OLS diagnostics surface is typed: `DiagnosticResult`

(`src/broadway/stats/diagnostic_models.py`) plus

`plot_residuals_vs_fitted` and `mean_specification_diagnostic`

(`src/broadway/stats/diagnostics.py`) — see `src/broadway/stats/API.md`.

---

## 3. Mode system — `DATA_MODE`

| Mode | Sample size | Time window | Purpose |

|------|-------------|-------------|---------|

| `dev` (default) | 2000 rows | 1 day | does the pipeline run |

| `live` | 200K + small groups in full | 1 month | real, accurate results |

- Cache files are mode-keyed: `data/processed/joined_sample_{MODE}.parquet`.

- Small groups (Staten Island 84, EWR 77) are always kept in full — never sampled away.

- Two sampling strategies, both mode-aware: `load_stratified_sample()` (random, stratified — scripts 04-09, 11, 12) and `load_time_slice()` (contiguous, time-sorted, filter pushdown — script 10). Never randomly sample the time slice.

```bash

DATA_MODE=dev  bash scripts/uv.sh run python -m project.scripts.08_ols_residuals_diagnostics

DATA_MODE=live bash scripts/uv.sh run python -m project.scripts.12_lgbm_baseline

```

---

## 4. Tests

```bash

bash scripts/uv.sh run pytest              # library (synthetic) + data layer (real .head(1000)/cache); no count gate — enforcement is the ≥95% coverage floor on src/broadway via scripts/run_local_ci.sh (D17b SSOT) plus governance probes

```

---

## 5. Config (single source of truth)

```

configs/

dataset/<name>.yaml      # generic/test schema (columns, dtypes, target, task)

experiment/<name>.yaml   # generic feature/model/split defaults

environment/<name>.yaml  # development / staging / production

step/<step>.yaml         # per-step knobs + stats/train/features SSOT

flow/<mode>.yaml         # mode-specific step lists (prediction/hypothesis/causal)

flow/stats_sequence.yaml # ordered stats-step list rendered into reports/index.md

sample/<name>.yaml       # generic SampleSpec for `stats --sample`

analysis/<name>.yaml     # generic analytical intent (--analysis <name>)

project/config/

dataset|analysis|experiment|project|sample|slice/<name>.yaml

                        \# project overlay; selected by project composition

```

`configs/analysis/` holds generic/test analytical use cases. The taxi development

profile lives under `project/config/` and overlays only matching config names.

Run project-bound platform commands through `uv run python -m project.cli` so

the overlay is selected explicitly; `ds-pipeline` remains data-agnostic.

`configs/sample/<name>.yaml` declares versioned named samples —

seed/size/columns/filters/schema generate immutable artifacts from a `.parquet`

or `.csv` source under

`data/samples/` (`<name>@v<N>.parquet` + provenance), validated by

`read_named_sample` before steps consume them by name.

YAML → Pydantic (`src/broadway/config/schema.py`) → `load_config()`. No

defaults, no `get(key, default)`, no hardcoded values anywhere.

`DatasetContract` carries no `row_count` — observed counts live in

`DatasetProfile` (discover) and `TransformAudit` (etl). Datetime dtypes are

normalized to canonical `datetime64` at the schema boundary

(`schema.py::normalize_dtype`).

`lookup_tables` entries support `value_policies` (per-column sentinel values)

and `na_values` (authored NA tokens) — both owned by the config, not inferred

from pandas defaults.

The raw feature schema comes from `configs/dataset/<name>.yaml`, not code:

adding or removing a raw feature means editing that YAML (probe the source

file's dtypes with `ds-pipeline columns --csv <path>`), then re-running

`ds-pipeline ingest --dataset <name> --experiment <name>` + `profile`. No code

change required.

Typed step outputs follow `artifacts/<step>/` and reports follow

`reports/`.

---

## 6. Where everything lives

| Concern | Location |

|---------|----------|

| Architecture map | `graphify` knowledge graph + `reports/lineage/graph.json` |

| Status / current work | `agents/ledger/STATE.md` |

| Stats library (agnostic) | `src/broadway/stats/` (+ `API.md` contract) |

| Decision + lineage graph | `src/broadway/lineage/` (records/graph/mermaid/state) |

| Dataset loaders + constants | `project/data.py` |

| Script index | `project/STATS.md` |

| Config schema | `src/broadway/config/schema.py` |

| HPO / Optuna training + mlflow viewing | `scratch/HPO_TRAINING.md` |

| Tests | `tests/` |

| Branch parity gate | `scripts/check_branch_parity.sh` |

### Conventions (for agents and humans)

1. No hardcoded values — config YAML / `schema.py` / env var only.

2. Shared functions live in one place and are imported, never duplicated.

3. The agent making a change updates structural maps (`graphify` knowledge graph, `reports/lineage/graph.json`) in the same commit when the change alters the structure.

---

## 7. Branch parity — main vs taxi

Branch model and parity workflow live in `agents/contracts/MAIN_AGENT_CONTRACT.md` §§2/12.

| Branch | Role | Contents |

|--------|------|----------|

| `sklearn` | active development line (MAIN_AGENT_CONTRACT.md §2) | platform + the NYC taxi demo (`project/`, scratch docs, generated `reports/`) |

| `taxi` | pass-along mirror of sklearn | fast-forward copy of sklearn |

| `main` | public platform | platform only — generic synthetic demo and `configs/sample/demo.yaml`; no `project/`, taxi config, or committed experiment output |