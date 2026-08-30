Here’s a cleaned-up, solid Markdown version with consistent headings, tables, spacing, and hierarchy.

---

# Artifact Bundles

Broadway produces three audience-specific bundles:

1. **MLE Bundle** — serving and infrastructure
2. **Data Tracking Bundle** — lineage and drift baselines
3. **DS Bundle** — story, logic, and business audit trail

---

## 1. MLE Bundle: Serving & Infrastructure

The MLE does not care about AUC or feature importance charts. They need a deterministic, containerizable package that will not crash the API.

**Location:** `artifacts/serving/`

| Artifact | Format | Purpose |
|---|---|---|
| Inference Pipeline | `.pkl` or MLflow PyFunc | The full inference bundle: preprocessor, feature engine, and model. Never hand the MLE a raw model that expects raw data. |
| Input / Output Signature | `openapi.json` or MLflow Signature | Exact JSON schema of what the API expects and returns. Example input: `{"fare": float}`. Example output: `{"prob": float}`. |
| Environment Lock | `conda.yaml` or `requirements.txt` | Exact dependency tree, preferably with hashes, used to build the Docker image. |
| Performance Profile | `latency_report.json` | P50, P95, and P99 inference latency in milliseconds, plus memory footprint in MB. Used for autoscaling and Kubernetes resource limits. |
| Fallback / Default Rules | `fallback.json` | What the API should return if the model times out or throws an error. Example: `{"default_prediction": 0.5}`. |

---

## 2. Data Tracking Bundle: Lineage & Drift Baselines

You cannot monitor production drift if you do not save the “truth” of what the model was trained on. This bundle captures the exact training-data state.

**Location:** `artifacts/tracking/`

| Artifact | Format | Purpose |
|---|---|---|
| Data Manifest | `data_manifest.json` | Cryptographic hash of the training parquet, exact row count, min/max timestamps, and schema version. |
| Drift Baselines | `baseline_stats.json` | Mean, standard deviation, and distribution bins for every feature at training time. Production monitoring compares against this. |
| Lineage Graph | `graph.json` and `graph.md` | Mermaid diagram and JSON mapping of the chain: Raw Data → Cleaned Data → Features → Model. |
| Data Card / Fact Sheet | `data_card.md` | High-level summary: date range, known biases, exclusions, and missingness rates. |

---

## 3. DS Bundle: The Story & Logic

This is the data science narrative, formalized so the business can audit it.

**Locations:** `reports/` and `artifacts/evaluation/`

| Artifact | Format | Purpose |
|---|---|---|
| Executive Summary | `reports/index.md` | The “so what?”: business metrics, expected profit, and go/no-go recommendation. |
| Walkthrough Timeline | `reports/timeline.md` | Step-by-step hypothesis testing, including CLI gates and decisions. |
| Feature Importance | `figures/shap_summary.png` | Global model explanations. |
| Error Analysis | `figures/residuals.png` | Where the model fails, for example: “fails on rides > $100.” |
| Counterfactuals | `reports/recourse.md` | Examples of DiCE outputs: what a user must change to flip the prediction. |

---

# Broadway

Broadway is a generalized ML experimentation platform. The reusable surface is the pipeline CLI:

```bash
ds-pipeline
```

Dataset-specific experiments live under `project/` on the development line only.

- Full architecture map: `dataflow.md`
- Current operational state: `agents/ledger/STATE.md`
- Branch model: `agents/contracts/MAIN_AGENT_CONTRACT.md` §2

The `sklearn` branch is the only active development line. The `taxi` branch is a fast-forward pass-along mirror. The `main` branch is frozen until declared main-day.

---

## Install

```bash
# Install dependencies, including the dev toolchain:
bash scripts/uv.sh sync --extra dev

# Add this only for genuinely large datasets:
bash scripts/uv.sh sync --extra dev --extra spark

# Optional: start MLflow + Postgres:
docker compose up -d

# No-Docker local MLflow server:
bash scripts/uv.sh run mlflow server \
  --backend-store-uri sqlite:///$(pwd)/.mlflow.db \
  --artifacts-destination file://$(pwd)/mlruns

# Reset the demo registry when a stale champion skews promotion/comparison:
rm -f .mlflow.db && rm -rf mlruns
```

The reset command removes only regenerable demo registry state:

- `.mlflow.db`
- `mlruns/`

It should never remove:

- `artifacts/`
- `data/`

End-to-end determinism between identical pipeline runs is enforced by:

```bash
scripts/check_e2e_determinism.sh
```

The whitelist and exit-code bar are documented in `SKLEARN_PIPELINES.md` under “End-to-end verification criteria.”

---

## Quick Start: Generic Demo

```bash
# 1. Infer a contract from generic demo input:
bash scripts/uv.sh run ds-pipeline discover \
  --csv demo/demo.csv \
  --target target \
  --task regression

# 2. Generate and validate an immutable generic named sample:
bash scripts/uv.sh run python -c '
from broadway.samples import generate_sample, read_named_sample
generate_sample("demo")
read_named_sample("demo")
'
```

The named-sample command fails if this file already exists:

```text
data/samples/demo@v1.parquet
```

Remove that ignored local artifact or bump the sample version before regenerating.

---

# Lifecycle

Broadway follows one coherent flow from dataset contract to champion model:

```text
DatasetContract
  → FeatureSpec
  → TrainingConfig
  → Optuna
  → TrainingResult
  → MLflow model/artifacts
  → EvaluationResult
  → promotion decision
  → champion model
  → prediction
```

| Stage | What It Is | Feeds |
|---|---|---|
| `AnalysisContract` | Authored intent: `configs/analysis/<name>.yaml` | Config |
| `DatasetContract` | Raw schema, target, and task: `configs/dataset/<name>.yaml` | ETL, features, stats |
| `FeatureSpec` | Engineered schema and fitted pipeline | Train |
| `TrainingConfig` | Model type and parameters: `configs/experiment/<name>.yaml` | Optuna, train |
| `Optuna` | Hyperparameter optimization and best params | Train |
| `TrainingResult` | Trained model, params, and artifact path | MLflow |
| MLflow model/artifacts | Logged run and model | Evaluate |
| `EvaluationResult` | Holdout metrics | Promotion decision |
| Promotion decision | Candidate vs. champion verdict | Champion model |
| Champion model | Promoted artifact | Prediction |

Declared intent through `AnalysisContract` gates which steps are valid:

- `stats` → hypothesis
- `causal` → causal
- `train` / `evaluate` → prediction

Inference artifacts are sklearn Pipelines logged with an explicit signature. They load through MLflow’s native PyFunc flavor and predict on raw input frames, meaning the pre-preprocessing feature frame. MLflow enforces the logged signature at prediction time.

Previously logged bare-model artifacts remain loadable through:

```text
src/broadway/training/models/pyfunc_wrapper.py
```

Pipelines are mode-specific. The `full` command is a dispatcher that reads `AnalysisContract.mode` and resolves the matching flow config:

```text
configs/flow/prediction.yaml
configs/flow/hypothesis.yaml
configs/flow/causal.yaml
```

---

# Deliverables

| Audience | Deliverable | File Types |
|---|---|---|
| DS | The story | `.md`, `.png`, `AnalysisPlan.json` |
| Model | The weights | `model.pkl`, `.onnx` |
| MLE | The serving bundle | `inference_schema.py`, `preprocessor.pkl`, `model_card.json` |
| Data Team | The observability baseline | `reference_profile.json`, `data_contract.yaml` |

The supporting artifact guide is `read.md`. It expands the DS, serving, and data-tracking bundles without making the README a second source of truth.

---

# 1. Pipeline CLI: `ds-pipeline`

Every step except `discover` takes the same three core flags.

| Flag | Required | Default | Meaning |
|---|---:|---|---|
| `--dataset <name>` | No | None | Load `configs/dataset/<name>.yaml` |
| `--experiment <name>` | No | None | Load `configs/experiment/<name>.yaml` |
| `--analysis <name>` | No | None | Load `configs/analysis/<name>.yaml` |
| `--environment <name>` | No | `development` | Load `configs/environment/{development,staging,production}.yaml` |

`discover` has its own flags:

| Flag | Required | Meaning |
|---|---:|---|
| `--csv <path>` | Yes | Raw CSV/parquet to infer schema from |
| `--target <col>` | Yes | Target column name |
| `--task <task>` | Yes | `regression` or `classification` |
| `--datetime-column <col>` | No | Datetime column name |
| `--ignore-columns <col>...` | No | Columns to mark as ignored, using `nargs *` |

---

## CLI Steps

| Step | Command | Produces | Status |
|---|---|---|---|
| `discover` | `ds-pipeline discover --csv … --target … --task …` | `configs/dataset/<name>.yaml` and `artifacts/discover/profile.json` | Works |
| `init` | `ds-pipeline init <csv> --name <n> …` | `configs/{dataset,analysis,experiment}/<n>.yaml`, `artifacts/discover/profile.json`, and profile lineage sidecar | Works, interactive or flag-driven |
| `profile` | `ds-pipeline profile --dataset <d>` | `artifacts/discover/profile.json` | Works |
| `columns` | `ds-pipeline columns --csv <path>` | Prints `name: dtype` per source column | Works, read-only |
| `ingest` | `ds-pipeline ingest --dataset <d> --experiment <e>` | `data/processed/training_data.parquet` and `ingest:<d>` lineage record | Works |
| `etl` | `ds-pipeline etl --dataset <d> --experiment <e>` | Cleaned/split parquet and `JoinAudit` / `LookupValueAudit` lineage nodes | Works |
| `contracts` | `ds-pipeline contracts …` | Validation pass/fail | Works |
| `features` | `ds-pipeline features …` | Fitted feature pipeline | Works |
| `stats` | `ds-pipeline stats {run,describe} --dataset <d> --analysis <a> --sample <name>` | `AnalysisPlan` JSON, `reports/results/describe.md`, and figures | Works; requires `--sample` |
| `causal` | `ds-pipeline causal --dataset <d> --analysis <a>` | `ExperimentDesign` and power analysis | Separate mode |
| `baseline` | `ds-pipeline baseline --dataset <d> --analysis <a>` | `BaselineResult` in `artifacts/baseline/` | Works |
| `train` | `ds-pipeline train --dataset <d> --analysis <a>` | `TrainingResult` and MLflow artifacts | Works |
| `evaluate` | `ds-pipeline evaluate --dataset <d> --analysis <a>` | `EvaluationResult` and promotion decision | Works |
| `full` | `ds-pipeline full …` | Dispatches to prediction, hypothesis, or causal flow based on `--analysis` | Works |
| `lineage` | `ds-pipeline lineage --analysis <a> --dataset <d>` | `reports/lineage/graph.json`, `graph.md`, and run-state summary | Works |
| `report` | `ds-pipeline report --analysis <a> --dataset <d>` | `reports/results/index.md` | Works; errors if walkthrough has not run |
| `walkthrough` | `ds-pipeline walkthrough --analysis <a> --dataset <d> [--sample <s>] [--force]` | `reports/index.md`, `reports/timeline.md`, `reports/results/`, and `reports/figures/` | Works |
| `decide` | `ds-pipeline decide --analysis <a> --method <m> --reason "..." [--kind omnibus\|posthoc]` | `AnalysisDecision` | Works; recording step, not pipeline step |
| `audit` | `ds-pipeline audit --dataset <d> [--analysis <a>]` | `reports/audit/*.md` | Works; reporting only |

Notes:

- `causal` is a separate analysis mode and is not part of `full`.
- `full` dispatches by reading `AnalysisContract.mode`.
- `baseline` is guidance, not a hard gate.
- `stats` requires a `run` or `describe` subcommand and a named sample.
- `causal`, `train`, and `evaluate` require `--analysis <name>`.
- `train` and `evaluate` report improvement over the persisted baseline.

---

# Decision & Lineage

Broadway generates a run graph from persisted artifacts and decisions rather than hand-maintaining diagrams.

```bash
ds-pipeline lineage --analysis taxi --dataset taxi
```

Produces:

```text
reports/lineage/graph.json
reports/lineage/graph.md
```

Each step writes a `LineageRecord` sidecar under:

```text
artifacts/lineage/records/
```

The `lineage` command assembles the chain:

```text
dataset
  → ingest
  → join
  → {etl, lookup_value}
  → analysis
  → baseline
  → …
  → decision
```

`DatasetSlice`s are authored project config:

```text
project/config/slice/
```

`DecisionRecord`s are runtime events:

```text
artifacts/lineage/decisions/
```

---

# Results Reports: `reports/`

`reports/` is the human-facing product surface: a navigable, git-tracked hierarchy built from machine evidence in `artifacts/`.

```bash
ds-pipeline walkthrough --analysis taxi_hypothesis --dataset taxi
```

Produces:

```text
reports/index.md
reports/timeline.md
reports/results/
reports/figures/
```

```bash
ds-pipeline audit --dataset taxi --analysis taxi_hypothesis
```

Produces:

```text
reports/audit/index.md
reports/audit/profile.md
reports/audit/transform.md
reports/audit/join.md
reports/audit/lookup_values.md
```

---

## Reports Directory Layout

```text
reports/
├── index.md
├── timeline.md
├── results/
│   ├── index.md
│   └── <step>.md
├── figures/
│   └── *.png
├── audit/
│   ├── index.md
│   ├── profile.md
│   ├── transform.md
│   ├── join.md
│   └── lookup_values.md
└── lineage/
    ├── graph.md
    └── graph.json
```

| Path | Owner | Purpose |
|---|---|---|
| `reports/index.md` | `walkthrough` | Progress dashboard |
| `reports/timeline.md` | `walkthrough` | Analysis timeline |
| `reports/results/` | `walkthrough` | Per-step result pages and index |
| `reports/figures/` | `walkthrough` | Figures referenced by result pages |
| `reports/audit/` | `audit` | Human-readable data readiness reports |
| `reports/lineage/` | `lineage` | Run graph from persisted lineage records |

The `audit` command is on-demand and pure-rendering. It reads persisted typed evidence and never re-runs ingest, ETL, stats, or profile.

Typed evidence includes:

- `StructuralCleanResult`
- `JoinAuditReport`
- `LookupValueAuditReport`
- `DatasetProfile`

---

# Timeline / Walkthrough

The walkthrough is an analyst-led hypothesis analysis timeline. It advances through this eight-step sequence:

```text
describe_groups
  → normality
  → variance
  → decide_omnibus
  → omnibus
  → decide_posthoc
  → posthoc
  → conclusion
```

The sequence is authored in:

```text
configs/flow/hypothesis_walkthrough.yaml
```

Thresholds live in:

```text
configs/step/walkthrough.yaml
```

Implementation lives in:

```text
src/broadway/timeline/
```

Run:

```bash
ds-pipeline walkthrough \
  --analysis <a> \
  --dataset <d> \
  --sample <s>
```

Or force recomputation:

```bash
ds-pipeline walkthrough \
  --analysis <a> \
  --dataset <d> \
  --sample <s> \
  --force
```

Each run advances the sequence and stops at the next decision gate.

The run is idempotent:

- Existing steps are skipped on resume.
- `--force` recomputes steps.
- `--force` never overwrites recorded decisions.

Decision steps require an analyst decision:

```bash
ds-pipeline decide \
  --analysis <a> \
  --method <m> \
  --reason "..."
```

For post-hoc decisions:

```bash
ds-pipeline decide \
  --analysis <a> \
  --method games_howell \
  --kind posthoc \
  --reason "..."
```

Supported omnibus methods:

- `welch`
- `anova`
- `kruskal`

Supported post-hoc method:

- `games_howell`

Step status vocabulary:

- `completed`
- `completed with note`
- `awaiting decision`
- `failed`
- `warning`

Report pages are humanized:

- Human step labels
- Three significant figures
- P-values floored at `< 0.001`

---

## Figures in the Timeline

Evidence steps attach figures via `FigureRef`:

```text
AnalysisStep.figures
```

Each `FigureRef` includes:

- `path` relative to `reports/`
- One-line “How to read” caption

`timeline.md` embeds figures as:

```markdown
![caption](figures/...)
```

Per-step pages under `reports/results/` embed the same figures as:

```markdown
![caption](../figures/...)
```

---

# Q-Q and Distribution Surfaces

Two Q-Q surfaces answer “is this normal?” at different scopes.

## Feature Q-Q

Implementation:

```text
src/broadway/discover/qq.py
```

Run by:

- `discover`
- `profile`

Behavior:

- Uses small multiples.
- One subplot per numeric feature.
- Per-feature z-score.
- Matching distribution histogram grid in raw units.
- Non-finite and zero-variance features are recorded, not plotted.
- Grids chunk beyond 12 features per figure.

Diagnostic zones are config-driven:

```text
qq_zones
```

Defined in:

```text
configs/step/viz.yaml
```

Diagnostic zones:

- Shade left/right tails beyond a z-score threshold.
- Shade central quantile band.
- Draw dashed zero-mass shelf when a notable fraction of plotted samples is exactly zero.

## Group Q-Q

Implementation:

```text
src/broadway/timeline/runners.py::run_normality
```

Behavior:

- Uses small multiples.
- One subplot per group.
- Per-group z-score.
- Capped at 12 groups.

---

## Audit Profile Evidence

The `audit` profile page renders feature grids in:

```text
reports/audit/profile.md
```

It reads from:

```text
artifacts/discover/qq_overview.json
```

The profile evidence includes:

- Feature Q-Q grid
- Feature distribution grid
- How-to-read lines
- Standardization notes

Q-Q values are shown as per-feature z-scores. Distribution values are shown in raw units.

The same page renders a per-feature distribution diagnostics surface:

| Metric |
|---|
| `mean` |
| `std` |
| `skew` |
| `kurtosis` |
| `zero_rate` |

It also renders:

```text
numeric_diagnostics.png
```

The heatmap columns are:

```text
skew
kurtosis
zero_rate
```

Each column is z-normalized, with raw values annotated in each cell.

This is a visual reference only. It does not make statistical verdicts or apply thresholds.

The discover Q-Q and distribution figures downsample to:

```text
qq_sample_size = 10,000
```

The sample size is computed once per figure and shown as a single `n = ...` in the figure title.

Discrete low-cardinality distributions use value-centered bins so bars center on observed unique values.

---

# Suggestions

Suggestions are de-prescribed.

`suggest.py` emits:

```bash
ds-pipeline decide --analysis <a> --method <method> --reason "..."
```

It never emits a pre-filled method.

The post-hoc gate adds:

```bash
--kind posthoc
```

---

# Git Tracking Policy

Tracked:

```text
reports/**
reports/index.md
reports/results/*.md
reports/figures/*.png
reports/lineage/graph.md
reports/lineage/graph.json
```

Ignored:

```text
artifacts/
data/raw/
data/processed/
```

---

# 2. Stats Scripts: `project/scripts/`

The numbered narrative is:

```text
ANOVA
  → assumptions
  → post-hoc
  → OLS diagnostics
  → remediation
  → non-linear baseline
```

Each script is a thin wrapper over:

- `broadway.stats`
- `project.data`

Run scripts via module form:

```bash
bash scripts/uv.sh run python -m project.scripts.NN_name
```

Build the cache first for scripts `04` through `12`:

```bash
bash scripts/uv.sh run python -c "from project import data; data.generate_sample_cache()"
```

| # | Module | Purpose |
|---:|---|---|
| 01 | `01_load_data` | Inspect schema, row count, and sample rows |
| 02 | `02_join_boroughs` | Join zone lookup and write `data/processed/quality_report.json` |
| 04 | `04_anova_boroughs` | One-way ANOVA: F, p, eta², omega² |
| 05 | `05_anova_assumptions` | Levene’s test, skew, kurtosis, and Shapiro |
| 06 | `06_anova_comparison` | Standard ANOVA vs. log ANOVA vs. Welch vs. Kruskal-Wallis |
| 07 | `07_games_howell` | Games-Howell post-hoc and Cohen’s d / Hedges’ g per pair |
| 08 | `08_ols_residuals_diagnostics` | BP, JB, DW, and residual plots |
| 09 | `09_log_target_ols` | Log-target OLS and HC3 robust standard errors |
| 10 | `10_durbin_watson_time` | Time-ordered DW and ACF plot |
| 11 | `11_interaction_ols` | Distance × borough interaction and nested F-test |
| 12 | `12_lgbm_baseline` | LightGBM baseline, time-based split, and tail MAE |

There is no script `03`. It was a superseded IQR experiment and was deliberately dropped.

The OLS diagnostics surface is typed:

```text
DiagnosticResult
```

Defined in:

```text
src/broadway/stats/diagnostic_models.py
```

Related functions:

```text
plot_residuals_vs_fitted
mean_specification_diagnostic
```

Defined in:

```text
src/broadway/stats/diagnostics.py
```

See:

```text
src/broadway/stats/API.md
```

---

# 3. Mode System: `DATA_MODE`

| Mode | Sample Size | Time Window | Purpose |
|---|---:|---|---|
| `dev` | 2,000 rows | 1 day | Validate that the pipeline runs |
| `live` | 200,000+ rows, with small groups in full | 1 month | Produce real, accurate results |

Cache files are mode-keyed:

```text
data/processed/joined_sample_{MODE}.parquet
```

Small groups are always kept in full and never sampled away:

| Group | Count |
|---|---:|
| Staten Island | 84 |
| EWR | 77 |

There are two sampling strategies, both mode-aware:

| Function | Strategy | Used By |
|---|---|---|
| `load_stratified_sample()` | Random, stratified | Scripts `04`–`09`, `11`, `12` |
| `load_time_slice()` | Contiguous, time-sorted, with filter pushdown | Script `10` |

Never randomly sample the time slice.

Examples:

```bash
DATA_MODE=dev \
  bash scripts/uv.sh run python -m project.scripts.08_ols_residuals_diagnostics

DATA_MODE=live \
  bash scripts/uv.sh run python -m project.scripts.12_lgbm_baseline
```

---

# 4. Tests

Run tests:

```bash
bash scripts/uv.sh run pytest
```

Coverage and enforcement:

- Library tests use synthetic data.
- Data-layer tests use real `.head(1000)` or cache.
- There is no count gate.
- Enforcement is the `≥95%` coverage floor on `src/broadway`.
- The local CI entry point is:

```text
scripts/run_local_ci.sh
```

The governance probes are part of the same quality gate.

---

# 5. Config: Single Source of Truth

```text
configs/
├── dataset/
│   └── <name>.yaml
├── experiment/
│   └── <name>.yaml
├── environment/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── step/
│   └── <step>.yaml
├── flow/
│   ├── prediction.yaml
│   ├── hypothesis.yaml
│   ├── causal.yaml
│   └── stats_sequence.yaml
├── sample/
│   └── <name>.yaml
└── analysis/
    └── <name>.yaml

project/config/
├── dataset/
├── analysis/
├── experiment/
├── project/
├── sample/
└── slice/
```

| Location | Purpose |
|---|---|
| `configs/dataset/<name>.yaml` | Generic/test schema: columns, dtypes, target, task |
| `configs/experiment/<name>.yaml` | Generic feature, model, and split defaults |
| `configs/environment/<name>.yaml` | Development, staging, and production settings |
| `configs/step/<step>.yaml` | Per-step knobs and stats/train/features SSOT |
| `configs/flow/<mode>.yaml` | Mode-specific step lists |
| `configs/flow/stats_sequence.yaml` | Ordered stats-step list rendered into `reports/index.md` |
| `configs/sample/<name>.yaml` | Generic named sample spec |
| `configs/analysis/<name>.yaml` | Generic analytical intent |
| `project/config/` | Project overlay selected by project composition |

`configs/analysis/` holds generic/test analytical use cases.

The taxi development profile lives under:

```text
project/config/
```

Run project-bound platform commands through:

```bash
uv run python -m project.cli
```

This selects the project overlay explicitly. `ds-pipeline` remains data-agnostic.

Named samples are declared in:

```text
configs/sample/<name>.yaml
```

They include:

- Version
- Seed
- Size
- Columns
- Filters
- Schema

They generate immutable artifacts under:

```text
data/samples/
```

Format:

```text
<name>@v<N>.parquet
```

Samples are validated by:

```text
read_named_sample
```

Before steps consume them by name.

Config flow:

```text
YAML
  → Pydantic
  → load_config()
```

Schema code:

```text
src/broadway/config/schema.py
```

Rules:

- No defaults.
- No `get(key, default)`.
- No hardcoded values.
- Config YAML, schema, or environment variables only.

`DatasetContract` carries no `row_count`. Observed counts live in:

- `DatasetProfile`
- `TransformAudit`

Datetime dtypes are normalized to canonical `datetime64` at the schema boundary:

```text
schema.py::normalize_dtype
```

`lookup_tables` entries support:

| Field | Purpose |
|---|---|
| `value_policies` | Per-column sentinel values |
| `na_values` | Authored NA tokens |

Both are owned by config, not inferred from pandas defaults.

The raw feature schema comes from:

```text
configs/dataset/<name>.yaml
```

Adding or removing a raw feature means editing that YAML, then running:

```bash
ds-pipeline columns --csv <path>

ds-pipeline ingest \
  --dataset <name> \
  --experiment <name>

ds-pipeline profile \
  --dataset <name>
```

No code change is required.

Typed step outputs follow:

```text
artifacts/<step>/
```

Reports follow:

```text
reports/
```

---

# 6. Where Everything Lives

| Concern | Location |
|---|---|
| Architecture map | `dataflow.md` |
| Status / current work | `agents/ledger/STATE.md` |
| Stats library | `src/broadway/stats/` and `src/broadway/stats/API.md` |
| Decision and lineage graph | `src/broadway/lineage/` |
| Dataset loaders and constants | `project/data.py` |
| Script index | `project/STATS.md` |
| Add project work | `project/ADDING_WORK.md` |
| Config schema | `src/broadway/config/schema.py` |
| HPO / Optuna training / MLflow viewing | `HPO_TRAINING.md` |
| Tests | `tests/` |
| Branch parity gate | `scripts/check_branch_parity.sh` |

---

## Conventions for Agents and Humans

1. No hardcoded values. Use config YAML, `schema.py`, or environment variables only.
2. Shared functions live in one place and are imported, never duplicated.
3. The agent making a change updates `dataflow.md` in the same commit.

---

# 7. Branch Parity: `main` vs. `taxi`

Branch model and parity workflow live in:

```text
agents/contracts/MAIN_AGENT_CONTRACT.md
```

Relevant sections:

```text
§2
§12
```

| Branch | Role | Contents |
|---|---|---|
| `sklearn` | Active development line | Platform plus NYC taxi demo, including `project/`, scratch docs, and generated `reports/` |
| `taxi` | Pass-along mirror of `sklearn` | Fast-forward copy of `sklearn` |
| `main` | Public platform | Platform only: generic synthetic demo and `configs/sample/demo.yaml`; no `project/`, taxi config, or committed experiment output |
