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

Declared intent (`AnalysisContract`) gates which steps are valid: stats→hypothesis, causal→causal, train/evaluate→prediction.

Pipelines are mode-specific; `full` is a dispatcher that reads
`AnalysisContract.mode` and resolves the matching `configs/flow/{prediction,hypothesis,causal}.yaml`.

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
| ingest | `ds-pipeline ingest` | `data/processed/training_data.parquet` (~8.5M rows, 8 cols) + `ingest:taxi` lineage record | works (Polars; CI-gated) |
| etl | `ds-pipeline etl --dataset <d> --experiment <e>` | cleaned + split parquet + `JoinAudit`/`LookupValueAudit` (`join`/`lookup_value` lineage nodes) | works |
| contracts | `ds-pipeline contracts …` | pass/fail validation | works |
| eda | `ds-pipeline eda …` | `reports/eda.html` | works |
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
`DatasetSlice`s are authored config (`configs/slice/`); `DecisionRecord`s are
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
    <step>.md         # one page per completed step (question / what was run / found / why it matters)
  figures/*.png       # charts (normality Q-Q plots, ...)
  audit/              # human-readable data readiness (owned by audit; on-demand, typed renderers)
    index.md          # data used, status, what changed, enrichment quality, caveats
    profile.md        # observed column facts (dtypes, nulls, cardinality, identifiers)
    transform.md      # structural canonicalization: row transitions + parse failures
    join.md           # lookup key-matching completeness
    lookup_values.md  # matched-value quality (nulls/sentinels per enrichment column)
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
uv run python -m project.scripts.NN_name
```

Build the cache first (needed by scripts 04-12):

```bash
uv run python -c "from project import data; data.generate_sample_cache()"
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
DATA_MODE=dev  uv run python -m project.scripts.08_ols_residuals_diagnostics
DATA_MODE=live uv run python -m project.scripts.12_lgbm_baseline
```

---

## 4. Tests

```bash
uv run pytest              # 293 tests: library (synthetic) + data layer (real .head(1000)/cache)
```

---

## 5. Config (single source of truth)

```
configs/
  dataset/<name>.yaml      # per-dataset schema (columns, dtypes, target, task)
  experiment/<name>.yaml   # features, model, split, metric
  environment/<name>.yaml  # development / staging / production
  step/<step>.yaml         # per-step knobs + stats/train/features SSOT
  flow/<mode>.yaml         # mode-specific step lists (prediction/hypothesis/causal)
  flow/stats_sequence.yaml # ordered stats-step list rendered into reports/index.md
  sample/<name>.yaml       # SampleSpec (name, role, path, description, column_mapping) for `stats --sample`
  analysis/<name>.yaml     # authored analytical intent (--analysis <name>)
  project/<name>.yaml      # dataset ingest knobs (configs/project/taxi.yaml)
  slice/<name>.yaml        # authored DatasetSlice (configs/slice/)
```

`configs/analysis/` holds one YAML per analytical use case (e.g. `taxi.yaml`,
`taxi_hypothesis.yaml`, `taxi_causal.yaml`).

YAML → Pydantic (`src/broadway/config/schema.py`) → `load_config()`. No
defaults, no `get(key, default)`, no hardcoded values anywhere.

`DatasetContract` carries no `row_count` — observed counts live in
`DatasetProfile` (discover) and `TransformAudit` (etl). Datetime dtypes are
normalized to canonical `datetime64` at the schema boundary
(`schema.py::normalize_dtype`).

`lookup_tables` entries support `value_policies` (per-column sentinel values)
and `na_values` (authored NA tokens) — both owned by the config, not inferred
from pandas defaults.

Typed step outputs follow `artifacts/<step>/` and reports follow
`reports/`.

---

## 6. Where everything lives

| Concern | Location |
|---------|----------|
| Architecture map | `dataflow.md` |
| Status / what works | `HANDOFF.md` |
| Stats library (agnostic) | `src/broadway/stats/` (+ `API.md` contract) |
| Decision + lineage graph | `src/broadway/lineage/` (records/graph/mermaid/state) |
| Dataset loaders + constants | `project/data.py` |
| Script index | `project/STATS.md` |
| Config schema | `src/broadway/config/schema.py` |
| Tests | `tests/` |

### Conventions (for agents and humans)

1. No hardcoded values — config YAML / `schema.py` / env var only.
2. Shared functions live in one place and are imported, never duplicated.
3. The agent making a change updates `dataflow.md` in the same commit.
