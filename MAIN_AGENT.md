# MAIN_AGENT.md — Handoff for the next agent

This is the authoritative context for continuing work on **Broadway**. Read this first, then `AGENT_CONTRACT.md` and `coding_style.md`.

---

## 1. What Broadway is

A **traceable tabular data science platform** (Python, `uv`-managed). Thesis: statistical analysis + predictive modeling where **evidence → decisions → lineage** is first-class. It is NOT a bare ML pipeline — the point is that every result carries provenance and every analytical decision is a recorded artifact.

Repo: `fbarulli/broadway`. Working directory `/home/opc/ONE/broad-way`.

## 2. Branches

- **`taxi`** — active dev/demo branch. **ALL code work happens here.** Currently checked out. HEAD is `f763036`.
- **`main`** — public-facing platform branch. Carries the generic platform + only `README.md` and `BROADWAY.md` as docs (no scratch notes). Do NOT add your personal notes to `main`.
- **`broadway`** — stale/legacy. Do not touch; it is superseded by `main`.

Rule: development happens on `taxi`. The `main` split already happened; don't redo it.

## 3. How work gets done (non-negotiable)

From `AGENT_CONTRACT.md`:
1. Only perform actions explicitly agreed with the user.
2. **Never code or verify yourself — delegate everything to agents** (via the Task tool, `general` subagent) with a precise contract. All coding, test-running, data refresh, dogfood, and verification is done by agents; the main agent only delegates, splits work for parallelism, and navigates decisions.
3. Write agent contracts with specific roles/files; dispatch the right number of agents.
4. Agents commit + push, run the full suite, and fix failures.
5. Reuse/adapt static contract templates to minimize tokens.
6. **Always present decisions to the user; never decide unilaterally.** Agents act the same way.

Follow `coding_style.md`: type hints on public functions, strategic logging only, catch exceptions only when recoverable, YAML = single source of truth (no `get(key, default)`, no hardcoded values), ~25-line functions, no dead/noise code.

**Test gate:** `cd /home/opc/ONE/broad-way && uv run pytest -q`. It must be green after every change. Currently **247 passed**.

## 4. Architecture map (all in `src/broadway/`, demo in `project/`)

### Contracts (the "ownership model")
| Concept | Model | Where |
|---|---|---|
| Accepted raw schema | `DatasetContract` / `ColumnSchema` / `ColumnRole` (feature/target/datetime/ignore) | `config/schema.py`, `configs/dataset/<name>.yaml` |
| Observed facts | `DatasetProfile` / `ColumnProfile` | `discover/profile.py` |
| Authored intent | `AnalysisContract` (name, mode, goal, row_definition, decision_moment, available_info, leakage_notes, success_criterion) + `HypothesisConfig` (group_column/group_values) | `analysis/contracts.py`, `configs/analysis/<name>.yaml` |
| Naive reference | `BaselineResult` | `baseline/` |
| Feature spec | `FeatureSpec` | `features/schema.py`, `project/features.py` (taxi) |
| Provenance | `ArtifactTrace`, `TransformAudit`, `LineageRecord`, `DatasetRef`, `DatasetSlice`, `DecisionRecord`, `RunState` | `trace.py`, `lineage/` |

**Core principle:** `DatasetContract` = what the data *should* look like (authored/canonical); `DatasetProfile` = what was *observed*. Inference hints are evidence, never persisted as truth.

### `ds-pipeline` CLI
- Special commands: `discover`, `profile`, `lineage`, `init`.
- Steps (automated, used by `full`): `etl`, `contracts`, `eda`, `baseline`, `features`, `stats`, `causal`, `train`, `evaluate`, `full`.
- `full` is a **mode dispatcher** (`configs/flow/{prediction,hypothesis,causal}.yaml`) driven by `AnalysisContract.mode`, via `resolve_full_steps`.
- Mode enforcement: `require_mode(analysis, AnalysisMode.X)` in each step (stats→hypothesis, causal→causal, train/evaluate→prediction).
- **`stats` is now a nested subcommand group**: `stats run` (automated ANOVA) and `stats describe` (manual walkthrough). See section 6.

### Pipeline flow (prediction)
`init/discover → etl → contracts → eda → baseline → features → train → evaluate`

### Structural cleaning (recent, `etl` step)
`load raw → drop exact dups → standardize missing encodings → parse datetime (record ParseFailure) + coerce numeric (record ParseFailure) → drop target-null → strict validate (build_raw_schema) → save <name>_canonical.parquet + StructuralCleanResult + lineage sidecar → split`.
- `cleaning/` package: `models.py` (`ParseFailure`, `StructuralCleanResult`), `structural.py` (`parse_datetime`, `standardize_missing`, `parse_numeric`).
- `contracts/pandera.py`: `pandera_dtype`, `is_numeric_dtype`, `build_raw_schema` (nullable=True now).
- `contracts/selectors.py`: `feature_columns`, `datetime_columns`, `target_columns`, `numeric_columns`.
- `data/loader.py`: `load`, `canonical_path(dataset, environment)`.

### Lineage & decisions
`ds-pipeline lineage --analysis <n> --dataset <d>` → `artifacts/lineage/graph.json` + `graph.md` (Mermaid) + run-state (goal/stage/open+resolved decisions/not_yet_run/ran_but_output_missing). Steps write `LineageRecord` sidecars via `lineage/records.py::write_record`. `DecisionRecord` = runtime judgment calls (gitignored), NOT deterministic policy.

### Group-comparison walkthrough (recent)
- `stats describe` → `stats/describe.py`: `GroupSummary` + `GroupStat`, `plot_group_distribution` (boxplot), `plot_group_sizes` (imbalance bar), lineage sidecar `describe:<analysis>`. Writes `artifacts/stats/describe.json` + `describe_boxplot.png` + `describe_group_sizes.png`.
- Only `describe` exists so far. `normality`/`levene`/`anova`/`welch`/`kruskal`/`games-howell` subcommands are planned but NOT built.

## 5. Known constraints / open threads (do NOT "fix" without user sign-off)

1. **The canonical is a 50K random sample**, not the full data — `etl` applies `ci_sample_size: 50000` **unconditionally** (a pre-existing behavior; the taxi `process.py` only samples in CI). Consequence: Staten Island was *absent* from the canonical `describe` (n=0), imbalance_ratio ≈ 472.
2. **`--sample` routing is the immediate next step** (section 6). Until then, `stats` reads the canonical.
3. **Data-validation gaps (S5 closed the numeric ones; still open):** column renaming is deliberately NOT auto-done; semantic/domain cleaning (outliers, ranges, leakage) is deliberately deferred to analysis decisions (per `DATA_VALIDATION.md`).
4. **Sampling boundary is unresolved** (user chose option C — split estimation vs diagnostics — but the estimation-side canonical is still the 50K sample).
5. OLS regression diagnostics (TODO_FIRST_STEP.md) exist as taxi scripts `project/scripts/08..12` but have NOT been formalized into a walkthrough yet.

## 6. IMMEDIATE NEXT TASK — first-class `SampleSpec` + `--sample` routing (LOCKED PLAN)

The user locked this plan; implement it exactly. This is the next agent contract.

### Concept
A **sample** = a concrete materialized dataset with provenance + purpose. `sample_role: diagnostic | estimation` says which population interpretation applies. The same analysis can run on either sample; the result must record **both `sample_name` and `sample_role`** (truthful provenance, wired together with the data source).

### Step 1 — code (one agent contract)
1. `SampleSpec` model in `src/broadway/lineage/models.py`:
   ```python
   class SampleSpec(BaseModel):
       model_config = ConfigDict(extra="forbid")
       name: str
       role: Literal["diagnostic", "estimation"]
       path: str
       description: str | None = None
   ```
   (`Literal` already imported in that file.)
2. `load_sample(name) -> SampleSpec` helper (new `src/broadway/lineage/sample.py` or in `lineage/records.py`) reading `configs/sample/<name>.yaml` via `yaml.safe_load`.
3. Configs:
   - `configs/sample/taxi_estimation.yaml`: `name: taxi_estimation`, `role: estimation`, `path: data/processed/taxi_canonical.parquet`
   - `configs/sample/taxi_diagnostic.yaml`: `name: taxi_diagnostic`, `role: diagnostic`, `path: results/joined_sample_live.parquet`
4. `--sample <name>` **required** on `stats describe` and `stats run` (in the nested `stats` subparser in `cli.py`). Resolve via `load_sample`.
5. Wire provenance:
   - `stats/describe.py::run(cfg)` → change to `run(cfg, sample: SampleSpec)`; read `pd.read_parquet(sample.path)` instead of `canonical_path`; set `GroupSummary.sample_name` + `GroupSummary.sample_role`.
   - `GroupSummary` gains `sample_name: str` + `sample_role: Literal["diagnostic","estimation"]` (required).
   - `AnalysisPlan` (`stats/plan.py`) gains `sample_name: str | None = None` + `sample_role: str | None = None` (optional; stamped by the stats step).
   - `stats/module.py::run(cfg, sample=None)` — `sample=None` keeps reading the canonical (unchanged automated `full`-flow behavior); if a sample is provided, read `sample.path` and stamp `sample_name`/`sample_role` on the `AnalysisPlan`.
   - `LineageRecord` (`lineage/models.py`) gains `sample_name: str | None = None` + `sample_role: str | None = None`; `write_record(...)` gains `sample_name=None, sample_role=None` params; the describe/stats steps pass them.
   - Add `sample_name`/`sample_role` handling to the graph if trivial (optional; the sidecar JSON is the minimum).
6. Tests (`tests/test_sample.py` or extend `test_describe.py`): `SampleSpec` validation + `Literal` enforcement; `load_sample` round-trip; `describe.run(cfg, sample)` reads `sample.path` and stamps `sample_name`/`sample_role`; lineage sidecar carries both. Update `test_describe.py`'s `run(cfg)` call sites to pass a `SampleSpec`.
7. Full suite green; commit + push to `taxi`.

### Step 2 — regenerate the diagnostic cache (data refresh, no code)
```
cd /home/opc/ONE/broad-way && DATA_MODE=live uv run python -c "from project import data; data.generate_sample_cache('live')"
```
This re-reads `data/processed/training_data.parquet` and refreshes `results/joined_sample_live.parquet`. NOTE: it regenerates from `training_data.parquet` (the full ~8.5M dogfood-cleaned data), NOT the canonical (which is a 50K random sample and can't serve as a stratified source).

### Step 3 — dogfood
```
uv run ds-pipeline stats describe --dataset taxi --experiment taxi --analysis taxi_hypothesis --sample taxi_diagnostic
```
Expect: `GroupSummary` with `sample_name=taxi_diagnostic`, `sample_role=diagnostic`, and **all** boroughs present (Staten Island / EWR no longer n=0).

### Step 4 — verify truthfulness
`describe.json` carries `sample_name` + `sample_role`; `uv run ds-pipeline lineage --analysis taxi_hypothesis --dataset taxi` shows the describe node and its provenance in `graph.md`/`graph.json`.

### Success condition
`stats describe`/`stats run` require `--sample` (walkthrough); results + lineage carry `sample_name` + `sample_role` matching the data actually read; full suite green; diagnostic sample no longer drops small boroughs.

## 7. After that (queued, in this order)

1. **Resolve the estimation-side sampling boundary** (the canonical 50K vs full-data question; likely make `ci_sample_size` CI-gated like `process.py::sample_for_ci`).
2. **Group-comparison walkthrough** — add `stats normality` (Q-Q), `stats levene`, `stats anova`/`welch`/`kruskal`, `stats games-howell`, each with a paired visual + lineage. Then the router (recommendation logic) only *after* manual dogfooding.
3. **OLS regression walkthrough** from TODO_FIRST_STEP.md (8 diagnostics incl. influence/VIF/confounding not yet in scripts 08–12).

## 8. Working-tree hygiene

Pre-existing modified/untracked files to **never stage or delete** (user's scratch): `BROADWAY.md`, `DATA_VALIDATION.md`, `EXTERNAL_HELP.md`, `GENERAL_TODO.md`, `TODO_*.md`, `GOALS.md`, `LEARN.md`, `trust.md`, `FEEDBACK.md`, `synth.md`, `SENIOR.md`, `project.md`, `HANDOFF.md`. Gitignored artifacts (data/, artifacts/, results/) are also never committed.

## 9. Quick reference commands

- `uv sync` — install deps.
- `uv run pytest -q` — full test suite (must be green).
- `uv run ds-pipeline <command> ...` — the CLI.
- `uv run python -m project.scripts.NN_name` — taxi analysis scripts.
- Agents: use the Task tool with `subagent_type="general"` and a precise contract; require full-suite green + commit + push.
