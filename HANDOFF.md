# broadway — Current Handoff

## Repository state
- Active development branch: `taxi`.
- Suite green: **466 tests**.
- The 5-part surface-polish follow-up is **complete**.
- `AGENT_WORKER_CONTRACT.md` added (immutable worker rules).
- A 3-agent read-only review pass completed.
- The walkthrough/timeline system is mature and dogfooded. Plotly and the `eda` package are fully removed.

## Product hierarchy & report surfaces
Hierarchy: **Audit → Timeline → Evidence → Suggestion → Decision → Result.**

Single-owner surfaces (each has exactly one writer-of-record):
- `reports/index.md` → walkthrough/dashboard only.
- `reports/results/` → walkthrough owns it (legacy `report` command is now a thin wrapper over the same renderer; delete the wrapper eventually, don't keep it permanently).
- `reports/timeline.md` → timeline.
- `reports/audit/` (incl. `profile.md` → "Profile evidence" section) → audit.
- `reports/lineage/` → lineage (provenance only, not a workflow UI).

Guardrail: `tests/test_surface_integrity.py` enforces (a) every relative markdown link in tracked reports/docs resolves, and (b) size caps — **5 MB HTML / 2 MB PNG**. Keep this test meaningful; don't raise caps to pass it.

## The two Q-Q surfaces (converged on small multiples — documented, not drift)
- **Features Q-Q** — from `discover`/`profile`, rendered on `reports/audit/profile.md` under "Profile evidence". **Small multiples**: one subplot per numeric feature, adaptive grid, figure scales to fill the screen. Built in `src/broadway/discover/qq.py`. Draws config-driven diagnostic zones (`qq_zones` in `configs/step/viz.yaml`: tail bands, central band, zero-mass shelf) as visual references only — the groups Q-Q draws none.
- **Groups Q-Q** — from the walkthrough normality step, on the timeline/results. **Small multiples (one subplot per group)**, a deliberate convergence for consistency (made in commit `00e0234`).

Rationale to preserve: 7+ features don't read overlaid (→ small multiples); the groups plot now converges on the same small-multiples layout so both Q-Q surfaces render consistently. If either surface ever stops reading well, record the reason before diverging again.

## Dataset context (taxi)
- Analysis `taxi_hypothesis`, sample `taxi_diagnostic`, dataset `taxi`. Outcome: `trip_duration_minutes`.
- Grouping col `pickup_borough`: Manhattan 179 502, Queens 18 113, Brooklyn 1 361, Bronx 361, Staten Island 83, Unknown 633, EWR 77. Total 200 130; **used 199 420** (710 excluded as "unlisted group" = 633 Unknown + 77 EWR). Analysis runs on 5 groups (Manhattan/Queens/Brooklyn/Bronx/Staten Island).
- Known data property: extreme imbalance — Staten Island's trace is near-invisible vs Manhattan. That's a data property, not a layout bug.

## Completed work (chronological)
1. Six-phase walkthrough foundation: `AnalysisStep`, `AnalysisDecision`, `Suggestion`/`Alternative`, question-oriented `hypothesis_walkthrough.yaml`, persistence under `artifacts/timeline/<analysis>/`, human renderer `reports/timeline.md`. Commits: evidence-integrity `e7d6bdd`, timeline `815c02e`, executable evidence `9d7926f`, decision gate `ce430da`, principal analysis `0d5b480`, suggestion layer `f63a438` + `aad3172`.
2. Principal analysis dispatch: ANOVA / Welch / Kruskal; effect-size evidence; post-hoc eligibility; two decision gates. **Kruskal effect size is deliberately not computed** (rank-based ε² pending) — never reused ANOVA η²/ω², never worded "not yet implemented".
3. A+B (`ea62c18`): single owner for `reports/index.md`; `--dataset` required for `walkthrough`.
4. **Set 1** (`ef3efae`): walkthrough owns `results/`; humanized rendering (3 sig figs, p floored at `< 0.001`, no dict literals/machine paths); effect-size framing (η² vs ω² one-liners); attrition line; plain-text status vocabulary (**completed / completed with note / awaiting decision / failed / warning**-only-if-interpretation-changing); `failed` step capture with the three guardrails.
5. **4-commit sequence** (all pushed, 384 tests):
   - `b9d8ceb` docs catch-up.
   - `d386cde` Set 2: describe_groups boxplot + group-size figures with "how to read" captions, link-depth fix, de-prescribed `--method <method>` suggestions.
   - `bb730e0` Joint Q-Q (Option C): normality gate → one joint-per-group Q-Q; removed the 5 individual `normality_*.png`.
   - `0cf67fa` Removed Plotly + `eda` (Option A), added `test_surface_integrity.py`, swept stale artifacts (619 MB `eda.html` gone).
6. **Small-multiples Q-Q + distribution grid** for the profile surface (per-feature subplots, adaptive grid, screen-filling, Q-Q point-thinned). Evidence models `QqFeature` / `QqOverview` (pydantic); one shared per-feature processing pass feeds both grids so order/exclusions match by construction; figure assignment keyed by feature name (never position); `source_path` is a required keyword-only arg; no input mutation. Followed by a docs re-sync and a dogfood pass.
7. **5-part surface polish** (all `taxi`): console humanize + de-prescribe `b30fe34`; posthoc pairs + effect sizes `2e951cf`; profile figures overhaul (seaborn→main dep, discrete/ID detection, automated bins, constrained layout) `66b1cec` + `17ca922`; timeline/results polish (human slug filenames, labeled statistics, eta²→omega² caveat to conclusion) `44fc1fa`.
8. **Kruskal epsilon²** `86df9f6` — `epsilon_squared()` `(H-k+1)/(N-k)` clamped [0,1]; Kruskal no longer emits eta²/omega²; backward-compat for old "not_computed" evidence.
9. **Q-Q min/max automation** `2cdc7d7` — features Q-Q independent axes + probplot fit line; groups Q-Q y=x diagonal data-derived.
10. **Groups Q-Q → small multiples** `00e0234`.
11. **W1 config + critical fixes** `17153d3` — `configs/step/viz.yaml` + `max_qq_groups`; stats thresholds → params; fixed run_variance (shapiro_alpha→significance_alpha) and run_omnibus (ignored alpha); zero-variance guard; shared plot styling; describe combine (single describe.png); retired legacy `stats describe` output.
12. **Per-feature distribution diagnostics** — `QqFeature`/`QqOverview` gain optional `skew`/`kurtosis`/`diagnostics_figures`; `DiagnosticsConfig` (`diagnostics` block in `configs/step/viz.yaml`); `_plot_diagnostics_heatmap` renders a single `numeric_diagnostics.png` (per-column z-score over `[skew, kurtosis, zero_rate]`, raw values annotated); `audit.py` renders a `mean`/`std`/`skew`/`kurtosis`/`zero_rate` table + the heatmap on `reports/audit/profile.md`. Visual reference only — no verdicts/thresholds.
13. **Q-Q figure overhaul** (this session, all `taxi`): value-centered discrete bins + config-driven downsampling + single `n` in suptitles `570a9a2`; `tip_amount` added to the working dataset `2b3c216`; contract-driven `ingest` + `columns` command `5b2b824`; config-driven diagnostic zones (tail/central bands, zero-mass shelf) `632afd0`→`d265726`; per-feature distribution diagnostics table + heatmap + decision flags `e90edd8`/`7a669bf`; timeline registry refactor (config-driven decisions + runner/suggestion registries) `e619a4f..8f2effa`; raw-vs-log Q-Q comparison `16db124`/`12fc1a7`; darker categorical palette sampling `4e245b1`; decision-mapped markers (percentile gridlines, tail boundary, robust IQR fit) `75b8e4e`/`d96414f`; all Q-Q rendering consolidated into `qq.py` `32dd1f6`; complete legend (fit/zones/markers) + heatmap annotation contrast `6a9855b`; horizontal raw/log layout (raw top, log bottom, features as columns) `3505570`.

## Current CLI usage
```
uv run ds-pipeline audit      --dataset taxi --analysis taxi_hypothesis
uv run ds-pipeline walkthrough --analysis taxi_hypothesis --dataset taxi --sample taxi_diagnostic
uv run ds-pipeline decide      --analysis taxi_hypothesis --method <method> --reason "..."
uv run ds-pipeline walkthrough --analysis taxi_hypothesis --dataset taxi --sample taxi_diagnostic
uv run ds-pipeline decide      --analysis taxi_hypothesis --kind posthoc --method <method> --reason "..."
uv run ds-pipeline walkthrough --analysis taxi_hypothesis --dataset taxi --sample taxi_diagnostic
```
Suggestions are **de-prescribed**: the product prints `--method <method>` templates, never a pre-filled method. `discover`/`profile` generate the Profile-evidence figures (`src/broadway/discover/qq.py`).

## Immediate work left — 5-part surface-polish plan
COMPLETE — see "Completed work" items 7–11 (commits `b30fe34`, `2e951cf`, `66b1cec` + `17ca922`, `44fc1fa`).

## Deferred queue (order confirmed)
1. **Replace `total_amount` → `fare_amount`** — remove `total_amount` from the working dataset, add `fare_amount` (base fare) in its place: `features/schema.py` + `configs/dataset/taxi.yaml` + test fixtures, then re-run `ingest` + `profile` + `audit`. Confirmed; next up.
2. **Q-Q doc pass** — `README.md`/`dataflow.md` Q-Q sections predate the zones/markers/raw-log work (the "overlaid → small multiples" wording was fixed, but the surfaces now carry zones + markers + a raw/log comparison, and `normality_qq` is a 2×n raw/log grid); `stats/API.md` drift; `HANDOFF.md` §"The two Q-Q surfaces" still says "groups Q-Q draws no zones".
3. **W2 main-sync** — taxi-free `main`; convert/exclude taxi-referencing tests (`test_contracts.py:31` real-parquet load, hardcoded taxi stats in `test_walkthrough.py`/`test_results.py`, `dataset="taxi"` coupling).
4. **Cleanup/polish slice** — orphaned legacy results-index (`render_index`/`load_stats_sequence`/`RESULT_RENDERERS`), doc drift (`README.md:92`, `stats/API.md:199` still describe retired `stats describe`), taxi strings in `audit.py` ("NYC boroughs"), dead code (`suggest.py` unused cfg, `decide.py::_question_for` re-reads YAML), silent posthoc skip, unlabeled Q-Q truncation.
5. **LoadAudit / ParsingPolicy**.
6. **Lineage-viz** (`graph_todo.md`).
7. Move suggestion templates + effect-size wording into config; delete legacy `report` wrapper.
8. Pin sample/evidence, then refresh reports (drift follow-up).
9. W1-flagged: Q-Q style constants Python-vs-YAML decision; hardcoded `figures/` path prefix.

## Working principles to preserve
- Results are the primary human-facing product surface. Audit explains what happened to the data. Timeline explains where the analyst is. Evidence reports what diagnostics found. Suggestions guide, never decide. Decisions belong to the analyst and are explicitly recorded. Lineage is provenance, not the workflow UI.
- **Authored intent ≠ observed evidence ≠ runtime decision.** No silent branches. No silent analytical verdicts.
- Reports render persisted evidence; they never become a second source of truth.
- Tests must never mutate shared runtime evidence or reports (conftest guard applies).
- Build and dogfood narrow vertical slices before adding abstraction.
- Architectural pillars: **typed evidence / dumb renderers** (pydantic records; renderers only read); **single-owner surfaces**; **compute/IO isolation** (stats/audit pure; timeline owns I/O); **config over hardcoded policy**; **ruthless pruning** (delete dead surfaces — the 619 MB `eda.html` lesson).
- Dogfood rule: every time source code or raw JSON must be opened to understand the analysis, record it as a product-surface gap (`Needed to inspect <source/json> to understand <question>. Expected human surface: <…>`). Report gaps first; patch only after.