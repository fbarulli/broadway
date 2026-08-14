# broadway — Current Handoff

## Repository state
- Active development branch: `taxi`.
- Suite green. Last confirmed count: **384 tests** after the Plotly/EDA-removal commit; Q-Q/distribution/docs tests added since — run the full suite to confirm the current total.
- The walkthrough/timeline system is mature and dogfooded. Plotly and the `eda` package are fully removed.
- Current work: about to start a **5-part surface-polish follow-up** (the active next slice). Nothing in it has been executed yet.

## Product hierarchy & report surfaces
Hierarchy: **Audit → Timeline → Evidence → Suggestion → Decision → Result.**

Single-owner surfaces (each has exactly one writer-of-record):
- `reports/index.md` → walkthrough/dashboard only.
- `reports/results/` → walkthrough owns it (legacy `report` command is now a thin wrapper over the same renderer; delete the wrapper eventually, don't keep it permanently).
- `reports/timeline.md` → timeline.
- `reports/audit/` (incl. `profile.md` → "Profile evidence" section) → audit.
- `reports/lineage/` → lineage (provenance only, not a workflow UI).

Guardrail: `tests/test_surface_integrity.py` enforces (a) every relative markdown link in tracked reports/docs resolves, and (b) size caps — **5 MB HTML / 2 MB PNG**. Keep this test meaningful; don't raise caps to pass it.

## The two Q-Q surfaces (deliberate divergence — documented, not drift)
- **Features Q-Q** — from `discover`/`profile`, rendered on `reports/audit/profile.md` under "Profile evidence". **Small multiples**: one subplot per numeric feature, adaptive grid, figure scales to fill the screen. Built in `src/broadway/discover/qq.py`.
- **Groups Q-Q** — from the walkthrough normality step, on the timeline/results. **Single overlaid figure**, one trace per group. Correct because group count is low (5); a deliberate choice, not an accident.

Rationale to preserve: 7+ features don't read overlaid (→ small multiples); 5 groups read fine overlaid (→ single figure). If the overlaid groups plot ever stops reading well, converge on small multiples — but record the reason.

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
All surface-polish, smaller than the deferred architecture work. Execute in order; one commit per part; suite green each time.

**1. Console humanization + de-prescribe (shipped-invariant bug).**
- `timeline/walkthrough.py::_print_decision_required`: change the gate's `Next:` line from `--method welch` to `--method <method>` (the Suggestion block is already de-prescribed; the `Next:` line is the contradiction). Verify the posthoc gate's `--kind <kind> --method <method>` and leave it.
- Humanize the console gate + suggestion rationale (3 sig figs, p floored `< 0.001`; no raw `p_value=0.0` / `statistic=4199.71…`). Reuse `humanize_float`/`humanize_pvalue` — **prefer extracting them to a shared formatting module** rather than importing `reports` into `timeline` (wrong direction); never duplicate.

**2. Posthoc completeness.**
- Render the actual significant pairs on `results/posthoc.md` + the timeline posthoc section: a table of `a vs b`, p, Cohen's d, Hedges' g, note. Answer the step's own question ("which pairs differ"). Show the denominator too ("17 of N pairs significant").

**3. Profile figures overhaul.**
- Seaborn: `sns.despine()` per subplot; `sns.set_palette("BuPu_r")` default (overridable); `sns.histplot` for the distribution grid. **Decision: promote seaborn to a main dependency** (matplotlib is already core; profile figures are core flow).
- Automated bins: continuous → `bins='auto'`; discrete/integer → integer-aligned bins / `histplot(discrete=True)`.
- Title overlap: `plt.subplots(..., layout='constrained')`, drop manual `tight_layout`/`subplots_adjust`.
- **Discrete Q-Q detection:** `n_unique <= MIN_UNIQUE_FOR_QQ` (default 15, overridable via env var `BROADWAY_QQ_MIN_UNIQUE`, mirroring `BROADWAY_IDENTIFIER_THRESHOLD`) → exclude from the Q-Q grid, keep in the distribution grid with integer bins, note *"excluded from Q-Q: discrete (N unique values)"*. **No transforms, no jitter** (they don't remove ties).
- **ID columns:** exclude by explicit declaration (`id_cols`/`exclude_from_profiling`) sourced from **per-dataset config (YAML)** and passed as an explicit param to `plot_numeric_qq` — ID-ness is authored intent, not a global. A `*_id` name heuristic may **flag/suggest only, never silently exclude**.
- Friendly captions (not raw comma lists), datetime-absent note (8 cols → 7 features explained), chunk/exclusion explanation (`_1` suffix + excluded features).

**4. Timeline/results polish.** Trust items first, cosmetic after.
- Trust: carry the η²→ω² "report ω²" caveat through to `conclusion.md`; label statistics (`F =`, `Levene statistic =`, `Welch`).
- Then: link `evidence_refs` (inline/link, not raw JSON filenames); human step names in result filenames/titles (replace `omnibus.md`/`posthoc.md`/`describe_groups.md` — update results renderer, orphan deletion, and `test_surface_integrity.py` links **together**); dedup `total_n`/`n_total`; fix `imbalance_ratio` precision; Q-Q legend (trace → group); reconcile "unlisted group" vs "Unknown".

## Deferred queue (order confirmed)
1. **Kruskal ε²** (rank-based effect size — smallest hole).
2. **Registry refactor** — a `STEP_REGISTRY` replacing the `if/elif` dispatch in `walkthrough.py`/`suggest.py` (do this **before** adding any new step).
3. **LoadAudit / ParsingPolicy** (the loader walkthrough; trace physical read → parsing → raw validation → canonicalization).
4. **Lineage-viz** (`graph_todo.md`) — `reports/lineage/graph.md` currently shows only data-prep, is stale/unrelated.
- Also pending: move suggestion templates + effect-size wording into config; delete the legacy `report` wrapper.

## Working principles to preserve
- Results are the primary human-facing product surface. Audit explains what happened to the data. Timeline explains where the analyst is. Evidence reports what diagnostics found. Suggestions guide, never decide. Decisions belong to the analyst and are explicitly recorded. Lineage is provenance, not the workflow UI.
- **Authored intent ≠ observed evidence ≠ runtime decision.** No silent branches. No silent analytical verdicts.
- Reports render persisted evidence; they never become a second source of truth.
- Tests must never mutate shared runtime evidence or reports (conftest guard applies).
- Build and dogfood narrow vertical slices before adding abstraction.
- Architectural pillars: **typed evidence / dumb renderers** (pydantic records; renderers only read); **single-owner surfaces**; **compute/IO isolation** (stats/audit pure; timeline owns I/O); **config over hardcoded policy**; **ruthless pruning** (delete dead surfaces — the 619 MB `eda.html` lesson).
- Dogfood rule: every time source code or raw JSON must be opened to understand the analysis, record it as a product-surface gap (`Needed to inspect <source/json> to understand <question>. Expected human surface: <…>`). Report gaps first; patch only after.