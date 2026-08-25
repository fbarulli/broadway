# PACKET B — MODEL-PLANE · SENIOR ARBITRATION REPORT (verbatim, 2026-08-24)

## Step-0 echo

- **Hash gate:** `git rev-parse --short HEAD` → `5016e93` = dispatch stamp ✓ · branch `sklearn` ✓
- **Tree state:** unrelated WIP present and untouched (`M` tests/{assumptions,baseline,governance_probes,lookup_value_audit,structural_cleaning}.py + contracts/FIXES/STATE/etl/cleaning; `??` ledger/, experiments/16–23). No lane-owned file touched; zero writes performed.
- **STATE.md contradiction echo:** none. Packet dedupe decisions match STATE (warning-sweep→lane E rider; dead-code KEEP ruled — treated as context only, per both documents). One evidence correction surfaced during audit (finding 8, below); it amends the brief rather than contradicting my instructions.

**Assumption audit beyond the brief (mandatory floor):** (1) sklearn does **not** return `inf` for zero-target MAPE on this stack — it eps-clips: `uv run python` probe → `mape_with_zero: 150119987579016.62` (scipy 1.16.2 / numpy 2.5.2 / sklearn 1.7.2). A *finite absurd* number is worse than inf: any finiteness gate misses it, and downstream consumers see a plausible float. (2) The evaluate **val**-missing guard — the "good" twin in finding 7 — is itself unpinned (`grep -rn "held-out" tests/` → nothing). (3) The MAPE class also flows through CV (`validation.py:_SCORING["mape"] = "neg_mean_absolute_percentage_error"`), so gating `compute_metrics` alone does not close it. (4) `tests/test_splitter_extended.py` exists but covers only `chronological_split` — the two `split()` raises remain genuinely unpinned. (5) In-repo consumer set of `mean_residual` is exactly one site (`evaluate/module.py:148` → EvaluationResult JSON).

## RULINGS

```
FINDING: 1 GATE-TRAIN-30 — run() entry raises pinned only transitively via CLI exit-1 substrings
VERDICT: ADOPT
root: guard semantics tested at the wrong surface (CLI dispatch), so refactoring the owning unit lands green
rationale: verified live — the ONLY direct module.run test (test_training_contracts.py:276) is happy-path;
  strongest alternative (delete guards, rely on pydantic required fields) loses because require_mode +
  four-section disjunction cannot be expressed as field presence. Tripwire is pure test-add, reversible.
now-fix: SLATE S1
```

```
FINDING: 2 GATE-TRAIN-31a — splitter.split's two raise sites unpinned
VERDICT: ADOPT
root: fail-loud vocabulary asserted nowhere, so the class (silent guard deletion) is open repo-wide
rationale: grep "time split requires|stratified split requires|classification task" tests/ → ZERO hits;
  test_splitter_extended.py covers only chronological_split. Alternative (parametrized guard-table test)
  loses — two guards don't justify machinery.
now-fix: SLATE S2
```

```
FINDING: 3 GATE-TRAIN-31b — train self-splits without val; evaluate HARD-requires it: one config, two contracts
VERDICT: MODIFY(to: asymmetry made an EXPLICIT pinned contract — paired same-config test asserting
  evaluate raises 'held-out' FileNotFoundError while train self-splits, plus docstring truth in training/module.py)
root: cross-step input policy lives implicitly in two modules' divergent loaders, owned by neither
rationale: making evaluate fall back to self-split is REJECTED — it trades a loud error for silent
  evaluation-set mutation (SENIOR completeness rule); forcing val everywhere breaks legitimate dev
  self-split workflow and exceeds window. Pinning the tension is the cheapest honest end-state.
now-fix: SLATE S3
```

```
FINDING: 4 GATE-TRAIN-32 — read_training_sample seed PIN untested below glue
VERDICT: ADOPT
root: default-argument binding to config (project/data.py:106,168) is load-bearing reproducibility
  asserted by no test
rationale: all five validated_by ids hit broadway.data.loader directly with explicit seeds — confirmed
  from gate + tree. Strongest alternative (runtime warn on seed=None) rejected: noise on deliberate
  shuffles; signature-binding pin catches the decoupling class deterministically.
now-fix: SLATE S4
```

```
FINDING: 5 GATE-TRAIN-33 — model-type↔study mismatch raise (sole study↔estimator tie) zero coverage
VERDICT: ADOPT
root: the only semantic join between HPO studies and the trained estimator is guarded by unpinned code
rationale: grep "produced no valid trial" tests/ → ZERO hits; raise lives at training/module.py:84.
  Alternative (assert at objective-construction time instead) is better placement but a behavior move
  exceeding window; pin first, relocate later if ever touched.
now-fix: SLATE S5
```

```
FINDING: 6 GATE-TRAIN-34 — both experiment-required raises (trainer :31-32/:59-60) unpinned
VERDICT: ADOPT
root: same class as finding 1 — guard strings ("model pipeline requires"/"train requires an experiment")
  appear in no test file (grep verified)
rationale: merges naturally into S1's file; alternative (single parametrized raise-catalog test for the
  whole training band) is the right eventual shape but bigger than this window.
now-fix: SLATE S1 (shared)
```

```
FINDING: 7 GATE-TRAIN-35 — _load_train_features lacks the missing-file guard its val twin has
VERDICT: ADOPT
root: loader hardening applied to one twin only; module's own docstring (:3-5) promises named loud failures
rationale: verified verbatim — val twin has exists()+named FileNotFoundError, train twin calls
  pd.read_parquet raw (:55-61). AUDIT ADD: the val guard is ALSO unpinned (grep "held-out" tests/ →
  nothing), so the fix pins BOTH branches or the asymmetry merely flips direction.
now-fix: SLATE S6
```

```
FINDING: 8 GATE-TRAIN-36a — zero-target garbage MAPE flows silently into dict/MLflow/JSON
VERDICT: MODIFY(to: input-side exact-zeros gate raising ValueError "mape undefined" naming the metric,
  NOT an output-finiteness gate)
root: metric validity conditions checked only for NaN/inf inputs, not for per-metric domain violations
rationale: EVIDENCE CORRECTION — the brief says inf; live probe shows sklearn eps-clips to
  150119987579016.62 (command above), so an OUTPUT finiteness gate — the packet's own screened fix —
  MISSES the case entirely. Zero-target pre-gate is deterministic, keeps the 7-key vocabulary SSOT,
  and matches the existing input-gate pattern (:28-31). Alternatives: swap mape→smape (vocabulary
  change = HUMAN territory, unnecessary), clip/warn (error-for-warning trade = rejection class).
  CV-path rider (neg_mean_absolute_percentage_error folds) mapped onto the board row.
now-fix: SLATE S7
```

```
FINDING: 9 GATE-TRAIN-36b — 'mean_residual' computes mean ABS residual; test pins the mislabel
VERDICT: MODIFY(to: RENAME key to mean_abs_residual everywhere; NO compatibility alias)
root: name promises signed bias, value delivers |bias| — a labeling lie locked in by its own pin
rationale: rename-vs-alias RULED AS REQUIRED: alias keeps two names for one fact (SSOT violation) and
  preserves the trap for every future consumer; flipping the VALUE to signed under the old name is the
  worst option (silent semantic change under stable key). Verified blast radius is one in-repo consumer
  (evaluate/module.py:148 → EvaluationResult JSON); persisted historical JSONs keep old key — acceptable,
  recorded on board row. std/max_abs keys already truthful; only mean_residual lies.
now-fix: SLATE S8
```

```
FINDING: 10 GATE-TRAIN-37 — lower-is-better hardcoded; arbitrary target_metric inverts promotion; exceedance branch untested
VERDICT: MODIFY(to: metric-direction map HIGHER_IS_BETTER = {r2, explained_var} owned in metrics.py next to
  METRIC_DECIMALS, consulted by should_promote; unknown-metric fallback stays lower-is-better with current
  rmse semantics; direct pins for improvement-clears-threshold AND equality-never-promotes in BOTH directions)
root: comparison direction derived from parameter NAMES while data flows from config — policy hardcoded
  where fact should be derived from the metric SSOT
rationale: verified — promotion.py:9 hardcodes candidate>champion=refusal; run() passes
  cfg.evaluate.target_metric (evaluate/module.py:121-129); only no-champion (:100) and worse-candidate
  (:106) are pinned. HUMAN-CALL considered and REJECTED: r2/explained_var being higher-better is a
  mathematical fact, not ratifiable policy — no vocabulary is invented, direction is derived from the
  existing 7-key SSOT. Config-side direction flag rejected: moves the error into authoring.
  This is the packet's most dangerous path (see judgment).
now-fix: SLATE S9
```

```
FINDING: 11 GATE-TRAIN-38 — standardized_coefs/scenario_dollars index model.params ungated
VERDICT: ADOPT
root: module declares a fail-loud duck-type contract (estimation_table TypeError :48-53) that sibling
  functions in the SAME file ignore
rationale: verified — bare model.params[predictor]/[term] indexing, KeyError sole failure mode.
  Deletion-first considered: cluster has ZERO production callers, BUT the KEEP precedent (STATE open
  arbitrations) plus stats/API.md contract plus emerging pricing-engine experiments (WIP
  experiments/more_modeling/16_production_pricing_engine.py) make deletion wrong now; gates mirror the
  module's own established pattern.
now-fix: SLATE S10
```

```
FINDING: 12 GATE-TRAIN-39a — check_champion_manifest.sh drops $4+ args silently
VERDICT: ADOPT
root: positional argv parsing with silent overflow — usage contract enforced at arity 1-3 only
rationale: verified — $3 is checked against --strict (non-empty else exit 2), $4+ never examined:
  `--tracking-uri <uri> --strict --json` reports success while dropping --json. Tests pass --strict as
  final arg only (test_champion_manifest.py:145,154). Loop-based parse rejecting unknowns is ~6 bash lines.
now-fix: SLATE S11
```

```
FINDING: 13 GATE-TRAIN-39b — log_dataset StopIteration raise is mlflow_utils' ONE untested raise site
VERDICT: ADOPT
root: version-drift guard against MLflow registry churn (the exact failure T-BUG-era code was written
  for) itself unpinned
rationale: verified at mlflow_utils.py:118-124; monkeypatching get_registered_sources→[] makes it
  hermetically testable. Alternative (integration test with real MLflow downgrade) absurdly expensive.
now-fix: SLATE S12
```

```
FINDING: 14 GATE-STATS-41 — load_plan has ZERO production callers; plan JSON write-only in-band
VERDICT: REJECT(with: keep load_plan exactly as-is)
root: n/a — there is no defect to root-cause; the observation describes a healthy reader-of-record
rationale: deletion-first was applied honestly: deleting load_plan + its malformed-plan tripwire
  (tests/test_loud_failures.py:31) removes the ONLY drift detector for ARTIFACT-STATS-PLAN; wiring a
  production reader invents a consumer to justify machinery (worse than the finding). Cost of keeping:
  2 lines. Revisit iff a second plan writer ever appears.
now-fix: none — do-nothing-and-document; note disposition on the band row
```

```
FINDING: 15 GATE-STATS-42 — partial-absence invariant enforced only by convention; walkthrough/describe re-decide
VERDICT: MODIFY(to: builder-level enforcement — build_declared_groups gains require_complete: bool = True
  raising the standard vocabulary; explicit opt-outs ONLY at walkthrough runners :107 and describe's plot
  pass :107; manual post-checks in stats entry/baseline become dead and are removed)
root: safety-critical invariant defaults to OFF at the construction site, so every future caller inherits
  silence (exactly the T-BUG-4 regression shape)
rationale: opt-in strictness (default False) REJECTED — leaves the class open. Arithmetic kills the
  window: groups.py +10, stats/module ±3, baseline/hypothesis −2, describe +2, timeline/runners +1,
  tests +12 = 6 files ≈30 lines > 4-file cap ⇒ DEFER despite correct end-state. describe's n==0 retention
  (GATE-STATS-49 red-annotation feature) is why opt-outs must be explicit and visible at call sites.
now-fix: DEFER → board-row B-STATS-ABSENT-CONTRACT (carries describe redundant-second-construction rider, FIXES.md:111-112)
```

```
FINDING: 16 GATE-STATS-43 — small_group_threshold binds on ONE surface; walkthrough judges vs library-default 30
VERDICT: ADOPT
root: per-surface keyword defaults mask unbound policy — same warning string, two effective floors,
  neither surface owning the number visibly
rationale: verified — runners.py:315-319 passes alpha only; anova.py defaults 30 (:79-80,:121-122,:164-165);
  WalkthroughConfig has significance_alpha/imbalance_ratio_threshold but no floor key. Fix threads the
  floor through WalkthroughConfig (default 30 = library parity) with authored yaml set to 10000 matching
  CFG-STATS-FLOOR. Alternative (make library default required, no default) breaks all three scipy-family
  signatures and exceeds window.
now-fix: SLATE S13
```

```
FINDING: 17 GATE-STATS-45 — golden-float absence: no golden pin for ANY statistic/p_value/shapiro output
VERDICT: ADOPT
root: statistical outputs asserted only approx-relative against hand computations, so scipy-version drift
  is undetected BY DESIGN
rationale: verified — no tests/*golden* file; sole bitwise pin is HC3 idempotency (robust). Goldens
  GENERATED THIS SESSION on scipy 1.16.2/numpy 2.5.2 (fixed synthetic data, seed 42): f_oneway stat
  32.179178768901224 / p 2.222866172769085e-12 · welch t −3.2630800860503633 / p 0.0015229058717828076 ·
  eta_squared 0.29746185111687995 · shapiro_p(seed-0 subsample) 0.922801400099041. rtol=1e-12 chosen over
  bitwise to avoid the ULP cross-stack fragility that is packet D's sibling (cross-referenced, not merged).
  RIDER: accepting a future scipy upgrade that shifts goldens is a human policy call — flag on the board
  row: regeneration requires diff review, never silent re-record.
now-fix: SLATE S14
```

```
FINDING: 18 GATE-STATS-46 — shapiro_max_n=5000 and seed 0 are unconfigurable literals on every real path
VERDICT: MODIFY(to: StatsStep optional keys shapiro_max_n=5000 / shapiro_seed=0 (defaults preserve current
  behavior, authored yaml untouched); assumptions.check_normality gains seed param; stats step binds both;
  walkthrough leg stays literal — DEFER rider on same board row)
root: tuning knobs hardcoded in platform logic — invisible to config-diff review (config-over-hardcoded-
policy violation)
rationale: verified — literals at assumptions.py:20,:29; StatsStep schema (schema.py:252-266) exposes
  acf_lags etc. but neither knob. WIP collision avoided: test pin goes to NEW tests/test_stats_tunables.py,
  NOT WIP-owned test_assumptions.py. Window respected only by scoping to the stats-step surface.
now-fix: SLATE S15
```

```
FINDING: 19 GATE-STATS-48 — purity violation (viz.yaml reach through palette_colors) + dead cluster + Agg import side effect
VERDICT: MODIFY(to: formal viz-config injection boundary — plotting helpers receive palette explicitly from
  their step-entry callers; load_viz_config() confined to step entries)
root: a "pure library" whose transitive imports depend on configs/step/viz.yaml existing on disk
rationale: chain verified (diagnostics.py:37 → viz.palette_colors(1) → default_palette → load_viz_config →
  disk read; same reach at describe.py:112). Mitigating fact the brief understates: both call sites ARE
  plotting functions inside IO-sanctioned step legs — severity genuinely low. Blast radius (viz.py +
  diagnostics + describe + timeline callers) exceeds window ⇒ DEFER. Dead-cluster KEEP stands per finding 11;
  Agg-at-import belongs to infra lane.
now-fix: DEFER → board-row B-STATS-PURITY (palette injection + Agg side effect + knob-policy rider)
```

```
FINDING: 20 BACKLOG — pickle drift (pipeline.pkl schema unpinned across sklearn upgrades)
VERDICT: ADOPT
root: artifact serialization format has no declared stability contract, so upgrades mutate deployed
  artifacts invisibly
rationale: evidence live — data/processed/feature_pipeline.pkl exists; log_model forces cloudpickle.
  Proper fix is CI round-trip matrix/version-pinning policy = dependency-lane adjacent (E scope).
now-fix: DEFER → board-row B-SKLEARN-PICKLE-PIN
```

```
FINDING: 21 BACKLOG — index alignment (silent reindex-class hazards in frame handoffs)
VERDICT: REJECT(with: no platform machinery; reopen only with a demonstrated misalignment site)
root: n/a — no concrete hazard instance identified anywhere in bands 04/05 or my audit
rationale: ruling on an unnamed class invites speculative alignment machinery (reindex-everywhere wrappers =
  wrapper-of-wrapper indirection SENIOR Q2 hunts). Investigation ticket, not a contract.
now-fix: none — investigate-first board note only if the main agent wants it tracked
```

```
FINDING: 22 BACKLOG — schema-capture-vs-truth annex (captured ≠ validated)
VERDICT: ADOPT
root: capture-time schema and validate-time truth have no declared relationship, allowing divergence
rationale: distinct concern from queued Slate-v4 A1 (builder unification) — folding it into A1 would bury
  a separate invariant; standalone annex row is the honest mapping.
now-fix: DEFER → board-row B-SCHEMA-CAPTURE-ANNEX
```

```
FINDING: 23 BACKLOG+STATE — hpo .get silent-empty metrics + DatasetContract lacks mapping source
VERDICT: ADOPT
root: optional-with-default reads (.get("broadway_metrics", {}) at hpo.py:86) and logical-name resolution
  without a declared mapping owner both fail silent on drift
rationale: STATE already ratified the mapping-source gap as backlog; this register row carries it to the
  board as one line, per packet instruction. Latent-rename class confirmed structurally (baseline path
  resolves logical names against canonical load).
now-fix: DEFER → board-row B-HPO-METRICS-MAPPING
```

## SLATE (ready-to-execute; each ≤4 files / ≤60 lines / reversible / test-first)

| # | Finding | Files | Lines | Acceptance |
|---|---|---|---|---|
| S1 | 1+6 | tests/test_training_contracts.py | ~25 (4 raise-pins: run×2 sections, trainer×2) | `uv run pytest tests/test_training_contracts.py -q` |
| S2 | 2 | tests/test_splitter_extended.py (append) | ~18 | `uv run pytest tests/test_splitter_extended.py -q` |
| S3 | 3 | src/broadway/training/module.py (docstring ~3) + tests/test_evaluate_contracts.py (~13) | ~16 | `uv run pytest tests/test_evaluate_contracts.py -q` |
| S4 | 4 | tests/test_data_loader.py | ~10 (signature-default==RANDOM_STATE pin + double-call reproducibility) | `uv run pytest tests/test_data_loader.py -q` |
| S5 | 5 | tests/test_hpo.py | ~12 (models omitting model.type → match="produced no valid trial for model") | `uv run pytest tests/test_hpo.py -q` |
| S6 | 7 | src/broadway/evaluate/module.py (+4 guard) + tests/test_evaluate_contracts.py (+14, BOTH branches) | ~18 | `uv run pytest tests/test_evaluate_contracts.py -q` |
| S7 | 8 | src/broadway/evaluate/metrics.py (+4 zeros-gate) + tests/test_metrics_extended.py (+6) | ~10 | `uv run pytest tests/test_metrics_extended.py -q` |
| S8 | 9 | src/broadway/evaluate/validation.py (1) + tests/test_evaluate_contracts.py (~5) | ~8 | `uv run pytest tests/test_evaluate_contracts.py -q` |
| S9 | 10 | src/broadway/evaluate/metrics.py (+6) + promotion.py (+8) + tests/test_evaluate_contracts.py (+18) | ~32 | `uv run pytest tests/test_metrics_extended.py tests/test_evaluate_contracts.py -q` |
| S10 | 11 | src/broadway/stats/robust.py (+8) + tests/test_stats_robust.py (+10) | ~18 | `uv run pytest tests/test_stats_robust.py -q` |
| S11 | 12 | scripts/check_champion_manifest.sh (+6) + tests/test_champion_manifest.py (+8) | ~14 | `uv run pytest tests/test_champion_manifest.py -q` |
| S12 | 13 | tests/test_mlflow_utils_extended.py | ~10 | `uv run pytest tests/test_mlflow_utils_extended.py -q` |
| S13 | 16 | src/broadway/timeline/sequence.py (+1) + configs/step/walkthrough.yaml (+1) + src/broadway/timeline/runners.py (+3) + tests/test_walkthrough.py (+11) | ~16 | `uv run pytest tests/test_walkthrough.py tests/test_config.py -q` |
| S14 | 17 | tests/test_stats_golden.py (NEW; goldens embedded above, rtol 1e-12) | ~45 | `uv run pytest tests/test_stats_golden.py -q` |
| S15 | 18 | src/broadway/config/schema.py (+2) + src/broadway/stats/assumptions.py (+3) + src/broadway/stats/module.py (+1) + tests/test_stats_tunables.py (NEW +12) | ~18 | `uv run pytest tests/test_stats_tunables.py tests/test_config.py -q` |

No slated file overlaps another lane's WIP (verified against `git status`; WIP-owned test_assumptions.py/test_baseline.py deliberately routed around via two new test files).

## DEFER (one line each → CHANGE BOARD #5 row)

- **B-STATS-ABSENT-CONTRACT** ← #15: require_complete=True builder enforcement; 6 files ≈30 lines breach window; carries describe second-construction rider.
- **B-STATS-PURITY** ← #19: viz-config injection boundary + matplotlib-Agg import side effect + dead-cluster knob policy.
- **B-SKLEARN-PICKLE-PIN** ← #20: cloudpickle artifact schema stability across sklearn upgrades.
- **B-SCHEMA-CAPTURE-ANNEX** ← #22: captured-schema ≠ validated-truth contract, standalone from A1.
- **B-HPO-METRICS-MAPPING** ← #23: hpo `.get` silent-empty + DatasetContract mapping-source gap.
- Rider on #17's row: scipy-upgrade golden-regeneration protocol needs human ratification before the first scipy bump (regenerate-with-diff-review, never silent re-record).

## Tally

`VERDICTS: 14 adopt, 7 modify, 2 reject, 0 human-call` — 23/23 ruled; 15 slated with file×line arithmetic, 5 deferred with board mappings, 2 killed outright (#14 load_plan keep-with-rationale, #21 unevidenced index-alignment machinery).

## Judgment

The most dangerous silent-wrong-number path is **#10's promotion-inversion chain**: a one-line config edit (`configs/step/evaluate.yaml: target_metric: r2`) flips `should_promote`'s hardcoded lower-is-better comparison so that *degradation clears the threshold and genuine improvement refuses promotion* — with zero errors, plausible-looking reasons, and the champion alias (a deployment action, not a recorded number) moved to worse models that the manifest script then certifies healthy. It is compounded by #8: the same metrics dict can carry sklearn's eps-clipped pseudo-MAPE (150119987579016.62 — finite, so any finiteness gate sleeps) into MLflow and the persisted EvaluationResult, poisoning exactly the values the inverted comparator consumes. Both fixes are slated and together close the path: direction derived from the metric SSOT, garbage magnitudes impossible to emit. Which item dies first if the board wants blood: #21, already killed here — and #14 survives only because its "fix" would delete the artifact's only drift tripwire.
