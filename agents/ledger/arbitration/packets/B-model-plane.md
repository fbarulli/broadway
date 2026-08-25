# PACKET B — MODEL-PLANE (bands 04-training-eval · 05-stats)

- Packet id: `B-model-plane` · findings: **23** (18 gate-band + 5 backlog adds)
- Sources: agents/ledger/GATES.md bands 04/05 · agents/ledger/FIXES.md §Slate v4/§backlog ·
  agents/ledger/STATE.md §Backlog/open-arbitrations · cross-refs into factsheets
  2026-08-24-{det-ledger,perf-baseline}.md and packets A/D.
- Required reading before ruling: GATES.md lines 384–639, FIXES.md:30–55 + :104–117,
  STATE.md:83–92.

## FINDING REGISTER

1. GATE-TRAIN-30 — training run() entry raises pinned ONLY via CLI wrapper exit-1 substrings; refactoring run()'s guards passes the whole suite silently.
2. GATE-TRAIN-31a — splitter.split's two raise sites UNPINNED (time-split w/o datetime_column; stratified-on-regression); silent guard deletion lands green.
3. GATE-TRAIN-31b — val-absent→self-split vs evaluate HARD-REQUIRES val file: same config satisfies one step, crashes the other.
4. GATE-TRAIN-32 — read_training_sample seed PIN untested below glue: decoupling the RANDOM_STATE default changes every experiment's sample invisibly; seed=None silently disables reproducibility. [sibling of A#22 bypass]
5. GATE-TRAIN-33 — model-type↔study mismatch raise (the ONLY tie between study and estimator) has ZERO coverage.
6. GATE-TRAIN-34 — both experiment-required raises in build_model_pipeline/train() unpinned.
7. GATE-TRAIN-35 — asymmetric loader hardening: _load_train_features lacks the missing-file guard its val twin has; deleted train parquet = raw pyarrow error, not the named loud failure.
8. GATE-TRAIN-36a — OUTPUT finiteness never gated: y_true zeros ⇒ MAPE=inf flows into metrics dict, MLflow, persisted JSON; input-only checks (:28-31).
9. GATE-TRAIN-36b — residual_summary 'mean_residual' actually computes mean ABS residual, test PINS the mislabeled semantics — bias consumers get |bias|.
10. GATE-TRAIN-37 — lower-is-better HARDCODED while caller feeds arbitrary target_metric: evaluate.yaml→r2 INVERTS every promotion decision; threshold-exceedance branch has no direct test at all.
11. GATE-TRAIN-38 — standardized_coefs/scenario_dollars index model.params with NO fitted/TypeError gate; KeyError on unknown term is sole failure mode, unpinned.
12. GATE-TRAIN-39a — check_champion_manifest.sh argv handling stops at $3: trailing flags after --strict silently dropped, no usage rejection.
13. GATE-TRAIN-39b — log_dataset StopIteration raise is the ONE untested raise site of mlflow_utils.py.
14. GATE-STATS-41 — load_plan has ZERO production callers; stats plan JSON is write-only in-band; reader-side drift surfaces nowhere except tests.
15. GATE-STATS-42 — 'raise on partial absence' invariant enforced ONLY BY CONVENTION: groups.py structurally cannot raise; every future caller re-decides; walkthrough keeps size-0 arrays by docstring only; describe re-runs builder ignoring absent AGAIN.
16. GATE-STATS-43 — floor binds on ONE surface: walkthrough never passes small_group_threshold → underpowered warnings judge vs library-default 30 while stats step judges 10000; same warning string, two effective floors. [dedupe: IS the FIXES.md floor-kwarg-tension rider — one finding, not two]
17. GATE-STATS-45 — GOLDEN-FLOAT ABSENCE: no golden pin exists for ANY statistic/p_value/shapiro output (approx everywhere, one bitwise HC3 pin) — scipy-version drift undetected by design.
18. GATE-STATS-46 — shapiro_max_n=5000 AND seed 0 are unconfigurable literals on every real path; no config key exposes them; tuning requires code edit invisible to config-diff review.
19. GATE-STATS-48 — PURITY VIOLATION + dead cluster: stats "pure library" reads configs/step/viz.yaml through viz.palette_colors (diagnostics.py:37, describe.py:112); NO production caller for ANY robust/diagnostics/time_series/baseline member; thresholds unpinned; train_lgbm pins no seed.
20. BACKLOG-FIXES — pickle drift (pipeline.pkl schema unpinned across sklearn upgrades).
21. BACKLOG-FIXES — index alignment (silent reindex-class hazards in frame handoffs).
22. BACKLOG-FIXES — schema-capture-vs-truth annex contract (captured schema ≠ validated truth).
23. BACKLOG+STATE — hpo `.get` defaults + DatasetContract lacks mapping source (baseline path uses logical names against canonical load — latent rename class). STATE backlog rows carried here as one register line each ruling surface.

Dedupe decisions: #16 ≡ FIXES.md floor-kwarg rider (merged, one entry). Golden-float: this packet owns the ABSENCE claim (#17); D owns det-ledger (e) ULP-exact-equality fragility (test_ml_pipeline) — siblings, cross-ref, do not merge. log_dataset second-read cost deduped to D (perf-baseline phase 5b / D9). Warning-sweep backlog rider routed to E (dependency/CI scope). Dead-code KEEP rulings (STATE open arbitrations) already resolved — context only.

## RULING FORMAT (one block per finding — mandatory shape)

```
FINDING: <register id> — <one-line restatement>
VERDICT: ADOPT | MODIFY(to: <concrete end-state>) | REJECT(with: replacement/do-nothing rationale) | HUMAN-CALL(owner: human, ask: <exact ask>)
root: <deepest cause whose repair kills the CLASS — mandatory non-empty>
rationale: <why; name strongest alternative considered and why it loses>
now-fix (ADOPT/MODIFY only): files=[...] · changed-lines≤N · acceptance=<exact command>
```

## SLATE-vs-DEFER SEPARATION

- SLATE = concrete-now-fix fits THIS packet's window: **≤4 files, ≤60 changed lines total,
  reversible, test-first preferred**. Slate items listed explicitly under `## SLATE`.
- DEFER = everything else → `DEFER → board-row: <proposed row id/scope>` mapped to CHANGE
  BOARD (#5). An item is slated-with-diffs or deferred-with-mapping; nothing between.
- Show files × lines arithmetic or the fix is not slateable.

Slate candidates screened (senior still rules):
- #8+#10: tests/test_metrics_extended.py append — zero-target MAPE inf tripwire +
  should_promote r2-inversion + threshold-exceedance direct pins ≈35 lines, 1 file,
  acceptance `uv run pytest tests/test_metrics_extended.py tests/test_evaluate_contracts.py -q`.
- #18: expose shapiro_max_n/seed via StatsStep keys — MODIFY candidate; test-first pin ≈20 lines;
  if config-key change pushes past window ⇒ DEFER with board mapping.
Do NOT edit WIP-modified tests/test_assumptions.py, tests/test_baseline.py (other lanes own them).

## MANDATE REMINDERS (non-negotiable)

- `root:` lines are MANDATORY on every block (MAIN_AGENT_CONTRACT §14 ROOT-CAUSE MANDATE).
- Lukewarm rulings FAIL the mandate — commit to a verdict class and defend it, or ESCALATE
  explicitly per SENIOR vocabulary.
- Human-only calls MUST be flagged `HUMAN-CALL` (e.g., ratifying a metric-vocabulary change,
  scipy upgrade policy) rather than deferred silently.
