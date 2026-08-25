# PACKET C — PLATFORM-WIRING (bands 06-timeline-lineage · 07-surfaces · 08-config-schema)

- Packet id: `C-platform-wiring` · findings: **28** (21 gate-band inline + 6 named F-SURF +
  1 backlog add; band 07 carries 0 inline `# FINDING` blocks but 6 named findings — count
  noted so tallies reconcile against GATES.md's official 78)
- Sources: agents/ledger/GATES.md bands 06/07/08 (incl. SURFACE OWNERSHIP MAP +
  COMPUTING-RENDERER FINDINGS) · agents/ledger/STATE.md §Backlog · cross-refs into
  factsheets 2026-08-24-{deploy-diff,det-ledger}.md and packets D/E.
- Required reading before ruling: GATES.md lines 641–1119, STATE.md:89–92.

## FINDING REGISTER

1. GATE-TLINE-52a — decision method vocabulary DOUBLE-OWNED, never cross-validated: record-time YAML allowlist vs execute-time hardcoded forks; a plausible-but-wrong method records "resolved" then crashes run_omnibus into a FAILED step.
2. GATE-TLINE-52b — the two ends of the decide fork disagree on bad-kind failure: executor prescribes a CLI command record() will reject; _print_decision_required hard-indexes thresholds.decisions → KeyError while record() uses .get.
3. GATE-TLINE-53 — writers-of-record duplicated: timeline/module save_step/save_decision vs lineage write_record, two identical mkdir+write_text layers over parallel env roots; any durability/atomic-write fix must land twice.
4. GATE-TLINE-55 — execute-time forks are the unvalidated twin of the allowlist; run_conclusion's else-branch will KeyError('epsilon_squared') on any future admitted non-anova/welch/kruskal method.
5. GATE-TLINE-56a — evidence nobody reads back: six evidence JSONs never opened by production code; all surfaces render the DUPLICATED result_summary copy; writer desync ships undetected by the product.
6. GATE-TLINE-56b — two modules own one artifact name: stats/describe.json (the only one read) vs timeline describe.json (rewritten, unconsumed) with divergent freshness.
7. GATE-TLINE-58a — lineage decision records have a reader but NO writer: DecisionRecord instantiated nowhere in production; open/resolved_decisions fed by hand-maintained test-fixture data.
8. GATE-TLINE-58b — node_id→filename collision silently overwrites records-of-record ('a:b','c' vs 'a','b:c' both → a_b_c.json), last-writer-wins, graph renders whichever survived.
9. GATE-TLINE-58c — KIND_LABELS + LINEAGE_STEPS omit live 'coercion' kind: dead label vocabulary coexisting with an unwritten-in-labels kind; coercion nodes never advance stage accounting.
10. GATE-TLINE-59a — project/scripts/* eleven teaching entry points bypass ALL gates: no AnalysisStep, no decisions, no lineage records — two sanctioned front doors, one keeps no records.
11. GATE-TLINE-59b — `ds-pipeline decide` branch unpinned end-to-end at CLI level (record tested directly; main() wiring never driven).
12. F-SURF-1 — plot module computes: qq.py z-standardization + probplot fit from raw arrays instead of reading persisted normality evidence.
13. F-SURF-1b — render-time stat not persisted: pooled-skew raw-vs-log layout choice unverifiable from artifacts (NormalityEvidence lacks it).
14. F-SURF-2 — duplicate computation: plot_describe_figures re-derives groups although caller persisted GroupSummary. [root-family shared with #15's redundancy rider in B#15]
15. F-SURF-3 — two renderers, one concept: production-dead reports/index.py render_index probes wrong filenames; if ever wired → refusal condition (two writers on results/index.md).
16. F-SURF-4 — .svg/.jpg coverage gap: surface-integrity caps cover .html/.png only; an .svg lands git-tracked with NO size gate.
17. F-SURF-5 — viz.yaml drives every figure FILENAME but no knob suppresses emission; figures regenerate unconditionally.
18. SURFACE-MAP watch — shared savefig helper _plot_raw_log_pairs serves surfaces 18/21 through different chains: one refactor from a collision (refusal-condition near-miss).
19. GATE-CFG-70a — "YAML = SSOT, no get-defaults" doctrine has ZERO enforcement tests; violations exist in-tree (samples/loader.py:38,41; builders.py kw.get defaults).
20. GATE-CFG-70b — BROADWAY_CONFIGS_DIR silently repoints EVERY load at a foreign tree, no warning.
21. GATE-CFG-71a — unset-variable LITERAL PASSTHROUGH: expandvars leaves ${VAR} verbatim; nothing scans resolved tree for residual ${; data_dir/password/tracking_uri become literal strings; loud only by int-coercion accident. [dedupe: SAME ROOT as deploy-diff F-2 in D — ruling here owns the resolver fix; D rows rule deployment-side consequences]
22. GATE-CFG-71b — resolver module ZERO test coverage (interpolation, passthrough, merge ordering).
23. GATE-CFG-72 — deep-merge precedence + list-replacement semantics UNTESTED; merge-before-resolve order correct today, unpinned tomorrow.
24. GATE-CFG-74a — strictness asymmetry: DatasetContract/EnvironmentConfig lack extra='forbid' while five siblings have it; typo'd yaml key silently ignored at exactly the truth-pinning layer.
25. GATE-CFG-74b — five dead EnvironmentConfig fields, zero consumers; production.yaml hardcodes sample sizes to 0 with no ge constraint; D11 already rules straight delete.
26. GATE-CFG-75 — silent-degrade dtype-map tail: unrecognized dtype string → pa.Object validates almost anything; boundary gate quietly loses teeth per column.
27. GATE-CFG-76 — B1 include-validation NOT LANDED (queued slate row): typo'd include name SILENTLY SKIPPED, shrinking the feature surface invisibly; no tripwire pins the silence either.
28. GATE-CFG-79 — C1 still queued: boundary tripwire hand-maintains its own _SAMPLE_SCHEMA literal instead of deriving from shipped config — second declaration surface, drift invisible.

Dedupe decisions: #21 ≡ deploy-diff F-2 (D): single root (expandvars passthrough + no residual-${ scan); C owns resolver-level ruling, D carries k8s/env consequence rows cross-referenced — do not double-fix. det-ledger (b) onboard literal-42 violates the same YAML-SSOT doctrine as #19 but lives in D per lane custody; cross-ref only. audit.py/qq.py optional render branches (STATE backlog) folded into #17's family (unconditional figure emission). Band 03's FEAT F5 ordered/unordered schema pairing is A/B custody — not duplicated here despite config adjacency.

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
  BOARD (#5). Slated-with-diffs or deferred-with-mapping; no middle state.
- Show files × lines arithmetic or the fix is not slateable.

Slate candidates screened (senior still rules):
- #9: add 'coercion' to KIND_LABELS (lineage/graph.py) + LINEAGE_STEPS (state.py) +
  extend tests/test_lineage_state.py pin ≈10 lines across 3 files,
  acceptance `uv run pytest tests/test_lineage_state.py tests/test_lineage_graph.py -q`.
- #26+#27: tripwire-first pair — unknown-dtype-string raises at pandera_dtype derivation +
  unresolved include name raises in build_generic_feature_specs ≈30 lines
  (tests/test_boundary_contracts.py append + 2 src guards). If B1 full contract lands better
  via queued slate row ⇒ MODIFY to that end-state or DEFER with board mapping.

## MANDATE REMINDERS (non-negotiable)

- `root:` lines are MANDATORY on every block (MAIN_AGENT_CONTRACT §14 ROOT-CAUSE MANDATE).
- Lukewarm rulings FAIL the mandate — commit to a verdict class and defend it, or ESCALATE
  explicitly per SENIOR vocabulary.
- Human-only calls MUST be flagged `HUMAN-CALL` (e.g., declaring project/scripts record-free by
  contract, deleting dead EnvironmentConfig fields on D11 authority) rather than deferred silently.
