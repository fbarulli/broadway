# PACKET E — GOVERNANCE-TEETH (band 09-infra · tripwire gap list · R11 proactive register · board process)

- Packet id: `E-governance-teeth` · findings: **29** (18 gate-band + 7 gap-list + 4
  R11/board rows; dedupes noted inline)
- Sources: agents/ledger/GATES.md band 09 (:1121–1245) · factsheets/2026-08-24-tripwire-complete.md
  §gap list items 1–7 · R11 proactive register (board row, per organizer brief) ·
  agents/ledger/FIXES.md incident logs · agents/ledger/STATE.md §EVENTS/hazards ·
  agents/ledger/DECISIONS.md D16/D16a/D17/D21/D22/D24/D25/D26 context.
- Required reading before ruling: GATES.md band 09 end-to-end; tripwire-complete factsheet;
  FIXES.md:119–220 incident logs; STATE.md:94–136.

## FINDING REGISTER — BAND 09-INFRA

1. INFRA-90a — no test executes run_local_ci.sh end-to-end (tier parsing, --static/--tier suppression, run() aggregation/banner all zero-coverage; tests read its TEXT only).
2. INFRA-90b — `--static` alone skips pytest+cov on a nominally full tier; "static is for iteration not landing" is contract PROSE, unenforced by code.
3. INFRA-91 — F1b validation extracts-and-executes the checker body against a PATH-shimmed git; the true `git show origin/sklearn:…` network path never exercised in CI.
4. INFRA-92a — mypy covers src/broadway ONLY; ruff just two project/ files — project/etl/process.py etc. lint/type-check in NO tier.
5. INFRA-92b — pytest runs tests/ only; project/tests/ never executes in any tier nor ci.yml (untracked test_ingest_*.py sit outside every gate TODAY). [dedupe: SAME finding as R11 queued tripwire 'project/tests CI scope' — one ruling surface]
6. INFRA-92c — probes validate the coverage-floor NUMBER as text; nothing proves a sub-95 run actually fails.
7. INFRA-93a — test_scripts_diff_empty_vs_main era-gates itself OUT during dev era; stock check() comparison path never executes in dev-era CI.
8. INFRA-93b — sync_to_main() checkout/rm/deletion-mirror machinery has zero test coverage anywhere.
9. INFRA-94 — anchor shape/resolution guards and both dev-era pass-along guards asserted nowhere (non-ancestor taxi / drifted taxi tip never fed through dispatch).
10. INFRA-95a — custody() ZERO coverage: verified live that main==anchor today so BOTH layers shortcut-bypass on every run — correct-by-construction but unproven.
11. INFRA-95b — stale-pin failure direction documented safe yet unpinned by any test.
12. INFRA-96a — ship.sh has zero automated validation; its exit-code law lives entirely in contract prose + DECISIONS.
13. INFRA-96b — .git/hooks/pre-push is UNTRACKED machine-local under .git/ (core.hooksPath unset): every fresh clone ships with NO local gate; "tracked counterpart" claim is procedural only.
14. INFRA-97 — determinism comparator well-tested but the --run runner has none, AND the script is wired into NO tier of run_local_ci.sh nor ci.yml — determinism checked only when someone remembers.
15. INFRA-98 — nothing enforces that commits actually CARRY a computed `Tier:` trailer; classifier classifies when invoked, no gate invokes it on staging. [dedupe: ≡ tripwire gap #3 enforcement half]
16. INFRA-99a — kubeconform scope hardcoded to k8s/optuna/; a future sibling directory under k8s/ silently unscanned. [cross-ref D DEP-M9 top-level path unvalidated]
17. INFRA-99b — cancel-in-progress × D25 push-always: rapid pushes structurally make a superseded tip's CI verdict unobservable; D25 compensation is procedural, not technical.
18. INFRA-99c — no test reads or validates ci.yml itself (delegation law, pinned SHAs, job wiring all unguarded).

## FINDING REGISTER — TRIPWIRE GAP LIST (prose-only lessons → teeth)

19. GAP-1 — Rogue staging ×3 strikes guarded by prose only. Candidate: gate fails if index non-empty / contains entries absent from HEAD at gate time. DESIGN NOTE for senior: sanctioned contracts DO stage exact `git add` lines — hard-fail would break lawful staging; needs allowlist/ack-line design, likely >window ⇒ probable DEFER-to-design.
20. GAP-2 — Fabricated confirmation channels ("user-confirmed set") prose-only. Candidate: such claims must cite a probe-C-resolvable event-id row or the block is invalid.
21. GAP-3 — Review-depth bypass (`Reviewer: none` shipped) partial→GAP. Candidate: pre-push/CI step rejecting commits lacking valid `Tier:` trailer; FULL tier requiring resolvable verdict id. [enforcement twin of #15]
22. GAP-4 — Event-id recomputation / store-then-hash verifier absent (zero recompute/sha8 hits in scripts/tests/src). Candidate: probe fetching each registry comment-id via gh api, byte-verifying sha8. [dedupe: IS the board-process store-then-hash verifier item — single entry]
23. GAP-5 — Foreign-WIP false-red/green: ruff scans TREE not commit (push-on-red root). Candidate: clean-snapshot lint/typecheck mode.
24. GAP-6 — Post-landing ruling conformance (generic D16 drift lesson). Candidate: machine-readable acceptance lines in DECISIONS rows + asserting probe.
25. GAP-7 — Self-directed initiative after mandate (phantom-channel). WEAK candidate only: report-schema "post-mandate actions: none" declaration cross-checked against registry.

## FINDING REGISTER — R11 PROACTIVE REGISTER + BOARD PROCESS

26. R11-teeth-map — Gap items 1–7 must each receive an explicit TEETH THRESHOLD mapping (D1–D7 slots of the register): which gap graduates to an enforcing gate now, which stays probe-level, which is accepted-risk-with-documentation. Senior rules threshold-by-threshold; empty slot = lukewarm = fails mandate.
27. R11-queued-tripwires — three queued tripwires need adopt/defer rulings: dead-code census · .gitignore pin (ignore-file drift guard, pairs with SURF-68 convention-only enforcement) · tamper lock (records-of-record write protection; pairs with C#7/#8 lineage writer gaps).
28. R11-priorities — mechanization priority call: the 32-rule rubric (SENIOR contract Q1–Q3 doctrine) cannot stay hand-applied forever; rule its mechanization order vs GAP items. Plus project/tests CI scope already deduped into #5.
29. BOARD-genesis — genesis event-id recomputation FAILED against stored bodies (13 byte variants, zero matches; ids hashed over pre-posting drafts); six pilot rows void-until-repinned; supersede-and-repin vs recipe-amendment explicitly filed as HUMAN CALL (anomaly row ae44dbfd). Store-then-hash posting is the interim control (#22 verifies it mechanically).

Dedupe decisions: #15≡GAP-3 enforcement half (single ruling, cite both). #5≡R11 project/tests CI scope (single ruling). GAP-4≡board-process store-then-hash verifier (single entry, #22/#29 are verifier-build vs pilot-row-repair halves — distinct asks, cross-linked not merged). Novel-blob false-positive incident: CLOSED by freeze-intact shortcut (FIXES.md:208–220); residual risk is exactly #10's missing custody() coverage — no separate row. D16 landing-drift incident closed same-hour; its generic lesson is GAP-6 only.

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
- #18: new tests/test_ci_yml_probe.py asserting ci.yml delegates platform gates to
  scripts/run_local_ci.sh (guards the delegation law textually) ≈20 lines,
  acceptance `uv run pytest tests/test_ci_yml_probe.py -q`. Test-first tooth for INFRA-99c;
  execution-level coverage stays DEFER.
- #21/#15: Tier-trailer pre-push rejection — DESIGN exceeds window (hook is machine-local,
  CI-side needs trailer grammar ratified first) ⇒ expected DEFER with board row naming the
  trailer grammar decision; do NOT ship a half-gate this window.
- #23: clean-snapshot lint mode — flag design in run_local_ci.sh plausible but interacts with
  #2/--static discipline; rule MODIFY-with-spec or DEFER, never a silent skip change.

## MANDATE REMINDERS (non-negotiable)

- `root:` lines are MANDATORY on every block (MAIN_AGENT_CONTRACT §14 ROOT-CAUSE MANDATE).
- Lukewarm rulings FAIL the mandate — commit to a verdict class and defend it, or ESCALATE
  explicitly per SENIOR vocabulary. "Prose-only lesson, keep as prose" is a REJECT that must be
  argued, not a default.
- Human-only calls MUST be flagged `HUMAN-CALL` rather than deferred silently. THIS PACKET'S
  STANDING EXAMPLE: genesis event-id repair (#29) is explicitly human-owned (supersede-vs-amend,
  ae44dbfd); routing it to a DEFER board row without the HUMAN-CALL flag violates the mandate.
