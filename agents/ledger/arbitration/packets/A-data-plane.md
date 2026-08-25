# PACKET A — DATA-PLANE (bands 01-ingest · 02-etl-lookup · 03-features)

- Packet id: `A-data-plane` · findings: **23** (21 gate-band + 2 backlog adds)
- Sources: agents/ledger/GATES.md bands 01/02/03 · agents/ledger/FIXES.md §Unscheduled
  findings/backlog · agents/ledger/STATE.md §Backlog/hazards · cross-refs into
  factsheets 2026-08-24-{det-ledger,perf-baseline}.md and packet D.
- Required reading before ruling: GATES.md lines 23–381 (band entries + findings),
  FIXES.md:45–55, STATE.md:89–92.

## FINDING REGISTER

1. GATE-INGEST-01 — get_raw_files FileNotFoundError branch has NO test anywhere; hardcoded `yellow_tripdata_*.parquet` prefix means no contract-declared dataset can ever route through this discovery gate.
2. GATE-INGEST-02 — unsupported-extension ValueError branch (loader.py:127) zero coverage; legacy twin read_raw_data bypasses the whole gate (no extension check, no merges, no collision guard).
3. GATE-INGEST-04 — legacy select_and_clean duplicates drop-semantics with DIFFERENT order vs canonicalize (dropna→dedup vs dedup→…→target-null); rename_map reaches code outside DatasetContract, drift caught only reactively.
4. GATE-INGEST-05 — unparseable datetimes become NaT, recorded not raised; nullable=True everywhere lets NaT-filled rows pass the dtype gate silently into canonical parquet. [root-family: see #6/#7]
5. GATE-INGEST-06 — fractional-refusal (FX-A05) reachable ONLY on NaN-free columns; mixed garbage+fractional columns degrade to float64-survives; LEGACY pipeline imports no structural primitive at all.
6. GATE-INGEST-07 — GUARANTEE NOBODY CHECKS: ParseFailure channel recorded-not-enforced; nothing raises on non-empty parse_failures; thousands of bad timestamps still exit green.
7. GATE-INGEST-08 — locked canonicalize order binds CONTRACT pipeline only; LEGACY enforces its own order; sole shared invariant is the textual '-N rows' grammar, validated nowhere at runtime. [partial overlap #12]
8. GATE-INGEST-09 — schema checks NAMES+DTYPES only; empty contract.columns builds a vacuous validating-any-frame schema, untested; legacy validate runs pre-write with no re-validation after write.
9. GATE-ETL-11 — NA semantics asymmetric across one merge: main frame uses pandas default NA set, lookup side keep_default_na=False; literal "NA" is NaN left / string right; audit matching diverges silently.
10. GATE-ETL-14 — lookup-value audit is EVIDENCE-ONLY: no threshold consumes affected_rate/sentinel_counts; a fully-null lookup passes etl.run silently.
11. GATE-ETL-15 — coercion evidence PARTIAL by construction: only int-target astype-backs recorded; float drift + datetime coerce emit nothing; rows_affected = whole column len; persistence branch untested.
12. GATE-ETL-18 — the "shared grammar" regex exists as two INDEPENDENT copies (process.py / module.py), no common symbol; editing one silently desyncs explained/unexplained accounting. [root shared with #7]
13. GATE-ETL-16 — POST-WRITE RELOAD VALIDATION DOES NOT EXIST in etl: parquet round-trip dtype drift surfaces only when features/training readers open the file.
14. GATE-ETL-17 — StageLedger stores ROWS-AFTER not deltas, appends zero-drop stages; consumers treating entries as drop counts overcount; no per-column dropna attribution anywhere.
15. GATE-ETL-19 — ingest path NEVER calls enforce_drop_fraction: unexplained loss at project ingest boundary recorded but never loud (only etl/features enforce).
16. FEAT F1 — contract file/symbol drift: builder.py/build_distance_features/'validate_engineered_schema' absent; real owners builders.py + generic.py trio. [absorbs GATE-FEAT-27 symbol-mismatch line]
17. FEAT F2 (GATE-FEAT-20) — log1p numerically pinned only via double-log chaining test; no single-hop golden for BUILDERS['log_distance'].
18. FEAT F3 (GATE-FEAT-23) — TargetEncoding.fit y-is-None raise unpinned by any test node id.
19. FEAT F4 (GATE-FEAT-25a) — ARTIFACT-TRAIN-PARQUET write + features run() have no direct test node id; content pinned transitively only.
20. FEAT F5 (GATE-FEAT-25b) — write-side engineered schema UNORDERED vs read-side ordered=True: column-order drift survives the write gate, explodes later at training/evaluate.
21. FEAT F6 (GATE-FEAT-27) — entire engineered read contract tested only transitively; refactor of generic.py internals leaves no direct failing node id.
22. BACKLOG-FIXES — read_sample bypass: pl.scan_parquet path skips load_with_audit entirely (no audit joins, no label guard). Cross-ref B#TRAIN-32 seed-pin (same loader surface).
23. BACKLOG-FIXES — Option C typed-source hard rejection unscheduled (follow-up to Option E coercion lineage).

Dedupe decisions: #4/#6/#11 kept separate (distinct surfaces: NaT-pass, parse_failures unenforced, coercion partial) but share ONE root family — evidence channels with no consumer/enforcer; rule once, cite thrice. #7/#12 partially overlap (grammar clause) but each carries a distinct claim — NOT merged. Golden-float absence in stats belongs to B (GATE-STATS-45); det-ledger item (e) ULP fragility lives in D — sibling failure modes, do not merge here. log_dataset second-read backlog row deduped into D (perf-baseline phase 5b).

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
  reversible, test-first preferred**. List slate items explicitly under a `## SLATE` heading.
- DEFER = everything else → write `DEFER → board-row: <proposed row id/scope>` mapping to a
  CHANGE BOARD (#5) row. Never blur: an item is either slated with exact diffs+acceptance or
  deferred with its board mapping. No middle state.
- Window-budget arithmetic must be shown (files × lines) or the fix is not slateable.

Slate candidates already screened for window fit (senior still rules):
- #1+#2: new tests/test_raw_discovery.py (FileNotFoundError branch names raw_dir) +
  tests/test_loader_unsupported_ext.py (ValueError branch) — 2 new files ≈30 lines,
  acceptance `uv run pytest tests/test_raw_discovery.py tests/test_loader_unsupported_ext.py -q`.
- #18: append y=None raise pin to tests/test_transformers.py ≈8 lines.
Do NOT propose edits to untracked-WIP files (project/tests/test_ingest_*.py are R2 lane property).

## MANDATE REMINDERS (non-negotiable)

- `root:` lines are MANDATORY on every block (MAIN_AGENT_CONTRACT §14 ROOT-CAUSE MANDATE).
- Lukewarm rulings ("seems fine", "consider later", no-op verdicts) FAIL the mandate — pick a
  verdict class and defend it, or escalate explicitly.
- Human-only calls MUST be flagged `HUMAN-CALL` (e.g., upstream data re-contracting, TLC schema
  renegotiation) rather than deferred silently. DEFER ≠ HUMAN-CALL: defer routes to the board;
  HUMAN-CALL stops for a human decision.
