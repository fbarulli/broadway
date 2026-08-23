# DECISIONS.md — Slate v4 decision sheet (D1–D10)

Arbitrated 2026-08-23 after six audit lenses (redundancy, gaps, SSOT,
hardcoded values, contradictions, better-ideas) plus a three-adversary
red-team panel (scope/necessity, sequencing/risk, fact-check) over range
`ea33370..be34c30` and the ratified FIX_4 closure (`79ac26c`).
Fact-check score: 10 CONFIRMED / 0 REFUTED. Sequencing verdict: adopt-revised.

- **D1 · Slate v4 (10 contracts)** — RATIFY: G0a platform-doc deletions;
  G0b governance truth; A1 schema-builder unification (+ riders:
  builders-lambda defaults, `coercion` lineage kind, `.uv-cache` gitignore);
  A2 datetime semantic compare; B1 build-time include-validation
  (`contract ∪ joined-lookup`, raise naming unknowns); B2 parity fail-loud on
  unresolved skips; C1 `_SAMPLE_SCHEMA` derives from config; C2 `etl.run`
  coercion-persistence test + loader docstring fix; C3 derived/encoded
  read-flip coverage (narrowed); D1 rev-parse ref pre-check exit 2.
  Killed by panel: warning-absence pins, SHARED expansion/extraction,
  naming helper, dtype-policy constant, fixture migration, harness dedup,
  congruency-as-doctrine.
- **D2 · Order & isolation** — G0 → A → B → C → D (sequencer-revised):
  A-before-B removes generic.py double-churn; B validated at BUILD time
  (load-time crashes parity collection via the 8-pair cross-product); G0
  lands before any D edit of the checker file.
- **D3 · Authorization mode** — batch: sequential dispatch, each contract
  runs worker → gates → adversarial reviewer → arbitration → commit/push;
  actuals sheet at completion.
- **D4 · Canonical gate invocation** — root `uv run pytest -q` pinned in
  every brief (CI `pytest tests/` quoted secondarily; ±19 difference);
  counts measured at step-0 paste, never projected; suite-total warnings
  NEVER gated (25 occ/7 groups vs ledger "9" — irreproducible); per-path
  pins only.
- **D5 · configs/experiments/mlflow.yaml** — LEAVE DIVERGENT, document it:
  deliberately taxi-purged on main (`4657013`, `a2f26e9`); adding to SHARED
  would let main-day `--sync` clobber main's synthetic variant with ratecode1
  names that do not exist there. Parity only after a data-agnosticize pass.
- **D6 · graph_todo.md** — undocumented working-tree deletion occurred
  2026-08-23; restored by main agent; provenance unclaimed. Stays restored.
- **D7 · HANDOFF final-SHA bookkeeping** — batch SHAs written into HANDOFF
  at slate completion, not before.
- **D8 · Worker evidence format** — codify "paste the command alongside the
  tail line" in WORKER_CONTRACT (lesson: root vs tests/-scope invocation made
  both 783 and 764 honest numbers); rides G0b.
- **D9 · Backlog confirmed parked**: log_dataset in-memory frames when
  lineage wires into training; Option-C typed-source hard rejection;
  composite-key encoding-naming revisit iff a config declares multi-column
  encodings.
- **D10 · Artifact hygiene** — regenerate train_features.parquet via real
  taxi flow after the slate lands (pre-guard artifacts predate A/B changes).

Standing facts: baseline after FIX_4 = 783P/1S/0X root-scope (764P tests/-scope),
1 skip is the PARITY_MAIN_DAY gate; parity gate red-by-design until declared
main-day. Batch lineage: H `3db7b4b` → FIX_1 `c324583` → FIX_2 `3ee1ef5` →
FIX_3 `ca8c123` → FIX_4 `79ac26c` → governance `3ea88d1`/`c34710c`/`be34c30`.

*Authorized for publication by the human operator via GUI session,
2026-08-23 ("push the D1-D10 ... if its easier push on sklearn").*

## D11–D13 — slate v5 rulings (ratified post-ADV-trio)
- **D11 Tier-4 dead-code doctrine**: stub modules consolidate into INVENTORY.md
  (name / one-line intent / why unbuilt) then delete stubs + dead fields in ONE
  commit. Test-only exports and dead EnvironmentConfig fields: straight delete,
  no inventory entry.
- **D12 decision-moment leakage**: ENFORCE gating; drop dropoff_location_id from
  the shipped taxi experiment surface. A contract layer silently overridden by
  config teaches the wrong lesson; amend-the-contract is rejected absent a real
  use-case change.
- **D13 tier order**: Tier1 confirmed bugs → Tier2 config coherence (absorbs
  B1/B2) → Tier3 data-gate hardening → Tier4 (post-D11) → Tier5 bulk.
- **D14 T-bug-1 scope expansion (user spot-check)**: estimation_table() blindly
  trusts the handed model object — HC3_SE *and* CI_low/CI_high mislabel
  nonrobust fits whenever callers don't pre-fit cov_type="HC3". Fix derives HC3
  independently of input fit; landmine test added (plain-fit input ⇒ HC3 ≠ OLS).
- **D15 proof-carrying wrapper layer REJECTED**: scoping agent mapped all nine
  affected findings (T-bug-1, F1, F8/F11/F14/F16, T-bug-4, F4/F19) to
  equal-safety direct fixes inside the existing idiom — pydantic parse
  validation, pandera schemas derived from DatasetContract/LookupSpec,
  config-threaded thresholds with required params (runners.py precedent).
  mypy IS CI-enforced but only sees type structure, not construction-site
  invariants, so wrappers would add a drift-prone second declaration surface
  (contra D5/D6) and nothing else. Tier 3 fixes are written against the
  existing machinery; new sub-finding adopted: stats/module.py silently skips
  empty groups — min_rows_for_sampling exists in config but is unenforced at
  the stats entry point.
