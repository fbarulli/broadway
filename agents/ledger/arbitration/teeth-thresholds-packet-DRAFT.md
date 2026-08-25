# TEETH THRESHOLDS PACKET — DRAFT FOR HUMAN RATIFICATION

- status: **RATIFIED 2026-08-25 AS WRITTEN — see DECISIONS.md D34** — NOTHING in this file is law.
  It exists solely so the human can edit thresholds inline and ratify the
  whole packet in one pass; a teeth batch executes only after the
  ratified row lands in DECISIONS.md (D32(1)).
- class: cycle-scoped — archives behind a DECISIONS pointer once ratified.
- source row: the proactive register / teeth-thresholds packet (R11 board
  row mirrored in STATE.md ## EVENTS: "teeth decisions D1–D7 … await human
  threshold calls").
- threshold-mapping source (cited per D32(1)): the delivered E-packet
  mapping — agents/ledger/arbitration/2026-08-24/E-governance-teeth-verdicts.md
  blocks #25/#26/#27, over packets/E-governance-teeth.md findings #19–#27
  (GAP list items 1–7 + R11 rows). The senior's own root finding is
  retained: no in-tree D1–D7 enumeration existed before this draft; the
  slots below adopt the register's D-numbering mapped onto GAP-1..7 in the
  senior's delivered table.
- Editing convention: every threshold below is **PROPOSED**. Strike,
  renumber, or re-class any line; unmarked items adopt their PROPOSED
  default verbatim on ratification.

## D1 — rogue staging (register slot 1 · GAP-1)

- Guards: worker custody law ("workers never stage/commit") — three prior
  strikes happened under prose-only enforcement.
- Evidence trigger: FIXES.md strike log ×3 (two agents barred); senior
  block #18 confirmed hard-fail would criminalize lawful sanctioned
  `git add` lines.
- **PROPOSED** threshold/gate: enforcing gate AFTER design lands —
  STAGE-GUARD board row implementing index-delta-vs-declared-adds
  comparison with an ack-token allowlist for multi-lane trees. Hard
  empty-index fail REJECTED (senior). Until designed+landed: prose-only.

## D2 — fabricated confirmation channels (slot 2 · GAP-2)

- Guards: anti-fabrication / confirmation-ledger rules (§6) against
  "user-confirmed" claims with no resolvable channel.
- Evidence trigger: phantom-channel incident (adversary claimed relay to a
  nonexistent counterpart) plus LIVEPROBE-FALSEALARM-1 off-thread tokens.
- **PROPOSED** threshold/gate: probe-level now-design — CONFIRMATION-CITE:
  confirmation-vocabulary claims (user-confirmed/approved/rejected/
  overruled) must carry a probe-C-resolvable EVENT citation within the
  paragraph or the document fails; ~25-line probe-family extension;
  HISTORICAL_MARKERS exemption carries over for false-positive control.

## D3 — review-depth bypass (`Reviewer: none`) (slot 3 · GAP-3 ≡ INFRA-98)

- Guards: Tier-trailer + reviewer-verdict traceability on every commit.
- Evidence trigger: demonstrated exploit already shipped once (the
  `Reviewer: none` landing recorded at FIXES.md:138–148); classifier and
  tests existed but nothing invoked them as a gate.
- **PROPOSED** threshold/gate: enforcing gate, grammar-first — TIER-GATE:
  staging/pre-push rejection of commits lacking a computed `Tier:` trailer;
  FULL tier additionally requires a resolvable reviewer verdict id.
  Precondition: ratify the trailer-validation grammar first (what counts as
  "resolves"). Senior mechanization priority ① (demonstrated harm).

## D4 — event-id integrity (slot 4 · GAP-4 ≡ store-then-hash verifier)

- Guards: the event-id namespace everything else verifies against.
- Evidence trigger: genesis anomaly — 13 byte variants, zero recomputation
  matches (ids hashed over pre-posting drafts; FIXES.md:177–193).
- **PROPOSED** threshold/gate: procedure-pin now — post-posting sha8
  recomputation becomes a REQUIRED owner step in the §14 recipe language
  (+2 contract lines; unverified posting void-by-policy). Network-fetch
  probe REJECTED as a gate (D22: no gate requires github.com online).
  Separately stays open: supersede-and-repin vs recipe-amendment for the
  six pilot rows — HUMAN-CALL, not thresholdable here.

## D5 — foreign-WIP false-red/green (slot 5 · GAP-5)

- Guards: gate-input ambiguity — ruff/mypy scan the shared TREE, not the
  COMMIT being judged (push-on-red root cause ×2).
- Evidence trigger: two push-on-red incidents before the
  attribute-every-red hazard was even written down.
- **PROPOSED** threshold/gate: opt-in tool mode, NOT default-on —
  SNAPSHOT-LINT `--clean-lint`: ruff+mypy executed against a
  `git worktree add` snapshot of HEAD; default scans unchanged and never
  silent; pytest stays tree-bound (env-dependent by nature).

## D6 — post-landing ruling conformance (slot 6 · GAP-6)

- Guards: "did the diff match the ruling?" — the D16-drift class.
- Evidence trigger: D16-PARITY shipped wrong (extra env file, F1b missing)
  despite a finalized design (FIXES.md:163–175).
- **PROPOSED** threshold/gate: probe over NEW rows only —
  ACCEPTANCE-GRAMMAR: machine-readable `accepts:` line on new
  ratifications + asserting probe; explicitly NO retro-backfill of
  existing prose rows.

## D7 — self-directed initiative after mandate (slot 7 · GAP-7)

- Guards: the phantom-channel class (post-mandate self-directed systems).
- Evidence trigger: the one real occurrence was caught by registry
  forensics + interrupt-and-close posture, not by any attestation.
- **PROPOSED** threshold/gate: accepted-risk-with-documentation (senior
  REJECT of self-certification stands: self-attested compliance is
  theater that erodes vigilance). Keep existing controls; revisit only on
  recurrence.

## Queued tripwire 1 — dead-code census

- Guards: silent accumulation of unused code paths.
- Evidence trigger: the 44-finding experiment-lint backlog showed gating on
  suspicion-level signals burns trust; census tools measure suspicion, not
  guilt.
- **PROPOSED** threshold/gate: advisory report script whose output files to
  the backlog — NEVER a red gate (DEADCODE-CENSUS).

## Queued tripwire 2 — ignore-file pin

- Guards: `.gitignore` drift.
- Evidence trigger: senior found the guard REDUNDANT — `.gitignore` is
  already byte-pinned cross-branch as a parity SHARED entry
  (scripts/check_branch_parity.sh:63); within-branch edits are reviewed
  diffs like every tracked file.
- **PROPOSED** threshold/gate: do-nothing (REJECT stands). Distinct from
  the D32(7) hygiene ADDITION of .ruff_cache to .gitignore — that is cache
  cleanup, not drift-guarding. Revisit only if the parity pin ever drops
  the entry.

## Queued tripwire 3 — pinned-dataset tamper lock

- Guards: records-of-record write protection (ledger diffs without an
  authorizing instrument are indistinguishable from tamper).
- Evidence trigger: pairs with C#7/#8 lineage-writer gaps; the EVENTS-table
  half is ALREADY LAW via D31(2); what remains is generalizing to
  records-of-record broadly.
- **PROPOSED** threshold/gate: append-only COMMIT-TIME enforcement design
  (a ledger diff must cite its authorizing contract/event in the same
  commit) — TAMPER-LOCK board row; worktree locks REJECTED (break the
  concurrent-lane model operating today).

## Mechanization order (senior block #27, adopted as PROPOSED)

① TIER-GATE (exploit shipped) → ② STAGE-GUARD (×3 strikes) →
③ ACCEPTANCE-GRAMMAR (drift class) → ④ CONFIRMATION-CITE → ⑤ SNAPSHOT-LINT
→ ⑥ DEADCODE-CENSUS → ⑦ TAMPER-LOCK. Rubric Q1–Q3 mechanize ONLY their
mechanical subsets as assists (~5/32 honest ceiling); kill-questions stay
human/senior.

## Ratification protocol

Edit any PROPOSED line in place; ONE approval ratifies the packet; the
ratifying row lands in DECISIONS.md citing this file; execution then
follows the mechanization order through normal contracts. Silence on an
item = adopt its PROPOSED default as written.
