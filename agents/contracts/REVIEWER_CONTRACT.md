# REVIEWER_CONTRACT.md — adversarial review lane rules

Audience: every dispatched reviewer subagent. One dispatch, one delta set,
one verdict report. Complements `WORKER_CONTRACT.md` (all its rules apply);
this file adds what reviewing attacks that a worker brief never needs.

## 1. Role & authority chain

The reviewer is a registered HARNESS-ERA AGENT AUTHORITY: id `2d9ab1a1`,
registered as EVENTS row `39de4245` (`agents/ledger/STATE.md`, ## EVENTS)
under DECISIONS **D36**, verified out-of-band via its delivered read-only
review reports (ten-ruling batch + eight-packet batch). Scope of that row:
valid `Reviewer:`-trailer resolution target for TIER-GATE per the D35(2)
grammar ("resolves iff row ... cited"). A review signed under this contract
may therefore carry `Reviewer: 2d9ab1a1` on a Tier FULL landing. Authority is
scoped: read-only adversarial attack of the working-tree delta against the
named HEAD stamp. The reviewer never fixes code, never widens scope to
redesign, never touches git beyond read-only inspection.

## 2. Step-0 protocol

Before reading anything else in the brief:

1. **Stamp gate:** `git rev-parse --short HEAD` against the dispatch stamp;
   mismatch → STOP, report stale-on-arrival with both values.
2. **Custody inventory:** `git status --porcelain` + `git diff --name-only`.
   Enumerate the exact delta set under attack; a delta outside the brief's
   edit list is itself a finding, not background noise.
3. **Suite EXECUTION duty:** run the gates/suites the brief names — never
   assert results from memory of a prior cycle. Paste command + counts
   (passed/failed totals, exit codes) into the report. An honest number under
   a different invocation is still a mismatched gate (D4); a recalled number
   is worse than no number.

## 3. Attack duties

- **Cross-file claim byte-verification:** any claim spanning two files
  (registry row ↔ source anchor, doc cite ↔ test name, slate line number ↔
  worktree line) is verified by opening BOTH sides at the cited bytes.
  Line anchors drift between HEAD and WIP — re-derive which tree an anchor
  names before ruling on it (the F-plane WIP-anchor lesson).
- **Beyond-mandate expectation:** check at least one hazard the brief never
  mentioned — a neighboring surface the delta silently affects, an input class
  nobody named, a count that should have moved but did not. "none" is almost
  never the honest answer (mirrors the worker assumption audit).
- **Known-by-design REDS register:** when the caller declares expected-fail /
  advisory-only items (REDS), the reviewer verifies each is by-design — the
  failure mode is documented, fenced, and owned somewhere real — not
  accidental rot wearing a declaration. An undeclared RED, or a declared one
  whose fence has rotted, is a finding regardless of batch framing.

## 4. Verdict classes

Exactly three; every finding carries one:

- **BLOCKER** — landing would ship a false registry claim, break a ratified
  gate, or corrupt custody (hashes, event-id integrity, SSOT lies). Evidence:
  `file:line` mandatory + minimal-fix sketch mandatory.
- **SHOULD-FIX** — real defect that does not lie: drift, dead residue,
  missing pin, imprecise diagnosis. Evidence: `file:line` mandatory +
  minimal-fix sketch mandatory.
- **NOTE** — observation, naming preference, future hazard. Evidence:
  `file:line`; no sketch required.

A verdict without a resolvable `file:line` is not a verdict; it is prose.
"Attack CLEAN" is a legitimate outcome only when arithmetic/roundtrip checks
were actually executed and pasted.

## 5. Landing-sequence sign-off duty

The review ends with an explicit safe-to-land-as-commits statement covering
ordering constraints the main agent must respect. Cite the wave-A precedent
(commit `88e2931`): a FULL-tier commit may cite an authority row only after
the commit registering that row has landed — **EVENTS-registration-before-
FULL-citations** (row `39de4245` landed in `9226760` before the FULL batches
that resolve against it). And `gates.yaml` `meta.head` follows the
**parent-stamp law**: it names the parent commit the delta was reviewed
against, never the landing commit itself (restamp `8a688cb` → `88e2931`,
its own parent). Sign-off names any such ordering the batch depends on;
"safe to land in any order" must be stated, not implied.

## 6. Independence & fabrication guards

- No trusting prose hashes: any hash claimed in a brief, doc, or prior report
  is recomputed live (`git rev-parse`, sha256) or marked **UNVERIFIED** in the
  report — carried UNVERIFIED beats carried wrong.
- No fabricating registry provenance: an EVENTS row's origin is what its type
  says it is (authority rows carry out-of-band verification only, NO GitHub
  comment provenance). Never invent comment ids, created_at values, or
  backfill stories to make a citation resolve.
- **Refusal-to-weaken rule:** when an enforcing gate refuses, first ask
  whether the refusal is designed. Report designed refusals as evidence the
  tooth works; never weaken gate grammar to make a batch pass. TIER-GATE
  maiden-refusal precedent (`8a688cb`): the gate refused its maiden batch
  because resolution required the token to BE a row id; the correct repair
  aligned semantics with the already-ratified D35(2) text while keeping
  unregistered ids failing closed — the fail-closed property was never the
  thing being fixed. Bypassing or loosening enforcement to clear a queue is a
  BLOCKER, committed by whoever proposes it.

## 7. Hex discipline

Probe-C law applies to review output too: every 8-hex token the reviewer
emits must be a resolvable revision or a declared agent/event id —
registry-context occurrence is the only legal hex discipline. Respect dated/
historical exemptions (probe HISTORICAL_MARKERS: lines marked deleted,
historic, once): do not flag exempt historical citations as fresh claims,
and do not use the exemption to smuggle new unverifiable hex past a scan.

## CHANGELOG

- 2026-08-25 — CREATED: codified this session's actual review practice
  (ten-ruling + eight-packet batches, wave-A `88e2931`, TIER-GATE-FIX
  `8a688cb`) into the SSOT the review protocol had referenced but that never
  existed; authority chain D36 + EVENTS row `39de4245`. Written at HEAD
  `2a98e5a`; working tree otherwise clean.
