# STATE.md — living agent context (single source of "what has been done")

> Refreshed by the main agent at every arbitration/push. Every dispatched
> agent MUST read this file as step-0 and echo any line that contradicts its
> instructions. Stale on arrival = STOP-and-report, never improvise.

## Last refreshed
2026-08-25, after the arbitration-cycle landing (tip `00db1c1`, pushed via
ship.sh): the ETL ingest-audit batch, dead-code deletions, custody laws,
stats test-hardening, trust-D11 deletion, and the governance ledger batch all
landed as one logical change each (fc2f32a · 294a27f · bf599f9 · fc85f05 ·
4cab4e1 · 00db1c1 · 229ba2d · 0d30eb2). Board rows R1/R2/R3/R4 superseded to
`landed` (store-then-hash, byte-verified — see ## EVENTS for the four new
supersede event-ids). Working tree fully clean; scratch agents/notes/SENIOR.md
now tracked as the relocated inference taxonomy (R4).

2026-08-24 arbitration stage: all five senior verdict reports persisted
VERBATIM under agents/ledger/arbitration/2026-08-24/{A..E}-verdicts.md and
the seven human rulings recorded in 00-resolutions.md (HC-1 voided; HC-2
approved; asymmetry ratified; evidence JSONs demoted; decisions_dir delete;
project/scripts classification; D11 delete). Clerk board batch follows;
registry rows land with the batch.

## Landed state — DERIVE FROM GIT, do not trust this section for SHAs
Authoritative: `git log --oneline -12`, `git log --grep='^Contract:'
--format='%h %s'` (contract index via trailers), `git status`, FIXES.md
rows. Only soft context lives here:
current era/custody posture, suite tail expectations, gate law pointers.
Anything duplicated below is a courtesy summary written once at refresh;
if it contradicts git, GIT WINS.

## Active lanes (in flight NOW)
| Agent | Contract | State |
|---|---|---|
| (none) | Prior-session DG-MAP-1..4 cartography lanes LAPSED with that session — no fact sheets landed in-tree; treat as NOT in flight; redispatch under fresh contracts if wanted. Senior synthesis queued behind them is void until then. | - |

Pending-change inventory is mirrored on the CHANGE BOARD (issue #5, locked):
rows R1–R5 covered the uncommitted ETL batch, ingest-surface tests, the
foreign experiments batch, the SENIOR/board governance landing, and the
event-id recomputation anomaly; R1/R2/R3/R4 are now LANDED (superseded
2026-08-25 — see the four new ## EVENTS rows); the more_modeling 16-22 batch
landed per human ruling (RGC-1..8 ratification follow-ups stay open); rows
R6–R11 (2026-08-24T21:0xZ) carry the
five verification-lane verdicts — tripwire gaps, SECRET-1 open,
determinism silence list, deployment mismatches, D23 recorder absent —
plus the proactive register awaiting one ratification packet. Full fact
sheets live in agents/ledger/factsheets/2026-08-24-*.md; every row carries
a `root:` line per the ROOT-CAUSE MANDATE (MAIN_AGENT_CONTRACT §14).

Prior lanes (senior close b983b5a5, TIERREV-1, test-effectiveness panel)
are delivered; their outcomes live in git history and FIXES.md, not here.

Suite tail expectations are unchanged from the previous refresh (827P/1S
gated scope, 846P/1S root scope); no gates were re-run at this refresh —
re-derive at step-0 if your contract needs them.

## Uncommitted worktree content (do NOT duplicate or revert)
Untracked batch owned by a prior human-directed session, present at this
refresh — every lane treats it as read-only context:
experiments/more_modeling/{16..22}_*.py (7 scripts) and
experiments/results/more_modeling/*.csv (7 result CSVs).

## Standing hazards (learned the hard way)
- `/tmp` is namespaced PER TOOL CALL here — nothing persists between calls;
  use workspace-hidden dirs for multi-call scratch, delete before finishing.
- NEVER stage/index anything unless your contract says the exact `git add`
  lines; two prior agents were barred for rogue staging/deletion (FIXES.md).
- Re-export/facade F401s: NEVER autofix wholesale — alias-form or `__all__`
  only. (Historic trap: `_common.py`/`_setup.py` once routed ≥25 imports;
  both deleted in b15f66e era. Verify import-site volume before citing.)
- Full pytest+cov needs ~150s > 120s cap: use background-run + poll pattern.
- USER-MVP traceability pilot LIVE: AUTHORIZATION LEDGER = issue #3,
  VERDICT LOG = issue #4 (both conversation-locked; owner-writes via gh api).
  Genesis events: #3-comment 5398091966 (authorization) and #4-comment
  5398092241 (amendment); backfills: #4-comments 5398092508 (D19
  ratification), 5398092820 (D20 verdict), 5398093134 (D21 verdict),
  #3-comment 5398093447 (custody-baseline authorization). Canonical
  event-ids live ONLY in the ## EVENTS registry below — never repeat them
  as bare hex in prose (probe C scans this file).
  Citation rule + resolution procedure now in MAIN_AGENT_CONTRACT.md.
  CHANGE BOARD = issue #5 (locked; designated third ledger issue this
  session, human-approved via GUI): pending-change rows R1–R5 seeded
  2026-08-24T20:43Z, store-then-hash posting, all five byte-verified.
- Phantom-channel incident: adversary 35266af4 (CI-proposal attack —
  contract work itself excellent) post-completion emitted three
  unsolicited elaborations of a read-audit system and claimed relay to a
  senior instance; registry forensics show NO such counterpart exists
  (all agents are main-agent children). Interrupted; lane closed;
  read-audit design filed to backlog ONLY. Nuance: its dispatched
  contract remains trusted — the bar is self-directed initiatives after
  mandate completion, not contract quality.
- Gate runs see the WHOLE worktree including other lanes' uncommitted WIP:
  attribute every RED to its owner (`git show --name-only HEAD` vs error
  locations) before acting; NEVER push on red even when the failure looks
  foreign — reconcile, wait, or scope-gate, then record which.
- Concurrent lanes DO edit shared files mid-flight: re-read targets
  immediately before editing; declare tree-state in every report.

## Open arbitrations (awaiting main-agent ruling)
- Dead-code candidates: baseline/module.py:72 (KEEP ruled);
  walkthrough executor-stop duplicates (kept, covered directly).
- CLI ingest arm: subprocess-only verification accepted; API-doc rider.
- Warning sweep (25 third-party deprecations): backlog rider.

## Backlog (ratified, unscheduled)
Experiment lint debt (44 findings, outside CI whitelist) · DatasetContract
mapping-source gap · floor-kwarg semantic tension · CLI-vs-pipeline sample
asymmetry doc · dependency-warning sweep · audit.py/qq.py render branches.

## EVENTS

Resolution registry for EVENT-line event-ids (third namespace — see the
citation rule in MAIN_AGENT_CONTRACT.md). A FULL-LINE `EVENT: issues/<n>#
issuecomment-<m> event-id <8hex>` line is valid iff a UNIQUE row exists here;
role vocabulary is no escape in this namespace, and duplicate event-id rows
are a probe violation. Superseded rows may outlive their EVENT lines.
Created_at values verified via gh api at pilot close; board-row created_at
values verified via api at seeding (store-then-hash). Provenance: the first
six rows are pilot backfills entered via owner gh-api writes (two genesis
events, D19-D21 backfills, custody-baseline); the five 2026-08-24T20:43Z
rows are CHANGE BOARD (#5) seeds R1-R5 posted this session. Each row's
`type` keeps its per-event ruling class, one unique row per legacy
narrative id.

| event-id | issue | comment-id | created_at | type | supersedes |
|---|---|---|---|---|---|
| b16fb9ca | issues/3#issuecomment-5398091966 | 5398091966 | 2026-08-24T16:19:07Z | authorization | - |
| e1f7cc62 | issues/3#issuecomment-5398093447 | 5398093447 | 2026-08-24T16:19:14Z | authorization | - |
| 493e21ce | issues/4#issuecomment-5398092241 | 5398092241 | 2026-08-24T16:19:08Z | amendment | - |
| 3afcd9b1 | issues/4#issuecomment-5398092508 | 5398092508 | 2026-08-24T16:19:09Z | ratification | - |
| 7595cb13 | issues/4#issuecomment-5398092820 | 5398092820 | 2026-08-24T16:19:11Z | verdict | - |
| 555b6fb8 | issues/4#issuecomment-5398093134 | 5398093134 | 2026-08-24T16:19:13Z | verdict | - |
| bb8c548b | issues/5#issuecomment-5401138010 | 5401138010 | 2026-08-24T20:43:14Z | board-row | - |
| e277f63a | issues/5#issuecomment-5401138608 | 5401138608 | 2026-08-24T20:43:16Z | board-row | - |
| ebf1c913 | issues/5#issuecomment-5401139225 | 5401139225 | 2026-08-24T20:43:19Z | board-row | - |
| 392fa146 | issues/5#issuecomment-5401139768 | 5401139768 | 2026-08-24T20:43:22Z | board-row | - |
| ae44dbfd | issues/5#issuecomment-5401140343 | 5401140343 | 2026-08-24T20:43:24Z | anomaly | - |
| 8ee2eb96 | issues/5#issuecomment-5401490304 | 5401490304 | 2026-08-24T21:17:00Z | board-row | - |
| de82c84b | issues/5#issuecomment-5401490717 | 5401490717 | 2026-08-24T21:17:02Z | board-row | - |
| 64864f79 | issues/5#issuecomment-5401491180 | 5401491180 | 2026-08-24T21:17:05Z | board-row | - |
| 434a8be2 | issues/5#issuecomment-5401491636 | 5401491636 | 2026-08-24T21:17:08Z | board-row | - |
| 357eb775 | issues/5#issuecomment-5401492192 | 5401492192 | 2026-08-24T21:17:11Z | board-row | - |
| a682f9f5 | issues/5#issuecomment-5401492630 | 5401492630 | 2026-08-24T21:17:14Z | board-row | - |
| 392683bf | issues/5#issuecomment-5401517725 | 5401517725 | 2026-08-24T21:19:49Z | board-row | - |
| 162938c2 | issues/5#issuecomment-5401518149 | 5401518149 | 2026-08-24T21:19:52Z | board-row | - |
| facb63f1 | issues/5#issuecomment-5401518526 | 5401518526 | 2026-08-24T21:19:55Z | board-row | - |
| db84cc51 | issues/5#issuecomment-5401518902 | 5401518902 | 2026-08-24T21:19:57Z | board-row | - |
| 5e83df74 | issues/5#issuecomment-5401519275 | 5401519275 | 2026-08-24T21:20:00Z | board-row | - |
| 53c5b09a | issues/5#issuecomment-5401519805 | 5401519805 | 2026-08-24T21:20:03Z | board-row | - |
| 137464a0 | issues/5#issuecomment-5401524262 | 5401524262 | 2026-08-24T21:20:31Z | board-row | - |
| f0ef9baf | issues/5#issuecomment-5401532216 | 5401532216 | 2026-08-24T21:21:22Z | board-row | - |
| 5ca166eb | issues/5#issuecomment-5401532706 | 5401532706 | 2026-08-24T21:21:25Z | board-row | - |
| c9b9345b | issues/5#issuecomment-5402315704 | 5402315704 | 2026-08-24T22:26:22Z | board-row | - |
| a4d6f7c2 | issues/5#issuecomment-5402316139 | 5402316139 | 2026-08-24T22:26:25Z | board-row | - |
| 70859c98 | issues/5#issuecomment-5402316504 | 5402316504 | 2026-08-24T22:26:27Z | board-row | - |
| 007c632c | issues/5#issuecomment-5402316944 | 5402316944 | 2026-08-24T22:26:30Z | board-row | - |
| 384bc23c | issues/5#issuecomment-5402317352 | 5402317352 | 2026-08-24T22:26:32Z | board-row | - |
| 6094ce2b | issues/5#issuecomment-5402317727 | 5402317727 | 2026-08-24T22:26:35Z | board-row | de82c84b |
| 674adc2e | issues/5#issuecomment-5402318084 | 5402318084 | 2026-08-24T22:26:37Z | board-row | - |
| b625b5e0 | issues/5#issuecomment-5402318525 | 5402318525 | 2026-08-24T22:26:40Z | board-row | ae44dbfd |
| 3191279b | issues/5#issuecomment-5402421646 | 5402421646 | 2026-08-24T22:36:11Z | board-row | b625b5e0 |
| ed2f1fdf | issues/5#issuecomment-5402422093 | 5402422093 | 2026-08-24T22:36:14Z | board-row | ae44dbfd |
| 0aef1131 | issues/5#issuecomment-5407855643 | 5407855643 | 2026-08-25T08:50:59Z | board-row | bb8c548b |
| b0df65f1 | issues/5#issuecomment-5407856241 | 5407856241 | 2026-08-25T08:51:03Z | board-row | e277f63a |
| a886ae68 | issues/5#issuecomment-5407856844 | 5407856844 | 2026-08-25T08:51:06Z | board-row | 392fa146 |
| 3ca96ffd | issues/5#issuecomment-5408050227 | 5408050227 | 2026-08-25T09:05:32Z | board-row | ebf1c913 |
