# STATE.md — living agent context (single source of "what has been done")

> Refreshed by the main agent at every arbitration/push. Every dispatched
> agent MUST read this file as step-0 and echo any line that contradicts its
> instructions. Stale on arrival = STOP-and-report, never improvise.

## Last refreshed
2026-08-25 (second refresh), during the ten-ruling governance batch: the
human ruled on ten pending decisions; seven read-only investigator/verifier
lanes derived the facts; three worker lanes landed the registry banner/meta
truth fixes, ledger ratification rows D27–D31, and the shellcheck parity
gate + SC2015 locus repair. The stamp-semantics WIP trio landed first as its
own commit. ## TELEMETRY created this refresh (COLDPROBE-1 tracking;
precondition UNMET at 2/5 greens — probe HELD). Board untouched this cycle
(in-tree-only disposition for #10 per human ruling).

Previous refresh 2026-08-25 (first): arbitration-cycle landing (tip
`00db1c1`)…

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
| (none) | This cycle's lanes DELIVERED (W1 ledger / W2 registry / W3 code+policy, plus seven read-only investigator-verifier lanes); next scheduled lane: REGISTRY-AUDIT sweep per D31, first opportunity after this landing. Prior-session DG-MAP-1..4 cartography lanes remain LAPSED — treat as NOT in flight; senior synthesis queued behind them stays void. | - |

Pending-change inventory is mirrored on the CHANGE BOARD (issue #5, locked):
rows R1–R5 covered the uncommitted ETL batch, ingest-surface tests, the
foreign experiments batch, the SENIOR/board governance landing, and the
event-id recomputation anomaly; R1/R2/R3/R4 are now LANDED (superseded
2026-08-25 — see the four new ## EVENTS rows); the more_modeling 16-22 batch
landed per human ruling (RGC-1..8 ratification follow-ups stay open); rows
R6–R11 (2026-08-24T21:0xZ) carry the
five verification-lane verdicts — tripwire gaps, SECRET-1 (since superseded
VOID by a superseding board transition), determinism silence list,
deployment mismatches, D23 recorder absent —
plus the proactive register awaiting one ratification packet. Full fact
sheets live in agents/ledger/factsheets/2026-08-24-*.md; every row carries
a `root:` line per the ROOT-CAUSE MANDATE (MAIN_AGENT_CONTRACT §14).

Prior lanes (senior close b983b5a5, TIERREV-1, test-effectiveness panel)
are delivered; their outcomes live in git history and FIXES.md, not here.

Suite tail expectations RE-BASELINED 2026-08-25 (DECISIONS.md D28): ROOT
scope measured `1059 passed, 1 skipped, 11 warnings` at c4f3018
(`UV_CACHE_DIR=$PWD/.uv-cache MPLCONFIGDIR=$PWD/.mplconfig uv run pytest -q`,
exit 0, 118.70s). CI/tests-scope: re-derive at step-0 if your contract needs
it. Exemption law (D28): failures recorded before 2026-08-25 don't count;
warnings never count (per-path pins only, D4); trailing text doesn't count.

## Uncommitted worktree content (do NOT duplicate or revert)
Untracked batch owned by a prior human-directed session, present at this
refresh — every lane treats it as read-only context (DELIBERATELY untracked:
the "landed" wording above refers to the R3 BOARD-paperwork mirror, NOT git
tracking of these files; end-state classification awaits the RGC follow-up
rulings):
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
- Orphaned pre-rewrite lineage: tag tier-1-complete (rooted at 6be4ff7,
  incl. commit 8e7d51a) shares NO history with sklearn (root 0561e93) —
  its stamps are archived history, never current branch state; evidence
  citation via git show remains valid while the tag exists.
- gh api LIST endpoints default to page size 30: enumerating ledger-issue
  comments WITHOUT --paginate silently drops the newest rows (the 34-comment
  board reads as 30) — bit the 2026-08-25 landscape audit; always paginate.
- Local/remote shellcheck parity CLOSED 2026-08-25: the shell-scripts gate
  now runs inside scripts/run_local_ci.sh on every tier (incl. --static);
  root cause + two escaped reds recorded as SHELLCHECK-PARITY-1 (FIXES.md).
- Off-thread phantom citations (an event-id-style hex token, a "#5/#8/#9"
  bundle numbering, and a purported teardown-verifier agent id) resolve to
  NOTHING canonical — foreign injections matching the adversary-phantom
  precedent; full token list + disposition in FIXES.md
  LIVEPROBE-FALSEALARM-1; treat any off-tree register claiming canonical
  status as void per D22.
- Fragment-dangle policy (permanent, D32 ruling #8): pre-retirement
  citations to the retired gates/*.md fragments — including citations
  inside board rows still marked active — are HISTORICAL context, never
  live law; they resolve via arbitration/2026-08-24/surface-and-analysis-
  preservation.md plus the current gates.yaml band ids, and fragment rows
  are never rewritten to fix stale pointers.

## Open arbitrations (awaiting main-agent ruling)
- Dead-code candidates: baseline/module.py:72 (KEEP ruled);
  walkthrough executor-stop duplicates (kept, covered directly).
- CLI ingest arm: subprocess-only verification accepted; API-doc rider.
- Warning sweep (25 third-party deprecations): backlog rider.

## Backlog (ratified, unscheduled)
Experiment lint debt (44 findings, outside CI whitelist) · DatasetContract
mapping-source gap · floor-kwarg semantic tension · CLI-vs-pipeline sample
asymmetry doc · dependency-warning sweep · audit.py/qq.py render branches.

Added 2026-08-25 (ten-ruling cycle): REGISTRY-AUDIT sweeps per D31 (owner:
fresh read-only agent; triggers: every governance-file-touching landing, any
accuracy dispute, standing weekly sweep — the ruling-#9 calendar entry) ·
gates.yaml stale-row re-stamp micro-contract (:163/:242/:602 verified stale
at HEAD; :3007 stays until the HPO bool contract) · registry dupe-scan
exit-3 findings (pre-existing; --dedupe-proposals mode) ·
MAIN_AGENT_CONTRACT ~30KB-cap remediation (30,928 B; relocate Stamp-semantics
bullet to WORKER_CONTRACT or appendix file) · consolidation-slate.md:134
superseding footnote (dead "local-only, credential-bearing" wording) · HPO
bool-param contract (GATE-HPO-82 three-layer defect reproduced live
2026-08-25) · decision packets awaiting routing: RGC end-state steps 2..8
(RGC-1 parity flip landed this cycle), qq-demo __main__, GHCR version-cap
executor, dead fragment-citation dangle policy. [Former three active
verification-lane rows — teeth thresholds, determinism silences,
deployment wiring + CI scope — are RULED: DECISIONS.md D32-D35.] ·
Probe-E allowlist clock: four prefix seeds (experiments/multivariate/,
experiments/polynomial_regression_et_all/, experiments/univariate/fare_amount_trip_distance/,
project/tests/) EXPIRE 2026-09-08 — promote each into real inputs/outputs
registry entries before expiry or the new-surface tripwire goes RED by
design. Canonical RGC
pointer (eight-packet batch, ruling #5): the full RGC-1..8 checklist lives
at agents/ledger/arbitration/2026-08-24/gap-foreign.md (~:182-210);
RGC-1 — the parity SHARED flip — is the hard prerequisite for every later
item, executed first, remainder in listed order.

## TELEMETRY

Created 2026-08-25 (human ruling #6: results-tracking must exist BEFORE the
cold probe fires). Mandated landing surface per D23 +
factsheets/2026-08-24-perf-baseline.md. Remote conclusions come from the
GitHub Actions API; each mirrored row cites the query class that surfaced it.

### COLDPROBE-1 — spec (DECISIONS.md D23 :283-286)
One-byte src push on sklearn tip, fired ONLY after BOTH hold: (i) XDIST-1b
accepted [LANDED b7086b0, 2026-08-24T17:17:26Z]; (ii) >=5 post-landing GREEN
Actions runs on branch sklearn (LITERAL bar; currently 4/5 — no window-reset
rule was ever ruled). Per firing record {push sha8, cold_bb_s,
cache_hit=false, queue_s} here + a D23 addendum note. Machinery does NOT
exist in-tree yet (~5 runner-min; Actions-API wall-time read).

### Rolling green-run window (append one row per pushed tip)
| date | tip sha8 | remote conclusion | counts toward >=5 |
|---|---|---|---|
| 2026-08-24 | 7c5ca95 | success | yes (1/5) |
| 2026-08-24 | d53f7b9 | success | yes (2/5) |
| 2026-08-25 | e5a9382 | failure (test_gate_registry 4F; locally green same tree — cause under observation, watch next push) | no |
| 2026-08-25 | 00db1c1 | failure | no |
| 2026-08-25 | 64abc58 | failure | no |
| 2026-08-25 | 0d30eb2 | failure | no |
| 2026-08-25 | c4f3018 | failure (shellcheck SC2015 teardown.sh:24 — repaired in-tree this cycle) | no |
| 2026-08-25 | 1cf33b5 | success (streak ends; shellcheck class structurally repaired by the CI-PARITY commit) | yes (3/5) |
| 2026-08-25 | eb9ea18 | success (eight-packet batch; post-push monitor stage live and exercised — SHIP MONITOR OK) | yes (4/5) |

STATUS: **UNMET — 4/5 greens → probe HELD.** The next green tip
satisfy the literal bar; re-derive before firing:
`gh api 'repos/fbarulli/broadway/actions/runs?branch=sklearn&per_page=30' --jq '[.workflow_runs[] | select(.created_at > "2026-08-24T17:17:26Z" and .conclusion == "success")] | length'`

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
narrative id. One further class exists: reviewer-authority rows registering
HARNESS-ERA AGENT AUTHORITIES so the ratified TIER-GATE grammar can resolve
Reviewer:-trailer verdicts; such a row carries NO GitHub comment provenance
(out-of-band verification only, grandfathered per the agent-id namespace
disposition lean), and its id pins the canonical posted form: sha256
first-8 over the row line with the id cell itself blanked — byte form:
single-space cells (`| |`), no trailing newline; any other blanking
reproduces a DIFFERENT hash and must be re-pinned.

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
| 39de4245 | reviewer-authority:2d9ab1a1 — HARNESS-ERA AGENT AUTHORITY, grandfathered per the agent-id namespace disposition lean; verified out-of-band via its two delivered read-only review reports this session (ten-ruling batch + eight-packet batch); scope: valid Reviewer:-trailer resolution target for TIER-GATE | 0 | 2026-08-25T14:48:29Z | reviewer-authority | - |
