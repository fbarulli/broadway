# STATE.md — current operational control record

`STATE.md` holds only active custody and retryable operational intent. Git is
the authority for landed history; GitHub Project #4 is a mirror, never evidence.

## CURRENT

| id | kind | status | owner | custody | updated | source | github_item | mirror_state | summary |
|---|---|---|---|---|---|---|---|---|---|
| STATE-20260904-006 | hazard | open | human owner | main agent | 2026-09-04 | owner directive 2026-09-04 (keep a list of all gaps found and add them to STATE) | PVTI_lAHOAZFnCc4Bhhjqzg5chi0 | synced | GAPS LEDGER (consolidated, every gap found this session, owner directive to keep and grow this list): (1) REMOTE CI WATCH-GAP - euromonitor ran red silently 2026-09-02 to 2026-09-04 (dead k8s/optuna glob after the optuna stack removal, runs 33645187295 and 33851743232 both failed unwitnessed) fixed by nullglob step rewrite in a41c5cf, prevention = ship.sh post-push monitor now verified live. (2) COVERAGE FLOOR DRIFT - pytest coverage was 94.98 percent at origin (under the 95 floor) with no local alarm because the floor only binds in the CI pytest tier, restored to 95.34 via audit-render and nlp helper coverage in 9d4308d, gap-prevention = local tier now runs the same cov-fail-under 95. (3) TIER-GATE AUTHORITY GREP LEAK - the reviewer-authority grep printed its matching row into the captured refusal reason so every authority-resolved Tier-FULL commit was unshippable, fixed with grep -q in scripts/tier_gate.sh. (4) LEDGER_COMMIT GATE-3 BLINDNESS - the STATE-linkage gate could not express close or void commits because the cited row had just left CURRENT, fixed by accepting archive markers. (5) TEMPLATE CLI DRIFT - both state templates taught flags the tool rejects (record add --updated, record close --id) - the exact traps hit live, fixed in 08e2c19. (6) NOTHING-EVER-CLOSES PATHOLOGY - the ledger accumulated open rows for delivered work (10 rows closed this session, batch ledger law now structurally forces terminal dispositions). (7) STATE NUMBERING GAPS - 0902 sequence missing 004 (never minted) and 012 and 016 (closed then referenced), recorded on -004. (8) EVENTS FORK UNDOCUMENTED - the 2026-08-24 senior-agent fork (event ae44dbfd spawning board rows b625b5e0 and ed2f1fdf) had no written disposition, now documented on -003 close. (9) COLDPROBE FIRING BRANCH AMBIGUITY - D23 pins the probe to the sklearn tip but the remote sklearn branch no longer exists (renamed taxi) and the frozen taxi snapshot is perma-red (its parity gate pins origin/sklearn, run 33808377555) - candidate firing line euromonitor, awaits owner ruling on STATE-20260904-005. (10) DEADCODE CENSUS BACKLOG LIVE 2026-09-04 (advisory per D34, never a gate) - A1 zero-ref NONE, A2 same-module-only 140 (all 4/5 ELEVATED, dominated by private helpers called from their own module like cli _build_parser - known-weak bucket), config-key heuristic 2 rows (configs/nlp.yaml model_zoo.minilm_l6 and model_zoo.bge_small) VERIFIED FALSE POSITIVES (nlp.py reads model_zoo at runtime via hpo spec names, indirection the heuristic cannot see), report at scratch/backlog/deadcode-census-2026-09-04.md. (11) SHIP MONITOR TIMEOUT RACE - ship.sh post-push Actions monitor can outlive its tool timeout when CI exceeds roughly 7 minutes (run 33864307689 took 7m18s), monitor should report still-running instead of dying, not yet fixed. (12) STASH HYGIENE - two sklearn-era stashes (7dcb34f and c4f120a WIP) remain from the dead branch era, landed stash dropped. (13) PARITY F1B PIN - the taxi snapshot parity gate hard-pins origin/sklearn which can never go green again, perma-red by construction, acceptable because taxi is a frozen archive branch per the 50x ruling. |
| STATE-20260904-007 | lane | open | human owner | main agent | 2026-09-04 | STATE residual audit 2026-09-04 follow-through | PVTI_lAHOAZFnCc4Bhhjqzg5c3j8 | synced | Post-audit follow-up lane (only open work remaining after the 2026-09-04 full-STATE verification sweep): (1) ship.sh monitor timeout race - the post-push Actions monitor can outlive its tool timeout when CI exceeds roughly 7 minutes (run 33864307689 took 7m18s while the monitor window expired), fix = monitor reports still-running with the gh run id instead of dying, then the operator re-checks; (2) stash hygiene - two sklearn-era stashes (7dcb34f and c4f120a WIP) from the dead branch era remain in the stash list, drop after confirming content is recoverable from history. Coldprobe firing (gaps ledger item 9) stays blocked on the owner ruling and is NOT part of this lane. |

## Access protocol

- Main alone uses `state_records.py record add|update|sync`; workers and
  reviewers only read and report.
- Local CURRENT intent is written before its mirror. A failed mirror leaves
  `mirror_state=pending`, and `record sync <id>` retries the same record.
- The helper changes only CURRENT; `## EVENTS` and the historical archive are
  immutable through this interface.

## Retention

The pre-foundation record is preserved verbatim in
`agents/ledger/archive/2026-08.md` from source commit `7dcb34f`. Current
operational history stays in git and Project #4 mirrors.

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
