# STATE.md — living agent context (single source of "what has been done")

> Refreshed by the main agent at every arbitration/push. Every dispatched
> agent MUST read this file as step-0 and echo any line that contradicts its
> instructions. Stale on arrival = STOP-and-report, never improvise.

## Last refreshed
2026-08-24, after GATE-SSOT + COV-95 landing (`ff595b4`).

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
| `b983b5a5` | Senior decision-pipeline close | delivered (3 rulings) |
| TIERREV-1 worker | Batch A: classifier + probes + Tier trailers | dispatched |
| `81abc1fe` `299ae05a` `91bfb004` | Test-effectiveness panel | delivered; findings queued for senior synthesis |

## Lane telemetry schema
Stage-split timing columns per dispatched lane (TIERREV-1, D19 P1 telemetry
condition). Schema ONLY — the main agent populates values at
dispatch/arbitration; no rows exist yet, none are fabricated here.
| Column | Meaning |
|---|---|
| `lane_id` | dispatched agent id (8-hex) or worker label |
| `contract_id` | dispatch contract label (e.g. TIERREV-1) |
| `classify_ts` | ISO-8601 UTC instant the change set was risk-tiered via `scripts/tier_classifier.py` |
| `dispatch_ts` | ISO-8601 UTC instant the worker launched |
| `verdict_ts` | ISO-8601 UTC instant the arbitration verdict landed |

Landed since last refresh: e9fce2d (D17 GATE-SSOT), 9f50574 (COV-95 +49 tests,
floor 95 bound in script), ff595b4 (ledger). Suite tail NOW: 827P/1S gated
scope (95.31%), 846P/1S root scope. Pending fix queue from panel:
get_champion MlflowException swallow (deploy-critical), config-silence
cluster, assertion-strength gaps (test_baseline decorative, message pins
~8%, timeline dark), mutation survivors M1/M10 (+verify M5/M2/M9 against
post-commit tree).

## Uncommitted worktree content (do NOT duplicate or revert)
(none — worktree clean as of ff595b4)

## Standing hazards (learned the hard way)
- `/tmp` is namespaced PER TOOL CALL here — nothing persists between calls;
  use workspace-hidden dirs for multi-call scratch, delete before finishing.
- NEVER stage/index anything unless your contract says the exact `git add`
  lines; two prior agents were barred for rogue staging/deletion (FIXES.md).
- Re-export/facade F401s: NEVER autofix wholesale — alias-form or `__all__`
  only. (Historic trap: `_common.py`/`_setup.py` once routed ≥25 imports;
  both deleted in b15f66e era. Verify import-site volume before citing.)
- Full pytest+cov needs ~150s > 120s cap: use background-run + poll pattern.
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
