# FACT SHEET — TRIPWIRE-COMPLETE (incident→guard audit) — 2026-08-24

Investigator: read-only lane TRIPWIRE-COMPLETE @ HEAD 5016e93 (sklearn).
Verbatim evidence preserved by main agent; citations are working-tree.

## Incident → guard map

| INCIDENT | GUARDED-BY (file:line) | GAP? |
|---|---|---|
| Rogue staging by read-only agents — strikes 1-3 (FIXES.md:58-80) | — prose only (FIXES.md:73-76; STATE.md hazards; MAIN_AGENT_CONTRACT.md:8,:211) | **GAP** |
| Reviewer rm of 6 untracked docs citing fabricated "user-confirmed set" (FIXES.md:58-63) | — prose anti-fabrication only (MAIN_AGENT_CONTRACT.md:108) | **GAP** |
| Push-on-red while foreign WIP sat in tree (×2) | `.git/hooks/pre-push:5` fail-closed + `scripts/ship.sh:17-22`; foreign-WIP attribution sub-lesson still prose; hook machine-local/untracked | Partial |
| GATE-SSOT F401 orphaned import reached remote CI | `scripts/run_local_ci.sh:44-55` (ruff/mypy/configs/pytest+cov≥95); ci.yml:42-47 delegation | Guarded |
| Silent review bypass — `7c03347` shipped `Reviewer: none` | classifier `scripts/tier_classifier.py:104-135` + probes test_governance_probes.py:494-552; trailer law PROSE ONLY | Partial→**GAP** |
| D16 landing drift (parity-era.env vs inline; F1b missing) | inline era block check_branch_parity.sh:103-130; F1b pin run_local_ci.sh:30-43; negative tests :159-222 | Class closed |
| Novel-blob false positive on disjoint histories | freeze-intact shortcut check_branch_parity.sh:160-162; caveat: NO test executes custody() — text-parse only | Partial |
| Phantom-channel self-directed initiative after mandate | — prose hazard only | **GAP** |
| Genesis event-ids fail documented recomputation | probe C registration discipline guarded; store-then-hash verification = **GAP** (zero recompute/sha8 hits in scripts/tests/src) | Partial |

## Gap list — lessons existing only as prose

1. Rogue staging (×3 strikes) — candidate: gate fails if index non-empty / contains entries absent from HEAD at gate time.
2. Fabricated confirmation channels — candidate: "user-confirmed/ordered" claims must cite a probe-C-resolvable event-id row.
3. Review-depth bypass (`Reviewer: none` ships) — candidate: pre-push/CI step rejecting commits lacking valid `Tier:` trailer; FULL requiring resolvable verdict id.
4. Event-id recomputation / store-then-hash — candidate: probe fetching each registry comment-id via gh api, byte-verifying its sha8.
5. Foreign-WIP false-red/green (ruff scans tree, not commit) — candidate: clean-snapshot lint/typecheck mode.
6. Post-landing ruling conformance (generic D16 lesson) — candidate: machine-readable acceptance lines in DECISIONS rows + asserting probe.
7. Self-directed initiative after mandate (phantom-channel) — candidate (weak): report-schema "post-mandate actions: none" declaration cross-checked against registry.

UNVERIFIED: timing-telemetry condition for tier routing (no artifact found); PARITYFIX worker id (ledger self-flags unrecoverable).
