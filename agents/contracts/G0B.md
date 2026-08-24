# G0B.md — governance truth + evidence-format rule (+ arbitration riders)

Version 1, frozen 2026-08-23. Ratified under DECISIONS.md D1/D2/D8; carries
G0A-arbitration rider OQ2 and the D8 worker-evidence rule. Worker: one turn,
working-tree changes + report only, ZERO git operations.

## Step-0 hash gate
First action: `git rev-parse --short HEAD` must equal `8e7d51a`. Mismatch →
STOP. Then the step-0 gate paste, under a hard 120000 ms command timeout:
`UV_CACHE_DIR=$PWD/.uv-cache MPLCONFIGDIR=$PWD/.mplconfig timeout 120 uv run pytest -q`
Expected tail in this environment: `783 passed, 1 skipped, 9 warnings …`.
Any difference → STOP and report.

## Task
Make the governance layer self-consistent at HEAD: fix the parity-checker
header's stale branch model and .github overstatement, resolve the
"zero git operations vs mandated rev-parse" contradiction via a read-only-
inspection carve-out in BOTH worker-facing docs, fix HANDOFF's dangling
section references, codify the D8 evidence-format rule, and land the already-
ratified universal-adversarial-review edit to MAIN_AGENT_CONTRACT.md.
Scope boundary: exactly FIVE files — scripts/check_branch_parity.sh,
WORKER_CONTRACT.md, MAIN_AGENT_CONTRACT.md, CONTRACT_TEMPLATE.md,
HANDOFF.md.

## Edit list (complete)
1. **scripts/check_branch_parity.sh** header comment ONLY (lines ~1–14):
   a. Replace "taxi — working branch" framing with: "`sklearn` is the only
      active development line (`MAIN_AGENT_CONTRACT.md` §2); `taxi`
      fast-forwards to it after each green push; `main` is frozen until
      declared main-day."
   b. Narrow ".github/" to ".github/workflows/" so prose matches SHARED.
   c. PRESERVE the literal string `scripts/` in the watched-surface sentence —
      tests/test_branch_parity_scripts.py asserts it (line ~43).
   d. Do NOT touch SHARED array, check loop, --sync, exit codes (lines 15+).
2. **WORKER_CONTRACT.md**:
   a. Custody section: append one line — "Exception: read-only git inspection
      (rev-parse/status/diff/log) is allowed and expected; write operations
      (add/commit/stash/branch/checkout/push) never."
   b. Report format §3: append — "Paste the COMMAND alongside every output
      tail (D4: an honest number under a different invocation is still a
      mismatched gate)."
3. **MAIN_AGENT_CONTRACT.md**: commit the ALREADY-PRESENT working-tree edit
   (universal adversarial review in §4 #5 / selection guide / §6 / §8) by
   including this file in your changed-files report; make NO further edits
   to it beyond one addition — §6 gains the sentence: "Every gate paste names
   its exact command (D4)."
4. **CONTRACT_TEMPLATE.md**: in Constraints, align custody wording by adding
   the same read-only-inspection exception line as 2a.
5. **HANDOFF.md** lines ~115–116: fix cross-references — "custody §3" →
   "custody preamble"; "pre-dispatch gate (§3d)" → "pre-dispatch gate (§9)".

## Constraints
- No other file changes; suite delta MUST be zero (docs + shell-comment only).
- The checker must remain executable and behaviorally identical
  (`bash -n scripts/check_branch_parity.sh` clean = paste it).
- Deletion-first; no new doctrine sections anywhere.

## Acceptance checks (evidence pastes)
1. `bash -n scripts/check_branch_parity.sh && echo SYNTAX_OK`
2. `grep -n "working branch" scripts/check_branch_parity.sh` → NO MATCH
3. `grep -n "\.github/" scripts/check_branch_parity.sh | head -3` → header
   says `.github/workflows/` only
4. `grep -n "scripts/" scripts/check_branch_parity.sh | head -2` → header
   still names scripts/ (test assertion intact)
5. `uv run pytest tests/test_branch_parity_scripts.py -q` → `1 passed,
   1 skipped` (unchanged)
6. `grep -n "read-only git inspection" WORKER_CONTRACT.md CONTRACT_TEMPLATE.md`
   → one match each
7. `grep -n "command alongside\|names.*exact command" WORKER_CONTRACT.md MAIN_AGENT_CONTRACT.md`
   → matches present
8. `grep -n "§3d\|custody §3" HANDOFF.md` → NO MATCH
9. Close gate: same step-0 pytest command; tail counts identical
   (783/1/9), wall-clock may differ.
10. `git status --porcelain` final paste: exactly the five files + documented
    WIP.

## Report format
Step-0 result + paste; per-file summary w/ line refs; all ten pastes;
assumption audit (≥3 re-verifications + ≥1 beyond-brief check); OPEN
QUESTIONS mandatory. Acknowledge the 120s cap in the report.
