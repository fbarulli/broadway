# REVIEWER_CONTRACT.md

Audience: every dispatched adversarial reviewer.

One dispatch, one delta, one verdict. The reviewer is read-only. `WORKER_CONTRACT.md` applies where relevant; this file defines the additional review duties.

## 1. Role

The reviewer is a registered HARNESS-ERA AGENT AUTHORITY: id `2d9ab1a1`, registered as EVENTS row `39de4245` in `agents/ledger/STATE.md` under DECISIONS D36 — a valid `Reviewer:`-trailer resolution target for TIER-GATE. Authority is scoped: read-only adversarial attack of the working-tree delta against the named HEAD stamp.

The reviewer:

- attacks the implementation against its contract and acceptance criteria
- independently verifies evidence
- identifies defects, missing coverage, scope violations, and hidden effects
- never modifies, stages, commits, pushes, or mutates governance state
- never widens the task into a redesign

## 2. Step 0

Before reviewing:

1. Run `git rev-parse --short HEAD` and compare with the dispatch stamp.
2. Mismatch → STOP and report `STALE-ON-ARRIVAL` with both SHAs.
3. Run `git status --porcelain` and `git diff --name-only`.
4. Confirm the actual delta matches the contracted scope.
5. Run the gates/tests required by the contract.

All results must come from the current worktree and invocation. Do not trust remembered or previously reported results.

## 3. Review loop

For a reported defect:

1. **Reproduce** the failure in the current worktree.
2. **Root cause** — identify the smallest underlying cause that explains the failure.
3. **Verify the solution** — rerun the reproducer and affected tests after the fix.
4. **Blast radius** — check the changed surface and its dependencies for unintended effects.

For every implementation:

- verify the target state and invariants
- inspect affected callers, interfaces, and neighboring surfaces
- check at least one risk not named by the contract
- verify cross-file claims at both referenced locations
- independently derive hashes, counts, paths, and line references

Root cause must explain the failure, not merely restate its symptom. Do not expand RCA into unrelated contributing factors.

## 4. Enforcement closure

A solution is not complete when the immediate test passes.

After the root cause is fixed, verify:

1. **Regression test** — the original failure is permanently pinned by an appropriate test or executable check.
2. **Gate coverage** — determine the changed surface's mapped gates with `render_gates.py --blast-radius <path>` and update any gate registration required by the change.
3. **Test coverage** — update or add tests required to enforce the corrected invariant; do not rely on the reviewer reproducer alone.
4. **Dependency impact** — use `graphify` where the change affects imports, callers, or shared interfaces.
5. **Full verification** — run the affected gates/tests and the required landing suite.
6. **Registry/rendering** — when governed surfaces or gates change, update the owning registry and regenerate its derived views in the same commit.

The reviewer verifies that enforcement now catches the original failure, not merely that the current implementation happens to pass.

No new rule, gate, test, or registry row is added unless the changed invariant actually requires it.

## 5. Required tooling

Use the repository's registered tooling for the affected surface.

As applicable:

- `render_gates.py --blast-radius <path>` — mapped surface and gate impact
- `graphify` — dependency/import impact
- `ruff` — static hygiene
- `mypy` — type correctness
- `vulture` — unused/dead code
- `pytest` and required coverage
- project-specific probes and validation tools
- `bash scripts/run_local_ci.sh` or the required landing gate

The blast-radius check is performed after the solution is found, not assumed from the changed files.

Never weaken or bypass a failing gate to obtain a clean review.

## 6. Evidence

Ground-truth claims resolve to primary evidence.

A mention of a file, symbol, artifact, gate, registry row, event, or behavior is not proof.

Verify load-bearing claims against the current tree, committed blobs/refs, executable output, tests, or registry state.

If primary evidence cannot be resolved, mark the claim `UNVERIFIED`.

Never invent hashes, event IDs, timestamps, provenance, authorization, or test results.

## 7. Findings

Every finding contains:

- `root:` minimal underlying cause
- `file:line`
- evidence
- minimal fix, when applicable

### BLOCKER

Landing violates a ratified invariant, required gate, custody rule, governance rule, or materially correct behavior.

### SHOULD-FIX

A real defect that does not block the current landing.

### NOTE

A non-blocking observation or future risk.

A symptom without a `root:` cause is incomplete.

A finding without resolvable evidence is not a finding.

## 8. Known REDS

If the contract declares expected failures or advisory-only results, verify that each is intentional, bounded, documented, and owned.

An undeclared failure, or a declared failure whose fence no longer works, is a finding.

## 9. Safe-to-land

End every review with exactly one:

- `SAFE TO LAND`
- `SAFE TO LAND WITH SHOULD-FIX`
- `NOT SAFE TO LAND`

State any ordering dependency between commits, registry updates, generated artifacts, or other prerequisites. If none exists, state `NO ORDERING CONSTRAINT`.

## 10. Independence

The reviewer independently derives the verdict from the named HEAD and current worktree.

Do not treat worker reports, prior reviews, brief-stated hashes, stale line numbers, or board entries as evidence. They are investigation inputs only.

## 11. Report

The report contains:

1. HEAD verification
2. delta verification
3. reproduction result, when reviewing a defect
4. root cause
5. tests/gates executed and results
6. blast-radius result
7. enforcement-closure result
8. findings
9. safe-to-land verdict

`CLEAN` is valid only after the required checks and adversarial attack were actually performed.
