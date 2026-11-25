# FACTORY.md — the software-factory assembly line (SSOT)

Owner: process (main agent). This file owns the **flow** only.

Every substantive law lives in its own SSOT document. This file references them by
path and never restates them — a second copy is a drift finding, not a convenience.

Law index:

| law | doc | enforced by |
|---|---|---|
| code shape, two-pass construction | `CODE_STANDARD.md` | ruff / mypy / review |
| library-first, dependencies | `CODE_STANDARD.md` §3–§4 | dependency gate |
| one canonical implementation | `CONSOLIDATION.md` | duplicate + import-boundary gate |
| measurement and budgets | `PROFILING.md` | profile gate |
| profile-driven optimization | `OPTIMIZATION.md` | profile gate + equivalence check |
| process logging, run correlation | `LOGGING.md` | logging gate |
| evidence on real data | `LIVE_VERIFICATION.md` | contract acceptance checks |
| custody, sizing, review | `MAIN_AGENT_CONTRACT.md`, `WORKER_CONTRACT.md`, `REVIEWER_CONTRACT.md` | dispatch + review |
| SSOT, config, pydantic, tracebacks | harness `AGENTS.md` (ER contract) | gates + review |

## 1. Principle

**No code without a contract. No contract without observable acceptance. No
acceptance without evidence.**

The factory builds from a specification, not from a conversation. A request that
cannot be expressed as a contract with an acceptance check is not ready to build.

## 2. Stations

Work moves left to right. A station does not begin until the previous station's
output exists as an artifact.

| # | station | actor | input | output | exit criterion |
|---|---|---|---|---|---|
| 1 | Spec | human + main | request, rulings | filled `CONTRACT_TEMPLATE.md` | scope, non-goals, and acceptance checks are explicit |
| 2 | Blast radius | investigator | contract | ownership + caller evidence | every touched surface has a registered owner; callers listed with `file:line` |
| 3 | Build — pass 1 | worker | contract + blast radius | single-responsibility diff | smallest correct change; typed; library-first; zero git writes |
| 4 | Recompose + Optimize — pass 2 | worker | pass-1 diff + baseline profile | fused/vectorized diff + measured delta | loops and traversals fused; behavior unchanged; only measured wins kept |
| 5 | Verify | worker, then main | optimized diff | evidence pack | live run reproduced independently; profile within budget; gates green |
| 6 | Review | reviewer | diff + contract + evidence | verdict | adversarial findings, or `CLEAN` with residual risks named |
| 7 | Land | main | verdict + evidence | merged commit | full gate green; merged into the integration line (not merely pushed) |

Station 4 requires a correct pass-1 build (L1 green plus one live L2 smoke) and a
baseline profile; optimization never repairs correctness.

Stations 2 and 6 are read-only. Station 7 is the only station with git write
custody.

## 3. Roles

- **human** — rulings on scope, policy, architecture, dependencies, and anything
  irreversible. The factory never converts an assumption into policy.
- **main / orchestrator** — plans, dispatches, verifies, commits, pushes. Verifies
  every claim against the real artifact. Does not write feature code that can be
  dispatched, and never accepts a worker's self-report.
- **investigator** — read-only fact sheet: live code, ownership, callers, hazards.
- **worker** — one contract, one delta, no custody, no scope invention.
- **reviewer** — read-only adversarial attack; reproduces, root-causes, verifies.

## 4. Handoff artifacts

Each station produces one durable artifact; the next station consumes it verbatim:

1. **Contract** — `CONTRACT_TEMPLATE.md` instance (station 1).
2. **Blast-radius sheet** — `scripts/blast_radius.sh` output (station 2).
3. **Pass-1 diff** — single-responsibility working tree (station 3).
4. **Pass-2 record** — what was fused/vectorized plus the baseline and re-measured
   profile (`OPTIMIZATION.md` §5) (station 4).
5. **Evidence pack** — command, dataset, artifacts, metrics, profile, gate exit
   codes (`LIVE_VERIFICATION.md` §3) (station 5).
6. **Verdict** — `CLEAN` / defect list, each with a reproducer (station 6).

## 5. Definition of done

A change is done when **all** hold:

1. a contract exists and its edit list matches the diff;
2. pass 1 and pass 2 are both performed and recorded;
3. `CODE_STANDARD.md` and `CONSOLIDATION.md` are satisfied;
4. the required gates are green on the changed tree;
5. live-data evidence is in the report and independently reproduced by main;
6. the profile is within budget, and kept optimizations are measured wins with
   unchanged results — or a written waiver names the cost;
7. the reviewer returned a verdict with no unresolved defect;
8. the change is **merged into the integration line**, not just pushed;
9. no scratch, worktree, or temporary artifact remains.

A pushed feature branch is an incomplete deliverable.

## 6. Forbidden shortcuts

- Code without a contract; a contract whose acceptance check is "it compiles".
- Skipping pass 2, or fusing two responsibilities into one function to save a loop.
- Optimizing without a baseline profile, or hiding a behavior change in an
  optimization commit.
- Synthetic or fixture data presented as acceptance evidence for artifact behavior.
- A new dependency that is not registered; an unregistered second helper.
- Timing calls scattered through the tree instead of the profiling seam.
- "Green locally" claimed without the command and its exit code.
- Closing a defect without pinning it with a regression check.
- Leaving a fix on an isolated branch that silently excludes a sibling's fix.
- Cleanup by deletion without the consequential-action gate.

## 7. Scaling rule

Use the smallest station set that keeps the evidence honest: MICRO may collapse
stations 2 and 6 into a recorded one-liner; LARGE and CRITICAL may not collapse
any station. Sizing caps stay in `CONTRACT_TEMPLATE.md` — do not restate them here.

## 8. Enforcement

- `CONTRACT_TEMPLATE.md` requires the law sections at dispatch, including the
  pass-2 and optimization records.
- The reviewer checks stations 4 and 5 against the artifacts, not the prose.
- `scripts/run_local_ci.sh` is the gate SSOT; no station invents a private gate.
- A law without a named check is a draft; it does not become blocking.
