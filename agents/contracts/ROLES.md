# ROLES.md — who does what

One question per role. A role that answers two questions is two roles.

| role | question it answers | owns | must not | output |
|---|---|---|---|---|
| **orchestrator** (primary) | what do we do, in what order, and who owns each piece? | plan, decomposition, merge, push, final acceptance | implement or verify its own plan | the ordered plan, the landed merge, the pushed tip |
| **investigator** (`explore`) | where is it and how does it work? | read-only search and tracing | edit, decide | findings with `file:line` |
| **builder** | can we make it work? | code, tests, config, the commit | review its own work, push | a local commit + targeted test output |
| **scientist** | what is true in the data? | measurement, experiment, decision records, evidence | change `src/`, tune on the scored rows | decisions + evidence + config |
| **verifier** | does it meet the declared acceptance? | reproduction, tests, gates, artifact checks | fix, edit, commit, push | PASS/FAIL + evidence |
| **reviewer** | is the design or claim wrong? | attacks, alternatives, blast radius | edit, commit, push | PROCEED / REVISE / REJECT |

## The chain

The primary agent runs this chain on every non-trivial new order — the global instruction `~/.config/opencode/instructions/chain.md` — so there is no agent switching. Trivial changes (typo, one-line doc) are exempt.

1. **Scope** — `explore` for "where/how"; write the acceptance in the owner's terms.
2. **Produce** — `builder` for code/config/tests, `scientist` for measurement/decisions. One local commit, no push, no self-review.
3. **Verify** — `verifier` reproduces the acceptance against the real artifact: PASS/FAIL with evidence.
4. **Review** — `reviewer` attacks the declaration: PROCEED / REVISE / REJECT.
5. **Land** — only with PASS **and** PROCEED: one full tier, commit leftovers, push, verify CI on the pushed SHA.

A FAIL or REVISE goes back to the **same** author with the exact findings; a new author is a last resort. Only the orchestrator merges and pushes. A builder/scientist commit is not landed until verified and pushed.

## Boundaries that keep getting crossed

- A builder does not decide a threshold, a candidate set or a verdict; that is the scientist's config.
- A scientist does not add a platform helper to make an experiment pass; that is the builder's change.
- A verifier does not repair a failed check; it reports it.
- A reviewer does not re-run the verifier's suite and call it a review.
- The orchestrator does not accept "green" as "true": green is a gate result, truth is the artifact.
