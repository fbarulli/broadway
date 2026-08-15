# HANDOFF.md — Working Style

## What this is

Broadway is a **data-agnostic analysis platform**. The taxi dataset is a test case, not the product. Everything we build must work on any dataset without code changes — domain knowledge lives in config and analyst decisions, never in platform logic.

This is not a task log. It is the **working style** — how we think, plan, and build. Read it before touching anything.

## The north star

**The platform must not know about the data.**

If you catch yourself writing `if feature == "tip_amount"`, or "monetary features", or "borough" into `src/broadway/`, stop. That knowledge belongs in:
- **Config** (`configs/`) — authored policy: thresholds, sample sizes, palettes, feature lists.
- **Analyst decisions** — recorded verdicts, not platform heuristics.
- **Evidence** — observed facts, persisted to JSON.

The platform provides generic machinery. The dataset and the analyst provide the specifics.

## The three layers

- `src/broadway/` — data-agnostic platform machinery. Reusable on any dataset: sampling, structural cleaning, splits, lookup joins, stats, renderers, lineage. Test: could this run on another dataset with zero code changes? If not, it does not belong in `src`. No column names, thresholds, or dataset terms.
- `project/` — dataset binding: config plus thin glue. Contains the dataset contract, feature registry, and thin wrappers binding `src` machinery to this dataset, e.g. `read_training_sample = read_sample(_contract, seed=RANDOM_STATE)`. No new platform logic.
- `experiments/` — analysis scratch space. Specific questions, domain-specific filters, plots, and one-off exploration. Promote a pattern to `src` only after repeated concrete use.
- Tiebreaker: if it names a column or hardcodes a threshold, it is not `src`. It belongs in `project/` config or an experiment.

## Architectural principles

**Config over hardcoded policy.** Thresholds, filenames, sample sizes, palettes, gates — all in YAML. Code carries no analytical policy. If a number affects the analysis, it's config.

**Typed evidence, dumb renderers.** Compute at profile/analysis time, persist to JSON evidence. Renderers read evidence; they never compute. Reports render evidence — they never become a second source of truth.

**Pure stats library.** `src/broadway/stats/` reads no config, touches no I/O, reads no env. Thresholds are function parameters; callers pass values in.

**Compute/IO isolation.** stats and audit are pure; timeline owns I/O.

**Single-owner surfaces.** Every report/figure has exactly one writer-of-record. If two code paths write the same surface, one is dead — delete it.

**Ruthless pruning.** Delete dead code, dead surfaces, dead branches. Cruft does not accumulate. A stale doc line is a bug.

## The decision hierarchy

```
Audit → Timeline → Evidence → Suggestion → Decision → Result
```

These layers must not blur:
- **Authored intent** (config, analysis definition) ≠
- **Observed evidence** (what the data shows) ≠
- **Runtime decision** (what the pipeline chose to do).

A suggestion is not a decision. Evidence is not a verdict. The pipeline shows its work; the analyst makes the call.

## Working style

**Narrow vertical slices before abstraction.** Build and dogfood a thin end-to-end slice before generalizing. Don't refactor dispatch chains mid-feature. Don't build the abstraction before the second concrete case demands it.

**Coarse plan upfront, precise contract just-in-time.** Sketch the full sequence (names, scope, dependencies, order) so the shape is visible and approved. But write each detailed contract immediately before dispatch, against the just-committed state. Detailed contracts for long coupled chains rot — author them fresh.

**Dogfood the surfaces.** Every time you must open source code or raw JSON to understand what the analysis did, that's a product-surface gap. Record it. The rendered reports should explain the analysis on their own.

**Green before commit.** Suite passes before every commit. One logical change per commit. No broken intermediate states. Sample-drift artifacts get `git restore`, not committed.

## The collaboration model

Three roles:
- **Decision-maker / architect** — sets direction, resolves open questions, owns the principles.
- **Orchestrator** — plans, writes contracts, dispatches sub-agents, verifies.
- **Sub-agents** — execute a single contract against `AGENT_WORKER_CONTRACT.md`.

**Honesty over agreement.** Agents flag problems, push back, and raise concerns — "not being a yes-man" is a stated value. A concern raised early is cheap; a concern suppressed is expensive. When the architect is wrong, say so with evidence; when an agent is wrong, own it cleanly and correct.

**Decisions are explicit.** When something is ambiguous, surface it and get a call. Don't silently pick. Don't fold scope creep into a commit. A silent decision is a defect.

## Conventions

- **Tests:** synthetic data, no taxi coupling in platform tests. No pixel assertions for figures — test structure (counts, zorder, presence). Backward-compat tests for old evidence.
- **Visuals:** restraint. Don't overwhelm the figure. Every visual layer behind a config toggle. Captions are fixed prose in code; numeric values derived from config.
- **Experiments:** scripts live under `experiments/<category>/<name>/` (committed); outputs live under gitignored `experiments/results/<category>/<name>/`. Experiment plumbing stays out of production `src/`.
- **Docs:** live with the code. A renamed/moved/retired surface without a doc update is a bug in the same commit.

## What we refuse

- Domain knowledge in platform code.
- Hardcoded analytical thresholds.
- Renderers that compute.
- Two owners of one surface.
- Batching detailed contracts across a coupled chain.
- Committing broken or drifted state.
- Letting experiments pollute the product.
- Yes-men.