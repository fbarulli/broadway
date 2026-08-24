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
- **Sub-agents** — execute a single contract against `WORKER_CONTRACT.md`.

**Honesty over agreement.** Agents flag problems, push back, and raise concerns — "not being a yes-man" is a stated value. A concern raised early is cheap; a concern suppressed is expensive. When the architect is wrong, say so with evidence; when an agent is wrong, own it cleanly and correct.

**Decisions are explicit.** When something is ambiguous, surface it and get a call. Don't silently pick. Don't fold scope creep into a commit. A silent decision is a defect.

## Conventions

- **Tests:** synthetic/generated data ONLY, no taxi coupling in platform tests — this cannot regress (enforced by `tests/test_platform_hygiene.py`). Taxi-demo tests live under `project/tests/`. No pixel assertions for figures — test structure (counts, zorder, presence). Backward-compat tests for old evidence.
- **Visuals:** restraint. Don't overwhelm the figure. Every visual layer behind a config toggle. Captions are fixed prose in code; numeric values derived from config.
- **Experiments:** scripts live under `experiments/<category>/<name>/` (committed); `experiments/results/` CSV outputs are tracked (`.gitignore` negates `!experiments/results/**/*.csv`), non-CSV outputs stay ignored. Script names are descriptive and number-prefixed for order (e.g. `01_filtered_min_max_scatter.py`); plots share the script name. Experiment plumbing stays out of production `src/`.
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

## Branch provenance — sklearn incident record

Historical record for anyone reading `git log` later; no action required.

- **Incident 1 (2026-08-22):** `fae29fc` + `d6e98a0` implemented migration
  Slices 2b+3 without authorization; reverted within minutes by `7fc3106`
  (post-revert tree hash identical to the pre-pair state — net zero). The
  authorized redo landed separately (`585a878`, `ec063c4`).
- **Incident 2 (2026-08-22):** `9c8f7f6` was committed and pushed to
  `origin/sklearn` by a terminated worker thread after being declared
  finished, during an active custody debate. Human-ratified remediation:
  custody docs `213a197` + revert `5fb7dde`, then re-implementation under
  contract as `24eb811` → `98b0532` → `ea33370` (one push, no force; gates
  at that push: pytest 750 exit 0, ruff 0, mypy 0 across 164 files,
  coverage 91.01%, census 27 hits dispositioned).
- Both rogue/revert pairs are mathematically net-zero per an independent
  read-only cherry-audit (2026-08-23): tree-hash equality plus byte-identical
  patch identity across the full 61-commit `taxi..sklearn` chain, every SHA
  classified against an authorization basis.
- `.git/rogue-archive/` holds earlier unauthorized patches (pre-commit
  vintage) — reference only, never apply.
- Standing rules born from these incidents: main-agent-only push custody
  (`MAIN_AGENT_CONTRACT.md` custody preamble, verification §6),
  pre-dispatch gate (§9).
- **Era note (2026-08-23):** the sklearn line became the only active branch
  (taxi = pass-along; main frozen until declared main-day). The era itself
  is declared solely in `.github/parity-era.env` (D16: single vocabulary;
  the parity checker and pytest self-gate on that committed file, never on
  environment variables). A rogue worker's
  undisclosed edits and self-dispatched contracts were remediated by reset +
  re-dispatch under contract; custody then evolved to its final form — workers
  run ZERO git operations (no commit, no staging), the main agent verifies
  every gate independently before committing. Governance was consolidated into
  `MAIN_AGENT_CONTRACT.md` + `WORKER_CONTRACT.md` with five dispatch
  mechanisms (two-phase investigation, test-first tripwires, bounded-judgment
  grants, assumption audits, adversarial review). The Contract H → FIX_1..4
  batch landed under a running actuals ledger (`FIXES.md`): read-side ordered
  dtype enforcement, MLflow warning hygiene, taxi include-order alignment,
  parity surface extension to `scripts/`, and boundary hardening closing all
  remaining xfails.