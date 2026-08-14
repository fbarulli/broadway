# graph_todo.md — lineage visualization proposal

Static, minimal, non-interactive. Replace the current Mermaid lineage graph with a
git-commit-tree-style rendering that shows the analysis as a branching DAG with a
tighter, clearer flow. It should look like learnGitBranching, not behave like it.

## Problem with the current graph

`reports/lineage/graph.md` is a Mermaid `flowchart LR` with three shortcomings:

1. **Machine node ids** — `ingest_taxi`, `lookup_value_taxi`, `describe:taxi_hypothesis`.
   Not human labels, no status, no sense of "where am I".
2. **Flat, unordered layout** — nodes are emitted in sort order, not investigation
   order, so the flow (`dataset -> ingest -> etl -> describe -> normality -> variance
   -> decide -> omnibus -> decide -> posthoc -> conclusion`) is not readable.
3. **No decision forks** — the branch points (`decide_omnibus` -> welch/anova/kruskal,
   `decide_posthoc` -> games_howell) are invisible. The chosen method is not marked;
   there is no HEAD, no "not taken" alternative.

## What it should look like

A static commit tree, exactly the learnGitBranching shape:

- **Nodes = commits.** Each `AnalysisStep` is a commit circle. Human label primary,
  step id secondary. Color encodes status: `completed` / `completed with note` /
  `warning` / `failed`, and `awaiting decision` at a gate.
- **Forks = decision gates.** A `decide_*` step branches into its eligible methods
  (`ALLOWED_METHODS` in `src/broadway/timeline/decide.py`). The recorded
  `AnalysisDecision.method` is the taken branch; the other eligible methods are drawn
  as greyed, dashed "not taken" branches.
- **HEAD = frontier.** A marker on the next pending step (the current position in the
  investigation).
- **Edges = dependency.** `produces` / parent-child, drawn as clean left-to-right
  lines, ordered by the walkthrough sequence (not sort order).

## Mapping

| learnGitBranching | Broadway |
| --- | --- |
| commit | `AnalysisStep` |
| branch | eligible method at a `decide_*` fork (`ALLOWED_METHODS`) |
| checked-out branch | `AnalysisDecision.method` |
| HEAD | frontier step (`awaiting decision` / next `not started`) |
| commit color | step status |
| tag | sample / source stamped on the step |

## Proposed changes (ordered)

1. **Branch model** — derive forks from the walkthrough sequence's `decision` steps
   and `ALLOWED_METHODS`. Add a `taken` (chosen method) vs `not taken` (greyed)
   distinction. This is presentation-only: `AnalysisStep`/`AnalysisDecision` are the
   source of truth, unchanged.
2. **Layout** — layered/tiered left-to-right, aligned to the walkthrough order so the
   investigation reads as a line with two visible forks. Much tighter than the
   current Mermaid sort-order.
3. **Renderer** — static SVG (Python, matplotlib-free; build SVG text or use a
   minimal writer). Output `reports/lineage/graph.svg` alongside (replacing) the
   Mermaid `graph.md`. Keep `graph.json` as machine evidence. No JS, no interactivity;
   it only changes when the `lineage` command re-runs.
4. **Surface-integrity** — the SVG is linked from `reports/index.md`/`timeline.md`,
   covered by `test_surface_integrity.py` (link resolution + size cap). Extend the cap
   to cover `.svg`.

## Non-goals

- No interactivity, no pan/zoom, no command-entry sandbox. The CLI (`walkthrough` /
  `decide`) is the sandbox; the SVG just renders the resulting state.
- No new writer of `reports/index.md`.
- Not a replacement for `timeline.md` (the linear walkthrough) — this is the lineage
  DAG only. A combined view is a later decision.

## Sequencing

Deferred behind the active queue: small-multiples Q-Q + distribution -> docs re-sync
-> dogfood -> epsilon-squared -> registry refactor -> LoadAudit. Slot this in after
the dogfood pass (dogfood the current markdown surface first, then add the DAG).
