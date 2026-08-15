# TODO — current state / next / deferred

Ephemeral task tracker. `HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

—

## Next

- **Q-Q doc pass** — `README.md`/`dataflow.md` Q-Q sections predate the zones/markers/raw-log work; `stats/API.md` drift; raw/log comparison + marker legend not yet documented.

## Deferred

- **W2 main-sync** — taxi-free `main`; convert/exclude taxi-referencing tests (`test_contracts.py` real-parquet load, hardcoded taxi stats in `test_walkthrough.py`/`test_results.py`).
- **Cleanup/polish slice** — orphaned legacy results-index (`render_index`/`load_stats_sequence`/`RESULT_RENDERERS`), doc drift, taxi strings in `audit.py`, dead code, silent posthoc skip, unlabeled Q-Q truncation.
- **LoadAudit / ParsingPolicy**.
- **Lineage-viz** (`graph_todo.md`).
- Move suggestion templates + effect-size wording into config; delete legacy `report` wrapper.
- Pin sample/evidence, then refresh reports.
- W1-flagged: Q-Q style constants Python-vs-YAML decision; hardcoded `figures/` path prefix.
