# AGENT_WORKER_CONTRACT.md — immutable worker rules

These rules never change. Apply them to every change, every time.

- No hardcoded values.
- ALWAYS present decisions to the user; NEVER decide unilaterally.
- Type hints on all public functions.
- Strategic logging only (stage boundaries, results, errors; never inside loops).
- Catch exceptions only when recoverable; let everything else bubble up.
- YAML = single source of truth: no `get(key, default)`, no hardcoded values.
- ~25-line functions; single responsibility; no dead/noise code.
- Derive, don't maintain: never write state that can be computed at render
  time (no caches, snapshots, or derived-status files) — compute from the
  tree/records instead. The platform derives; it does not store derived state.
