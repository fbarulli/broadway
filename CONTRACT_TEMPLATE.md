# CONTRACT_TEMPLATE.md — mandatory dispatch skeleton

Every agent contract fills this skeleton before dispatch (AGENT_CONTRACT.md
§3a/§3b). A contract the worker has to interpret is incomplete — it must
EXECUTE, not explore.

## Task

One or two sentences: the goal, the target end-state, and the scope boundary
(what is explicitly NOT in scope).

## Edit list (complete)

For every file that changes:
- path: exact current content → exact replacement (or the precise end-state
  for new files), plus every regenerated artifact the command sequence
  rewrites.
- If the worker would have to search to locate a target or a side-effect, the
  contract is incomplete.

## Constraints

- Immutable worker rules: `AGENT_WORKER_CONTRACT.md` (type hints on public
  functions, strategic logging only, catch only recoverable exceptions, YAML
  single source of truth, ~25-line single-responsibility functions, no dead
  code).
- Invariants: full suite green, no surface-ownership changes, no silent
  policy, backward compatibility.
- Files NOT listed in the edit list must not change.

## Acceptance checks (evidence format)

Numbered; each is an exact command + expected result + what the worker pastes
back (exit codes, counts, diffs, git status). Prefer structural checks over
substring greps when the artifact is structured (e.g. for HTML: assert the JS
sits between `<script>` tags, not merely that its text appears somewhere).

## Commit + push

- Branch: `taxi`. One commit, message: `<…>`.
- Push: `git push origin taxi`, no force. Report external failures; never
  force-push.
