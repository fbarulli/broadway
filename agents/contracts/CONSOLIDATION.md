# CONSOLIDATION.md — one canonical implementation per behavior (SSOT)

Owner: canonical-helper law.

This law exists because the factory's most common defect is not a wrong function —
it is **two** functions that do the same thing, in different owners, drifting apart.

## 1. Rule

Before writing code, search for the behavior. If it exists, use it. If it exists in
the wrong place, **move it to its canonical owner and re-point every caller** in the
same change. A near-duplicate is a defect even when both copies work.

"Tests are consumers" is false. A helper has a production caller at commit time, or
it is deleted.

## 2. Placement

| behavior | canonical owner |
|---|---|
| reusable analysis/ML machinery | `src/broadway/<domain>/` |
| dataset binding (config + thin glue) | `project/` |
| one-off analysis step | `project/experiments/<series>/` |
| deterministic operations | `scripts/` |
| small support/glue file | `HELPER_FILE_OWNERSHIP.md` |

A new file declares its owner, its class (`SSOT | derived | cycle-scoped`), and its
production caller. No owner means no place in the tree.

## 3. Duplication classes (all fail review)

- **D1 — copied block.** A block of logic repeated across files or experiments.
- **D2 — near-duplicate function.** Same behavior, different name, formatting, or
  parameter order.
- **D3 — parallel registry/config.** A second list, default, path, or threshold that
  restates an SSOT value.
- **D4 — reimplemented library behavior.** Hand-rolled statistics, splits, encoding,
  metrics, or persistence that an installed library provides (`CODE_STANDARD.md` §3).
- **D5 — experiment reimplementation.** A step under `project/experiments/` that
  re-defines a `src/broadway` helper instead of importing it.

## 4. Detection (machine, not memory)

- Duplicate-block detector over `src/`, `project/`, `scripts/` with the threshold in
  config; report exact locations.
- Near-duplicate AST scan (normalized identifiers) for D2.
- Import-boundary scan: experiments and scripts import platform helpers; a local
  redefinition is a finding.
- Cross-owner name scan: the same function name under two owners is a pointer to
  verify, not proof of duplication.
- New-helper check: owner + production caller + class declaration present.

## 5. The consolidation move

One commit: move to the canonical owner, update every importer, delete the original,
and pin the behavior with one focused test. Mixed consolidation plus feature work is
refused — it hides the regression surface.

## 6. Enforcement

- The duplicate and import-boundary checks start advisory, then become blocking when
  the baseline is clean; the threshold lives in config, not in the script.
- The reviewer searches for an existing implementation before approving new code and
  names the canonical owner in the verdict.
- `vulture` owns dead-code detection; `HELPER_FILE_OWNERSHIP.md` owns helper
  registration.
