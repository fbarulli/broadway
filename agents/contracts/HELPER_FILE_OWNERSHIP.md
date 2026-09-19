# HELPER_FILE_OWNERSHIP.md — ownership rule for small helper files

Class declaration (per R6 below): **SSOT** — single source of truth for
helper-file ownership law. Nothing else in the tree restates or renders it.

## 1) Purpose

Small helper files — viz/reporting/glue utilities living outside the platform
`src/broadway` surface — historically landed with no owner of record. With no
owner, a registry probe had no authority to consult, and that gap produced two
recurring incident classes: false "no caller / orphaned" claims raised against
helpers whose real consumers sat outside the probed lane, and genuinely
orphaned helpers that survived because nobody owned their lifecycle enough to
delete them. Both are ownership failures, not tooling failures; this file
closes the gap by making ownership explicit at file granularity.

## 2) Definition

A **small helper file** is a standalone support/glue script OUTSIDE the
`src/broadway` platform surface, typically < ~150 lines — e.g. plotting,
export, or report-rendering utilities. Excluded: `experiments/**` analysis
scripts (governed by their own conventions under MAIN_AGENT_CONTRACT §14) and
anything inside platform `src/` (already covered by ordinary surface law).

## 3) Rules

- **R1 — owner-of-record.** A file-top comment `# owner: <lane-or-role>` is
  REQUIRED before first productive use. This mirrors the registered
  lifecycle-owner law (MAIN_AGENT_CONTRACT §14 Cache & teardown law: new
  durable object creators require a registered lifecycle owner before first
  use).
- **R2 — production-caller-or-delete.** Every landed function in a helper has
  a production caller by its commit, or a documented extension point plus a
  test pin. "Tests are consumers" is false. The sizing/orphan-vs-dead law is
  imported VERBATIM BY REFERENCE from MAIN_AGENT_CONTRACT §7 (**Sizing**);
  helpers receive no exemption from it.
- **R3 — single-writer.** Helper edits ride the owning lane's contract only;
  a cross-lane edit requires re-registration (owner change) FIRST.
- **R4 — placement.** Executable ops helpers → `scripts/`; agent tooling →
  `agents/tools/`; analysis → `experiments/`. NEVER repo-root scratch
  (scratch siting, MAIN_AGENT_CONTRACT §6).
- **R5 — registry.** Helpers touching gated surfaces enter
  `agents/ledger/gates.yaml` rows like any other surface (§14 Registry
  maintenance duty).
- **R6 — class declaration.** Every helper declares its class at creation:
  `SSOT | derived | cycle-scoped` (§14 Ledger & artifact hygiene).

## 4) Relationship to existing law

- MAIN_AGENT_CONTRACT §7 **Sizing** — production-caller + orphan-vs-dead law,
  imported by reference for R2.
- MAIN_AGENT_CONTRACT §14 **Cache & teardown law** — lifecycle-owner
  registration duty mirrored by R1.
- MAIN_AGENT_CONTRACT §14 **Registry maintenance duty** + hygiene classes —
  gates.yaml row duty behind R5; SSOT/derived/cycle-scoped vocabulary behind
  R6.
- WORKER_CONTRACT **Custody** — a helper is editable only via its owning
  lane's dispatched worker; R3 restates that custody rule at helper
  granularity.
