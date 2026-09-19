# GOVERNANCE-POINTER.md

Static signpost for the public platform line (`main`). Deliberately
minimal: the living governance ledger evolves on development branches and
is NEVER mirrored here — copies drift, and single-owner surfaces are law.

## Doctrine of this line, in full

1. The platform (`src/broadway/`) is data-agnostic: it knows no column
   names, thresholds, or dataset terms. Dataset identity lives in
   `configs/` and project bindings only.
2. Every claim must be verifiable and transparently evidenced — traceable
   to a primary source a fresh reader can resolve. A mention of a thing
   is not proof the thing exists.
3. Evidence → decisions → lineage is first-class: results carry
   provenance; analytical decisions are recorded artifacts.

## Where the living ledger lives (development branches)

- `agents/ledger/HANDOFF.md` — working style & doctrine (SSOT)
- `agents/ledger/STATE.md` — living agent context + EVENTS registry
- `agents/ledger/DECISIONS.md` — ratified decisions (law, append-only)
- `agents/contracts/` — orchestrator / worker / reviewer contracts
- `agents/ledger/gates.yaml` — machine-readable gate/surface registry

This file records pointers only, so it cannot rot. Last reconciled
against dev tip `sklearn@3fde83c`, 2026-08-26, by human ruling
(project-board Q1, option A: pointer only).
