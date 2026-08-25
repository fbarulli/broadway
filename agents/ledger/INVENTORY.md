# INVENTORY.md — dead-code / stub module inventory (D11 Tier-4)

Class: SSOT — the single record of platform modules that are declared but
unbuilt (stubs), or dead-and-deleted. D11 ruling (DECISIONS.md): stub modules
consolidate here (name / one-line intent / why unbuilt) BEFORE deletion, so the
deletion commit carries its rationale and no surface disappears silently.
Straight-delete items (test-only exports, dead config fields) need no entry.

Deletion-first: entries are the WHY, never a promise to build.

## Deleted 2026-08-24 (src/broadway/trust/) — committed with this inventory

The `trust` package was the trust/monitoring pillar of the platform plan
(FEEDBACK.md, GOALS.md): structural modules were reserved in the build order but
never implemented beyond a docstring each. No production importer existed (gap-
topology C2 confirmed zero external references; the package survived only through
its own test suite). Deleted per D11 as stubs; if the trust/monitoring pillar is
ever built, it is re-introduced against real consumers, not pre-reserved.

| module | one-line intent (docstring) | why unbuilt |
|---|---|---|
| trust/__init__.py | package marker | never implemented |
| trust/drift.py | PSI, KS test, KL divergence — train vs serve distribution comparison | trust pillar structurally reserved, not built (FEEDBACK #1) |
| trust/fairness.py | subgroup disparity, equal opportunity, demographic parity | trust pillar structurally reserved, not built |
| trust/interpretability.py | SHAP summary/dependence plots, permutation importance | trust pillar structurally reserved, not built |
| trust/leakage.py | target leakage detection — correlation with timestamps, ID-based checks | trust pillar structurally reserved, not built |
| trust/module.py | orchestrates trust submodules — run post-train or on schedule | trust pillar structurally reserved, not built |
| trust/sensitivity.py | perturbation analysis — how do predictions change under input noise? | trust pillar structurally reserved, not built |
| trust/uncertainty.py | prediction intervals (bootstrap), conformal prediction | trust pillar structurally reserved, not built |

## Open (declared, unbuilt, still present)

`src/broadway/unsupervised/` and `src/broadway/selection/` are the same class of
dead package (gap-topology C2: zero external importers, absent from STEP_MODULES)
and remain in the tree pending a parallel D11 disposition.
