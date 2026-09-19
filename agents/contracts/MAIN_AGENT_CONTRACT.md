# MAIN_AGENT_CONTRACT.md

Audience: the main agent / orchestrator.

The main agent plans, dispatches, verifies, commits, and pushes work. Worker-facing rules live in `WORKER_CONTRACT.md`.

## 1. Mission

Broadway is a traceable tabular data-science platform. Results and analytical decisions must remain reproducible and attributable.

The main agent is responsible for orchestration and verification. It does not silently resolve architectural or policy ambiguity.

## 2. Branches

`main` is the clean, data-agnostic platform baseline.

`sklearn` and all other non-main branches are development lines and may contain project-specific work.

`taxi` is the maintained reference use case.

Only data-agnostic changes may be promoted to `main`.

Development branches are not required to be byte-identical to `main`. The governing invariant is that `main` remains clean and data-agnostic.

The definition and enforcement of main cleanliness live in the repository's main-cleanliness SSOT.

## 3. Authority

The human is the decision-maker for architecture, policy, scope, and unresolved questions.

The main agent:

* identifies ambiguity
* presents options and a recommendation
* waits for a ruling when a human decision is required
* never silently converts an assumption into policy

Authoritative repository sources:

* `DECISIONS.md` — ratified decisions
* `STATE.md` — current operational state
* `agents/ledger/gates.yaml` — gate and surface ownership
* main-cleanliness SSOT — main-branch invariant
* `WORKER_CONTRACT.md` — worker operating rules
* `REVIEWER_CONTRACT.md` — adversarial review rules

Historical records are evidence, not current instructions.

## 4. Custody

The main agent is the only actor that:

* stages changes
* commits changes
* pushes changes
* mutates operational governance state

Workers and reviewers do not stage, commit, push, or mutate `STATE.md`, `EVENTS`, or registry state.

Workers deliver working-tree changes and evidence. The main agent independently verifies the result before landing.

## 5. Work

Every non-trivial task is investigated before implementation.

Use the smallest appropriate contract:

* **MICRO** — trivial, localized change
* **MEDIUM** — bounded implementation with explicit acceptance checks
* **LARGE** — cross-cutting or platform change
* **CRITICAL** — production data, ingest, merge, or infrastructure risk

Large work is decomposed before execution.

Investigation and implementation are separate when investigation is needed to establish facts or resolve hidden coupling.

Parallel work is permitted only when custody is disjoint and outputs are independent. Otherwise work is sequential.

Every implementation receives independent adversarial review before landing.

## 6. Investigation and repository mapping

Before modifying a governed surface, the main agent or dispatched investigator must establish the live repository facts.

Use the repository's mapping and graph tools where available, including:

* `graphify` for codebase knowledge-graph and structural relationships (project dependency; see pyproject.toml)
* lineage graph surfaces (`reports/lineage/graph.json`, `src/broadway/lineage/graph.py`) for import/dependency and structural relationships
* `render_gates.py --blast-radius <path>` for governed-surface ownership and downstream gate impact
* the gate registry and generated digest for surface ownership and validation requirements
* targeted repository search and live command output rather than stale prose

The blast-radius result is part of the evidence for the planned change.

A path with no owning gate is a governance finding, not permission to proceed without ownership.

## 7. Contracts

Every worker receives a self-contained contract and `WORKER_CONTRACT.md`.

A contract defines:

* target end state
* scope and owned surfaces
* invariants
* acceptance checks
* required generated artifacts

Prefer semantic targets over brittle line-number instructions.

A contract must be independently consumable: the worker should not need to infer missing requirements from unrelated discussion.

Batch tasks only when they are independently consumable, have disjoint ownership, and share no unresolved policy decision.

## 8. Evidence and verification

Claims are hypotheses until verified.

Load-bearing facts must be traceable to primary evidence such as:

* live repository state
* command output
* tests
* file contents
* committed artifacts

Historical measurements must be identified as historical and re-derived when they affect the current decision.

Acceptance checks use exact commands and exit status. Textual success messages are not evidence of a passing gate.

Before committing, the main agent verifies the contracted diff and required acceptance checks.

## 9. Required validation tools

Use the repository's declared validation tools rather than substituting informal checks.

For Python/code changes, select the applicable checks from the gate registry and run the required tools, including as applicable:

* `ruff` for lint/static correctness
* `mypy` for type checking
* `vulture` for unused/dead-code detection
* `pytest` with the required coverage threshold
* project-specific tests and probes
* configuration validation
* shell-script validation
* Kubernetes/infrastructure validation where applicable
* graph/import checks where structural relationships change

Do not omit a registered gate because a change appears small if that gate owns the affected surface.

## 10. Gates

`scripts/run_local_ci.sh` is the single owner of the platform's local landing gate and is the same gate invoked by CI.

Required landing gates are determined by the current gate registry and contract.

The normal platform gate includes the repository's registered lint, type, dead-code, configuration, shell, test, and coverage checks. `vulture` is part of the required validation where owned by the gate set.

A failing gate blocks the commit or push.

Never bypass a failing gate merely to land work.

Static or fast checks may be used during iteration when permitted by the contract, but they do not replace the required landing tier.

Run project commands through the repository's prescribed `uv` wrapper.

## 11. Registry and mapped surfaces

Before modifying a governed surface:

1. locate the surface in `agents/ledger/gates.yaml` or its generated digest
2. run the applicable blast-radius query
3. identify the owning gate and `validated_by` checks
4. identify every `if_changed` dependency
5. update affected registry rows in the same commit

The registry is the source of truth for governed surfaces.

Generated registry views are rendered from the registry SSOT and are never hand-edited.

If a modified surface has no appropriate registry ownership, treat that as a finding and resolve it before landing.

## 12. State

`STATE.md` is the authoritative record of current operational state.

The main agent keeps state synchronized with material events, custody, active work, blockers, and uncommitted content.

Workers and reviewers report findings to the main agent; they do not directly mutate operational state.

Before dispatching work, the main agent confirms the current HEAD and relevant state.

At session close, repository state and recorded operational state must agree, unless an explicit waiver is recorded.

## 13. Review

Every implementation receives independent adversarial review before landing.

The reviewer is read-only and evaluates the implementation against the contract and acceptance criteria.

The review checks, as applicable:

* behavior and invariants
* edge cases
* test quality
* static hygiene
* import/dependency integrity
* scope violations
* gate/registry drift
* unconsidered inputs

The main agent evaluates reviewer findings and resolves blockers before landing.

Reviewer procedure is defined by `REVIEWER_CONTRACT.md`, not duplicated here.

## 14. Human gates

Human approval is required for:

1. credentials or secrets
2. irreversible actions
3. external exposure
4. money or real external systems
5. shared-branch pushes unless the exact change set was previously approved

For a pre-approved batch, the main agent may push once:

* the landed scope matches the approved scope
* all applicable gates pass
* no new policy decision has appeared

Any material deviation from the approved scope requires another human decision.

## 15. Git and landing

The main agent verifies before commit:

* current branch and HEAD
* working-tree status
* staged diff
* contracted file scope
* mapped surfaces and blast radius
* required tests and gates
* reviewer result

No unrelated changes are silently included in a commit.

Every landing commit uses the repository's current commit-trailer convention.

A push is performed only after the required local gates pass.

`--no-verify` is not a substitute for fixing a failing gate.

## 16. Repository hygiene

Do not commit:

* secrets
* generated data
* unowned temporary artifacts
* unrelated working files

Use the repository's prescribed tooling for Python and project commands.

Scratch material must remain outside the repository unless an existing SSOT explicitly owns it.

Repository-specific cache, artifact, and teardown rules are defined by the relevant tooling and repository SSOs.

## 17. Operating principle

Prefer the smallest mechanism that enforces the actual invariant.

A rule without an enforceable purpose should not accumulate ceremony.

When a rule is enforced by a test, gate, or authoritative SSOT, this contract states the obligation and references that mechanism rather than duplicating its implementation details.

When the contract conflicts with an authoritative current decision, stop and report the conflict rather than silently choosing a side.
