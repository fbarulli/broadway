# MAIN_AGENT_CONTRACT.md

check if you can spin up other agents reliably. if you cannot due to provider rate limitation, consult human.

Audience: the main agent / orchestrator.

The main agent plans, dispatches, verifies, commits, and pushes work. Worker-facing rules live in `WORKER_CONTRACT.md`.

## 1. Mission

Broadway is a traceable tabular data-science platform. Results and analytical decisions must remain reproducible and attributable.

The main agent orchestrates and verifies; it does not resolve architectural or policy ambiguity on its own (see §3, Authority).

## 2. Branches

`main` is the clean, data-agnostic platform baseline.

All the others are treated as `dev`

Only data-agnostic changes may be promoted to `main`. Development branches are not required to be byte-identical to `main` — the governing invariant is that `main` stays clean and data-agnostic.

The definition and enforcement of main cleanliness live in the repository's main-cleanliness SSOT.

## 3. Authority

The human is the decision-maker for architecture, policy, scope, and unresolved questions.

The main agent identifies ambiguity, presents options and a recommendation, and waits for a ruling when a human decision is required. It never silently converts an assumption into policy, and never silently resolves a conflict between this contract and an authoritative current decision — either case is reported, not decided alone.

Authoritative repository sources:

* `DECISIONS.md` — ratified decisions
* `STATE.md` — current operational state
* `agents/ledger/gates.yaml` (or equivalent registry) — gate and surface ownership
* main-cleanliness SSOT — main-branch invariant
* `WORKER_CONTRACT.md` — worker operating rules
* `REVIEWER_CONTRACT.md` — adversarial review rules

Historical records are evidence, not current instructions.

## 4. Work

Every non-trivial task is investigated before implementation. Root cause is identified and the smallest step towards addresing it is the goal. No work-arounds.

Use the smallest appropriate contract:

* **MICRO** — trivial, localized change
* **MEDIUM** — bounded implementation with explicit acceptance checks
* **LARGE** — cross-cutting or platform change
* **CRITICAL** — production data, ingest, merge, or infrastructure risk

Large work is decomposed before execution. Investigation and implementation are separate when investigation is needed to establish facts or resolve hidden coupling.

Parallel work is permitted only when custody is disjoint and outputs are independent; otherwise work is sequential.

Every implementation receives independent adversarial review before landing (see §9, Review).

## 5. Contracts

Every worker receives a self-contained contract and `WORKER_CONTRACT.md`.

A contract defines: target end state, scope and owned surfaces, invariants, acceptance checks, and required generated artifacts.

Prefer semantic targets over brittle line-number instructions. A contract must be independently consumable — the worker should not need to infer missing requirements from unrelated discussion.

Batch tasks only when they are independently consumable, have disjoint ownership, and share no unresolved policy decision.

## 6. Investigation, Registry, and Blast Radius

Before modifying a governed surface, the main agent or a dispatched investigator establishes the live repository facts and the surface's registry standing. This is one pass, not two separate steps:

1. Locate the surface in `agents/ledger/gates.yaml` (or its generated digest) — identify the owning gate(s), `validated_by` checks, and every `if_changed` dependency.
2. Run the applicable blast-radius query (`render_gates.py --blast-radius <path>`) for governed-surface ownership and downstream gate impact. The result is part of the evidence for the planned change.
3. Use the repository's mapping and graph tools for structural facts, including `graphify` for codebase knowledge-graph relationships (project dependency; see pyproject.toml) and lineage graph surfaces (`reports/lineage/graph.json`, `src/broadway/lineage/graph.py`) for import/dependency relationships. Prefer targeted repository search and live command output over stale prose.
4. A path with no owning gate is a governance finding, not permission to proceed without ownership. If a modified surface has no appropriate registry ownership, resolve that finding before landing.
5. Update affected registry rows in the same commit as the surface change.

Generated registry views are rendered from the registry SSOT and are never hand-edited. The registry is the source of truth for governed surfaces.

## 7. Evidence and Verification

Claims are hypotheses until verified.

Load-bearing facts must be traceable to primary evidence: live repository state, command output, tests, file contents, or committed artifacts. Historical measurements must be identified as historical and re-derived when they affect the current decision.

Acceptance checks use exact commands and exit status. Textual success messages are not evidence of a passing gate.

Before committing, the main agent verifies the contracted diff and required acceptance checks.

## 8. Gates and Validation

`scripts/run_local_ci.sh` is the single owner of the platform's local landing gate and is the same gate invoked by CI. Required landing gates are determined by the current gate registry and contract.

For Python/code changes, select the applicable checks from the gate registry and run the required tools, including as applicable:

* `ruff` for lint/static correctness
* `mypy` for type checking
* `vulture` for unused/dead-code detection (part of the required validation where owned by the gate set)
* `pytest` with the required coverage threshold
* project-specific tests and probes
* configuration, shell-script, and Kubernetes/infrastructure validation where applicable
* graph/import checks where structural relationships change

Do not omit a registered gate because a change appears small if that gate owns the affected surface. Use the repository's declared validation tools rather than substituting informal checks, and run project commands through the repository's prescribed `uv` wrapper.

A failing gate blocks the commit or push. Never bypass a failing gate merely to land work. Static or fast checks may be used during iteration when the contract permits, but they do not replace the required landing tier.

## 9. Review

Every implementation receives independent adversarial review before landing.

The reviewer is read-only and evaluates the implementation against the contract and acceptance criteria, checking as applicable: behavior and invariants, edge cases, test quality, static hygiene, import/dependency integrity, scope violations, gate/registry drift, and unconsidered inputs.

The main agent evaluates reviewer findings and resolves blockers before landing. Reviewer procedure is defined by `REVIEWER_CONTRACT.md`, not duplicated here.

## 10. State

`STATE.md` is the authoritative record of current operational state.

The main agent keeps state synchronized with material events, custody, active work, blockers, and uncommitted content. Workers and reviewers report findings to the main agent; they do not directly mutate operational state.

Before dispatching work, the main agent confirms the current HEAD and relevant state. At session close, repository state and recorded operational state must agree, unless an explicit waiver is recorded.

STATE record ids use the `STATE-YYYYMMDD-NNN` shape; the date segment is a declared record-id namespace, not a commit hash. Ledger prose must not emit a bare 8-hex digit run — write dates as `YYYY-MM-DD` (enforced by the 8-hex governance probe).

## 11. Custody, Git, and Landing

The main agent is the only actor that stages, commits, pushes, or mutates operational governance state (`STATE.md`, `EVENTS`, registry state). Workers and reviewers deliver working-tree changes and evidence; the main agent independently verifies the result before landing.

Before every commit, the main agent verifies: current branch and HEAD, working-tree status, staged diff, contracted file scope, mapped surfaces and blast radius, required tests and gates, and reviewer result. No unrelated changes are silently included in a commit.

Every landing commit uses the repository's current commit-trailer convention. A push is performed only after the required local gates pass. `--no-verify` is not a substitute for fixing a failing gate.

Do not commit secrets, generated data, unowned temporary artifacts, or unrelated working files. Scratch material stays outside the repository unless an existing SSOT explicitly owns it. Use the repository's prescribed tooling for Python and project commands; repository-specific cache/artifact/teardown rules are defined by the relevant tooling and SSOTs.

## 12. Human Gates

Human approval is required for:

1. credentials or secrets
2. irreversible actions
3. external exposure
4. money or real external systems
5. shared-branch pushes, unless the exact change set was previously approved

For a pre-approved batch, the main agent may push once: the landed scope matches the approved scope, all applicable gates pass, and no new policy decision has appeared. Any material deviation from the approved scope requires another human decision.

## 13. Operating Principle

Prefer the smallest mechanism that enforces the actual invariant. A rule without an enforceable purpose should not accumulate ceremony.

When a rule is enforced by a test, gate, or authoritative SSOT, this contract states the obligation and references that mechanism rather than duplicating its implementation details.