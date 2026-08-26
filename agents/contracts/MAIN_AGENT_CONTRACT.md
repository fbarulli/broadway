# MAIN_AGENT_CONTRACT.md — orchestrator operating rules (single source)

Audience: the main agent. Everything needed to plan, dispatch, verify, and
integrate work. Worker-facing rules live in `WORKER_CONTRACT.md`; every worker
instruction carries that file (or a hard reference to it).

Ask questions in output, alongside observations. Always ask if you have them.
**Custody (ratified 2026-08-23):** workers never commit and never stage — they
deliver working-tree changes plus a report; the main agent runs every gate
itself, commits only when all are green, and pushes (per-push human go unless
the human authorized the batch).

## 1. What Broadway is

A **traceable tabular data science platform** (Python, `uv`-managed). Every
result carries provenance and every analytical decision is a recorded artifact.
**Evidence → decisions → lineage** is first-class.

Working directory: `/home/opc/ONE/broad-way`.

## 2. Branches

- **`sklearn`** — project home, the ONLY active line. All code work happens here.
- **`taxi`** — demo pass-along: fast-forwarded to `sklearn`'s tip after each
  green push; never diverges.
- **`main`** — frozen until the human declares main-day; sync then via the
  parity checker with its full surface. The era (dev|main) is declared ONLY
  inline in `scripts/check_branch_parity.sh` — the single era declaration
  (D16; relocated inline by D21); the
  checker and the pytest gate both read that declaration, never an
  environment variable.
- **`broadway`** — stale/legacy. Do not touch.

## 3. Roles

- **Decision-maker / architect** — the human: direction, open questions,
  principles.
- **Main agent (orchestrator)** — plans, writes contracts, dispatches,
  verifies, commits, pushes. Does NOT implement.
- **Sub-agents** — execute one contract against `WORKER_CONTRACT.md`.
- **Adversarial reviewer** — read-only approach reviewer against the
  working-tree delta (registered authority per EVENTS 39de4245; protocol:
  `REVIEWER_CONTRACT.md`). Verdicts BLOCKER/SHOULD-FIX/NOTE land in the
  session record; enforcement lives in gates, not in a review tier.

The main agent does not decide unilaterally. Weigh tradeoffs → options +
recommendation → wait. Honesty over agreement: flag problems early; "not being
a yes-man" is a stated value. A silent decision is a defect.

## 4. Dispatch architectures — pick per task

Five mechanisms, chosen by risk and test-expressibility; they compose.

1. **Two-phase (investigate → implement).** Design-heavy or hidden-coupling
   work: Phase A is a read-only investigator answering pointed written
   questions against live code AT DISPATCH TIME, producing a dated fact sheet;
   Phase B's implementer brief cites that sheet instead of frozen prose. The
   human reviews the short fact sheet, not a long brief.
2. **Test-first (the suite as spec).** When a fix can be expressed as a failing
   test, land the tripwire FIRST as its own tiny contract; the implementation
   contract reduces to "make these tests pass, touch nothing else." Prefer over
   edit lists wherever a tripwire can be written.
3. **Bounded-judgment grants.** Contracts enumerate judgment DOMAINS the worker
   MAY exercise (internal naming, decomposition within size limits, fixture
   mechanics, comment wording) plus acceptance properties that must hold.
   Scope, behavior/policy, public surfaces, schema/config semantics stay
   reserved. Ambiguity outside the grant halts to an OPEN QUESTION. Defaults
   codified in `WORKER_CONTRACT.md`.
4. **Assumption audit** — mandatory in every report, every tier (see §6).
5. **Adversarial second agent.** After any implementation settles — EVERY
   tier, no exceptions — a fresh read-only reviewer attacks the diff hunting
   vacuous tests, silent behavior change, and unconsidered input classes; the
   main agent arbitrates findings before verification closes and nothing
   commits un-reviewed. Plan-level slates get their own red-team panel
   (scope/necessity, sequencing/risk, fact-check) before execution.

Selection guide: Micro/Medium keep the plain format (+ assumption audit);
behavioral fixes prefer test-first; design-heavy/coupled tasks run two-phase;
every tier ends with the adversarial review of #5; critical-path data/infra
adds interactive human checkpoints at each decision point.

## 5. Fact discipline (live checks beat frozen prose)

Prose facts are snapshots that go stale the moment the tree moves (proven
2026-08-23: hand-derived measurements were unrepresentative; a reviewer caught
more reading `parse_numeric` than any checklist produced).

- Brief facts are **hypotheses until re-derived at run time**. Prefer
  executable checks whose output is pasted over stated measurements; label any
  cited measurement historical and pair it with the live re-derivation command.
- **Step-0 hash gate (mandatory):** worker's first action is
  `git rev-parse --short HEAD` against the dispatch stamp; mismatch → STOP
  before reading further.
- Stamp semantics (dispatch stamps are RELATIVE — HEAD-at-dispatch-time,
  never hardcoded absolute SHAs; absolute SHAs are provenance anchors
  only). This line is the whole doctrine; no external appendix.
- **Running ledger:** chained batches carry an actuals-only ledger in the
  governing index file; actual ≠ projected halts the queue until reconciled.

## 6. Verification (evidence, not claims)

- Acceptance checks: exact commands, expected exit codes/counts, evidence
  format to paste. A report without evidence is incomplete. Counts are exact —
  no approximate pass conditions. Every gate paste names its exact command
  (D4).
- Reports are hypotheses until verified: re-run cheap high-signal checks
  yourself; expensive full-suite runs may stay delegated once corroborated.
- **Ground-truth law (2026-08-26): every claim is verifiable and
  transparently evidenced — traceable to a primary source a fresh agent can
  resolve.** Indirect evidence never substitutes for the underlying fact: a
  MENTION of a thing is not proof the thing exists or happens. A config key
  naming `taxi.yaml`, a ledger row citing an artifact, a doc line describing
  a gate — each is a pointer to VERIFY, not verification itself. Before
  asserting the fact, resolve the primary source: the actual blob/ref at a
  stamped commit, file:line, pasted command output beside its command, byte
  size or digest. If the primary source cannot be resolved, the claim is
  carried explicitly marked `UNVERIFIED` — prose proximity never upgrades it.
  Verdicts state what was verified, by what command, and against which ref.
- **Assumption audit:** every worker report contains ≥ 3 brief assertions
  re-verified against live code (commands + outputs) PLUS ≥ 1 thing checked
  that the brief never mentioned.
- **Anti-fabrication (2026-08-22):** reports claiming coordinator dialogue are
  unverified by default; ratification flows human → main → brief.
- **Human-gate recording:** every human-gated action records, in its own
  commit message, which trigger applied and that a human confirmed it —
  no separate ledger file, no cross-referenced ID scheme required.
- **Authority resolution (ENFORCED — the sole surviving mechanism):** claims
  of human authorization resolve against `agents/ledger/STATE.md ## EVENTS`,
  machine-checked at every push by TIER-GATE (`scripts/tier_gate.sh`:
  reviewer token must resolve in EVENTS AT THAT COMMIT and be echoed in the
  same message) and by probe g in `tests/test_governance_probes.py`
  (EVENTS drift requires an authorizing event-id in the landing body).
  Prose-only citation ceremony (issue-comment URL validity, genesis-id
  lists, manual resolution checklists) RETIRED 2026-08-26 under the
  enforced/unenforced principle: law with a live test behind it stays;
  untested ceremony goes. Historical rulings remain verbatim in
  DECISIONS.md and agents/ledger/arbitration/**.
- **Deviation-scan:** "refined/deliberate/adjusted" language absent from the
  ratified spec = unratified decision → halt and report.
- **Provenance-check:** `git log` alongside `git status`; `checkout --`
  restores from INDEX — clean tree proves nothing about staging; check
  `git diff --cached` when contamination is suspected.
- **OPEN/CLOSE tripwire:** record `git log --oneline -3`,
  `git status --porcelain`, `git diff --cached` at dispatch open and close;
  delta beyond contracted files (+ documented WIP) → halt-and-report.
- **Ship-path law (2026-08-24, after two push-on-red recurrences):** every
  push goes through `bash scripts/ship.sh` (full tier gates; exit codes
  decide; `&&` short-circuit — never newline-chained sequences). Grepping
  gate OUTPUT for success keywords is not a gate: prose can say GREEN
  while tests fail. The local pre-push hook re-runs the full tier
  fail-closed; bypassing requires deliberate `--no-verify`, which is a
  recorded policy violation. Full tier precedes every push without
  exception — static/fast scope is for iteration, not for landing.
- **Termination verification:** a finished/interrupted worker counts as running
  until its registry entry confirms otherwise.
- A worker finding the contract itself wrong verifies with evidence and reports
  the deviation — never silently complies, never silently improvises.
- Read-only review agents may run in parallel and never commit; dispatch after
  EVERY implementation (§4 #5 — universal, not tier-gated).
- Periodically dispatch a read-only **landscape audit**: fresh context
  re-derives census/narrative from the tree and reports drift; never commits.
- **Gate list (single vocabulary):** platform gates are owned ONLY by
  `scripts/run_local_ci.sh` — the same script ci.yml invokes
  (anti-D15/D16d-C7). Every Medium+ contract runs
  `bash scripts/run_local_ci.sh` and pastes the five PASS/FAIL banners before
  commit; doc-only micro edits may pass `--static`. Docker-only CI checks
  (shellcheck/kubeconform/orchestrator dry-run/build-and-boot) are named in
  the script header. A push is authorized only when local tiers are green AND
  the branch tip's last CI run is green.

- **Standing adversarial vectors (every review mandate carries them):**
  1. *Static-hygiene* — reviewer runs ruff (`F,E9` error-class + default rules)
  over the exact diff scope and probes import-graph integrity for moved names;
  re-export/facade modules are NEVER autofixed wholesale (none exist at
  HEAD since the b15f66e-era cleanup deleted `_common.py`/`_setup.py`; if
  any reappear, verify live import sites BEFORE assuming volume) —
  alias-form or `__all__` only.
  2. *Gate-divergence watch* — any ci.yml change must land through
  `run_local_ci.sh`; a reviewer greps ci.yml for non-comment gate commands and
  fails the review if the script is no longer the single owner.
  3. *Coverage-gaming* — new tests must assert real behavior; a test executing
  a line without meaningful asserts is rejected; branch-level dead arms
  (e.g. `n==1 → NaN`) get explicit pins or a documented pragma justification.

- **Two-author test rule:** the implementing worker writes primary behavior
  tests; a SEPARATE red-team test author receives only the contract spec
  (never the diff) and contributes edge/adversarial tests against it before
  merge. New tests may not be WEAK/TAUTOLOGY-class; every new raise site gets
  a `match=` pin; mutation spot-probes run periodically as a standing audit.

- **Dispatch context rule:** every dispatch (worker, reviewer, adversary,
  senior) opens with a CONTEXT block — current HEAD SHA, lanes in flight
  with agent ids, files under other lanes' custody, path to
  `agents/ledger/STATE.md` — and the receiving agent's step-0 becomes:
  read STATE.md, echo any contradiction with dispatch instructions, STOP
  stale-on-arrival instead of improvising. Main agent refreshes STATE.md
  at every material event (BOARD-CHECKPOINT DISCIPLINE 2026-08-26,
  superseding arbitration/push-only); a dispatch without a CONTEXT block
  is incomplete.

- **Live ops board (GitHub Projects, ratified 2026-08-26):** the
  AVAILABILITY layer for lane/checkpoint state — survives local loss and
  session death; STATE.md stays the PRIMARY record and every card body
  points back to it. Project: #4 "Broadway Ops Board", owner `fbarulli`,
  project node id `PVT_kwHOAZFnCc4Bhhjq`,
  https://github.com/users/fbarulli/projects/4 . Update it at EVERY
  material event — same triggers as the STATE.md checkpoint discipline.
  Mechanics (figured out live 2026-08-26; gh ≥2.96 behavior):
  * Auth once: `gh auth refresh --hostname github.com -s read:project,project`
    (device flow; plain tokens lack project scopes).
  * ADD a card — `gh project item-add` takes ONLY `--url` (existing
    issues/PRs); draft cards go through GraphQL:
    `gh api graphql -f query='mutation($p:ID!,$t:String!,$b:String!){addProjectV2DraftIssue(input:{projectId:$p,title:$t,body:$b}){projectItem{id}}}' -f p=PVT_kwHOAZFnCc4Bhhjq -f t="TITLE" -f b="BODY"`
    Payload field is `projectItem`, NOT `item`; a GraphQL error rejects
    the WHOLE query — no partial cards land.
  * SET Status — Status field id `PVTSSF_lAHOAZFnCc4Bhhjqzhgc_SQ`
    (re-derive anytime: `gh project field-list 4 --owner fbarulli`);
    option ids Todo `f75ad846` · In Progress `47fc9ee4` · Done `98236657`;
    `mutation($p:ID!,$i:ID!,$f:ID!,$o:String!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$o}}){projectV2Item{id}}}`.
  * CARD HYGIENE: title prefixes carry state vocabulary —
    `[LANDED-PENDING]` (diff presented, awaiting human go) · `Q#` (open
    question routed to human) · `[DONE RECORD]` (closed lane outcome);
    body MUST cite primary sources (file:line, byte sizes, refs) and name
    the STATE.md row it mirrors; each report is processed into the board
    EXACTLY once (registry discipline). These GitHub node ids are
    system-resolvable constants of this project, not ledger event-ids.

- **Commit trailer convention** (every main-agent commit, forward-looking):
  trailing lines `Contract: <id>`, `Gates: <gate verdict + suite tail>`,
  `Reviewer: <agent-id | none> <verdict>`, `Tier: FULL|CHECKLIST`,
  `Ledger: FIXES.md`. Gives every agent a queryable contract index via
  `git log --grep='^Contract:' --format='%h %s'` — the landed-state channel
  lives in git itself; STATE.md carries only what commits cannot (lanes,
  custody, hazards, open items). `Tier:` FULL|CHECKLIST — computed by MAIN
  AGENT ONLY via `scripts/tier_classifier.py` at staging time; never
  worker-declared; unknown/mixed ⇒ FULL.

- **Session-close rule:** no ratified state survives a session boundary
  unlanded — session close means `git status` reconciled against STATE.md's
  uncommitted-content section being empty or explicitly waived.
- **Heartbeat rule:** any dispatch expected >20 min carries mandatory
  heartbeats (current step + fresh tool-output tail); two missed/stale
  heartbeats ⇒ interrupt-and-reconcile. Exempts Micro contracts.
- **Ledger folding:** no standalone `Ledger:` commits except at batch
  boundaries — contract actuals fold into the contract commit body;
  FIXES.md rows update at batch close.
- **Scratch siting:** agent scratch NEVER lives at repo root — `mktemp -d`
  outside the repo, deleted before report. Root dot-dirs are hygiene-test
  failures.
- **Decomposition (D20):** every dispatch is MICRO or MEDIUM sized; LARGE
  plans decompose first. INVESTIGATE and EXECUTE are separate contracts;
  executors consume pasted fact sheets verbatim and may assert nothing
  absent from them. Parallel lanes require custody-disjoint +
  output-independent; default sequential.

## 7. Contract requirements

- Follow `CONTRACT_TEMPLATE.md` — skeleton mandatory. A contract the worker
  must interpret is incomplete.
- Target end-state and explicit invariants (suite green, no surface-ownership
  changes, no silent policy, backward compatibility) — not brittle line numbers.
- Self-contained: complete edit list + complete regenerated-artifact list; if
  a worker must search for a target or side-effect, the contract failed.
- Coarse plan up front; detailed contract just-in-time against the just-
  committed state. Re-read touched files immediately before dispatching.
- Batch only when ALL hold: ≤3 tasks, disjoint files, no shared contracts, no
  symbol-renaming refactor.
- Completed deferred items are REMOVED from queues; git history is the record.
- **Sizing:** one contract = one independently consumable behavior change;
  every landed function has a production caller by its commit ("tests are
  consumers" is false; "consumed next contract" is dead code); target ≤ ~5
  files / ~150 changed lines; multi-commit only when intermediates would be
  dead/broken on origin, pushed as one event; bloat scan before dispatch;
  census-rescope after any census miss; interface changes travel with callers.
  *Orphan vs dead:* an orphan has documented extension point/future consumer
  and is test-pinned (inventory, not findings); anything else unused within the
  change is deleted by the same contract.

## 8. Proportional process

- **Micro** (one-file tweaks): main agent directly; adversarial review for
  anything touching tracked surfaces beyond the edit's own doc.
- **Medium**: one worker, trimmed contract, main re-verifies cheap checks.
- **Large** (platform/`src`, cross-cutting): full ceremony — template, all
  gates, adversarial reviewer after (universal rule).
- **Critical** (production data paths, ingest/merge, infra): Large + two-phase
  investigation + interactive human checkpoint per decision point.

## 9. Pre-dispatch gate (non-negotiable)

Nothing launches until all seven gates are green; recorded in the brief:

1. Decisions consolidated — every open question has an explicit human answer;
   no partial ratification (contract-error corrections excepted: numbered
   amendment through the SAME thread, never silent).
2. Brief frozen first — written to its file, versioned, byte-consistent with
   what the human saw; dispatch references the file, not chat memory.
3. Single-writer window declared — one owner per surface per phase; concurrent
   amend/rebase of one branch by two writers prohibited (2026-08-23 race).
4. Contract self-containment verified against the just-committed tree.
5. Dispatch plan stated — agents, worker vs read-only, foreground/background,
   settle condition, who processes the result.
6. Registry discipline — agent ids logged; each report processed exactly once.
7. Post-settle protocol — independent re-verification of cheap high-signal
   checks; evidence + next decision presented to the human before push/merge.

## 10. Work-splitting

Independent tasks → parallel agents. Dependent tasks wait for upstream
commit + push. Workers run no git operations, so push races are structurally
impossible; the main agent pushes one branch at a time, only after
verification.

## 11. Coding style & docs

Immutable coding rules: `WORKER_CONTRACT.md`. Plots use seaborn unless stated.
Docs updated in the SAME commit as any change (`README.md` current at ALL
times: test counts, commands, paths; `dataflow.md`, `src/broadway/stats/API.md`,
`tests/README.md`). Scratch docs are never touched (`01.md`, `agents/notes/TODO_*.md`,
`GOALS.md`, `LEARN.md`, `trust.md`, `synth.md`, `project.md`,
`FEEDBACK.md`, `DATA_VALIDATION.md`, `GENERAL_TODO.md`, `project/STATS.md`;
`SENIOR.md` retired from this list and from the tree 2026-08-26).
`HANDOFF.md` is maintained on explicit user request.

## 12. Git & product policy

- Track `reports/**`; ignore `artifacts/`, `data/processed/`, `data/raw/`,
  `/results/`; never commit secrets or generated data.
- Results are the product surface; record evidence, don't auto-fix; config-
  driven everything; derive, don't maintain.
- Full suite required when `src/` or `tests/` changes; exit codes captured
  directly, never through a pipe; no broken intermediate state.

## 13. Quick reference

- `uv sync`; `uv run pytest -q`; `uv run ds-pipeline <command>`;
  `uv run python -m project.scripts.NN_name`.
- Agents: fresh subagents per contract; workers run no git operations at all —
  main agent verifies, commits, pushes.

## 14. Human gates & push discipline

Five human-check triggers. Everything not matching one of them is decided
by "gates pass" alone — no ruling, verdict, or board step exists or is
required:

1. Credentials / secrets.
2. Irreversible actions.
3. External-exposure changes.
4. Money or real external systems.
5. Every shared-branch push: the exact command and the full diff are
   presented to the human BEFORE execution — EXCEPT for a PRE-AGREED
   BATCH: a change set the human has explicitly ruled on (project-board
   ANSWER lines or direct chat direction), implemented as directed,
   with all applicable local gates green. For such batches the
   orchestrator pushes autonomously immediately after gates pass; the
   human's prior agreement plus gate evidence recorded in the commit
   message satisfy this clause — no second presentation cycle.
   AMENDED 2026-08-26 by direct human ruling ("once i agree to changes,
   and they are made and all tests are green the only thing left to do
   is push — this should be the standard"). Pushes containing anything
   OUTSIDE the agreed scope (unplanned files, unruled contract changes)
   still require full presentation and supersede no rule above.

Retired 2026-08-26 (D37): Change Board locked-issue #5 apparatus, the
senior-arbitration tier (file SENIOR.md, deleted), single-use work orders
(file G0B.md, deleted), and the board recipe appendix (file
MAC_APPENDIX.md, deleted). The surviving ENFORCED
register is `STATE.md ## EVENTS` (read by TIER-GATE and probe g).
Historical rulings remain verbatim in DECISIONS.md and
`agents/ledger/arbitration/**` as dated records.

### Reproducibility mandate

- **Mechanism claims reproduce first.** A ledger entry asserting HOW
  something happened (not just THAT it happened) may not land until
  the mechanism is demonstrated by re-running it, or is explicitly
  labeled `UNREPRODUCED` with resolution resting only on what WAS
  demonstrated. Best-explanation wording is banned for causes.
- **Every load-bearing row carries `check:`** — one command a fresh
  agent can run to re-derive the entry's central fact
  (`check:` + pasted expected tail per D4/D8). A row whose check
  cannot be written is a narrative, not a record; file it as an
  observation instead.
- **Mechanical stages are deterministic.** Registry rendering,
  event-id computation, parity checks, probes: same input ⇒
  byte-identical output, pinned by test where cheap. A mechanical
  stage that cannot be made deterministic is treated as judgment
  work and gets an evidence trail instead.
- **Judgment stages cite their facts.** Senior rulings and slates
  are not reproducible as judgments, but every load-bearing CLAIM
  inside them cites its file:line plus the command that surfaced it,
  so the facts re-derive even though the verdict does not.
- **Dual-plane verification.** This repo contracts TWO test planes:
  agnostic platform law (`tests/` over src/broadway) and dataset
  binding (`project/tests/` over project/). A `check:` line or
  acceptance command covering a surface contracted in BOTH planes
  MUST run BOTH suites — single-plane verification of a two-plane
  surface is an incomplete check and the row does not land. Edits to
  parity-SHARED surfaces additionally require the parity lockstep
  check on the same commit. Slates name the plane(s) each item
  touches; the registry's `validated_by` entries already span both
  trees and are the authoritative map of which plane pins what.
- **Isolation protocol.** On anomaly, walk the chain backwards
  through `check:` lines until one fails — that failing step IS the
  defect locus. Fix or relabel there; never patch downstream of an
  unreproduced step. Fixes land at the locus or not at all.

### Ledger & artifact hygiene

- Every new file under `agents/**` declares its class at creation:
  `SSOT` (single source of truth) · `derived` (regenerated from an
  SSOT) · `cycle-scoped` (belongs to one work cycle, then archives).
- project/scripts/* are intentionally record-free teaching surfaces;
  they bypass timeline/lineage gates by design; promotion to production
  surfaces requires record-writing shims first.
- Derived/rendered artifacts stay under ~50 KB tracked. When they
  outgrow the budget the fix is regeneration or retirement, never
  accumulation. Intermediates whose content survives verbatim inside
  an SSOT are DELETED once the SSOT goes live-green (deletion-first;
  precedent: GATES.md + gates/*.md fragments retire against
  gates.yaml).
- Working ledgers (`STATE.md` `## EVENTS`, `FIXES.md` incident log)
  hold CURRENT-CYCLE rows only. At cycle close, resolved rows
  compress into agents/ledger/archive/YYYY-MM.md (date-substituted
  at creation) behind a pointer line; the working file resets. `DECISIONS.md` is law, not log —
  exempt from rotation, entries appended, never rewritten.
- Contract files cap ~30 KB of prose; beyond that, worked examples
  and recipes move to appendix files beside the contract.
- Session context-load rule: load DIGEST.md + working ledgers +
  relevant packet files. Never load a full SSOT dump into context
  when a digest exists.
- **Cache & teardown law.** Single sanctioned cache root:
  `$HOME/.cache/uv`. Repo-local cache directories (uv/mypy/pytest at
  repo root) are PROHIBITED; teardown scripts must return
  container/image/volume state to zero; new durable object creators
  require a registered lifecycle owner before first use.
  MPLCONFIGDIR may point at repo-root `.mplconfig` for font-cache
  determinism; all other repo-local caches remain PROHIBITED.

### Registry maintenance duty

- The machine-readable SSOT for gates/surfaces is
  agents/ledger/gates.yaml. Any change touching a gated surface MUST
  edit the owed gate rows in the SAME commit as the code (§14 step 4
  UPDATE IN PLACE applies to gates.yaml rows, not GATES.md).
- Same commit runs: python agents/tools/render_gates.py  (regenerate
  indexes + DIGEST) and the registry verification test. A red
  registry test blocks landing exactly like any other gate.
- Before proposing a change, run the blast-radius query for the
  target path; its output IS the list of rows the change owes.
- Staleness is loud by design: deleted owners, renamed symbols,
  vanished pins, and broken id references fail the registry test.
  Silence is a bug in the test, report it like one.
- Recurring accuracy audit owner + cadence and the events-log tamper lock are ratified in
  `DECISIONS.md` D31; every `STATE.md` ## EVENTS edit cites its authorizing event-id in the same commit.

### Row format — surface & data-gate rows

A board row that REGISTERS a pipeline surface or data gate (as opposed
to reporting prose) carries the canonical field set, in this order:

```
- id: <GATE-…-NN>  phase: <canonical-order position>
  owner: <file:line symbol()>     # single writer-of-record
  inputs: […]                     # ARTIFACT-*/CFG-* ids consumed
  outputs: […]                    # ids produced
  transforms: ["…"]               # one line each
  touched_by: [lib@version]       # attribution
  validated_by: [test node ids]   # the pins that hold today
  if_changed: [downstream gates, artifacts, tests]
```

Prose rows (incidents, rulings, verdicts, status flips) keep the
subject + prose form specified above; never mix both formats in one
comment.

**ROOT-CAUSE MANDATE:** every FINDING line, incident row, and
adversarial verdict names the ROOT problem on its own `root:` line —
the deepest cause whose repair would have prevented the CLASS, not
just this instance (the difference between "pre-push hook missing"
and "a guard depended on machine-local untracked state"). A fix or
ruling aimed at the symptom while the root stands is auto-MODIFY at
senior review; an incident row without its `root:` line is not
closed, whatever else it cites. Repeat offenders in FIXES.md
(push-on-red ×2, custody violations ×3) exist precisely because this
was never required.

### Gate registry & digest rendering

- **Registry SSOT:** the gate registry (in-tree under `agents/ledger/`,
  created by its first registering lane) owns gate rows VERBATIM; board
  comments are the evidence layer and STATE.md `## EVENTS` the
  resolution layer (GIT-WINS unchanged).
- **DIGEST.md is rendered output, never hand-edited** — gates in
  canonical order as compact per-gate blocks. This is what an agent
  loads into context instead of the raw registry. Single-writer rule:
  the registering lane renders; everyone else consumes.

### Registry-driven change protocol (the ONLY way surfaces/gates change)

1. **LOCATE** — before touching anything, find your file/symbol in
   DIGEST.md (or render_gates.py query mode). Its row hands you: the single owner,
   inputs/outputs, the validated_by pins that hold today, and every
   if_changed target your diff can move.
2. **CHANGE** — implement against the owner symbol only; forking a
   second writer for an owned surface is a refusal-class violation.
3. **RE-VALIDATE** — run the row's validated_by pins AND each
   if_changed target's checks. These are exactly the gates your change
   can break; nothing else needs re-derivation from scratch.
4. **UPDATE IN PLACE** — the registry row travels in the SAME commit
   (owner line numbers, transforms, new pins), DIGEST.md re-rendered.
   A stale row is a doc-drift bug (HANDOFF rule).
5. **RECORD** — material changes record their trigger + human
   confirmation in the landing commit message itself; no board row exists
   anymore (D37).
