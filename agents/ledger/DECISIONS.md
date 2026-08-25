# DECISIONS.md — Slate v4 decision sheet (D1–D10)

Arbitrated 2026-08-23 after six audit lenses (redundancy, gaps, SSOT,
hardcoded values, contradictions, better-ideas) plus a three-adversary
red-team panel (scope/necessity, sequencing/risk, fact-check) over range
`ea33370..be34c30` and the ratified FIX_4 closure (`79ac26c`).
Fact-check score: 10 CONFIRMED / 0 REFUTED. Sequencing verdict: adopt-revised.

- **D1 · Slate v4 (10 contracts)** — RATIFY: G0a platform-doc deletions;
  G0b governance truth; A1 schema-builder unification (+ riders:
  builders-lambda defaults, `coercion` lineage kind, `.uv-cache` gitignore);
  A2 datetime semantic compare; B1 build-time include-validation
  (`contract ∪ joined-lookup`, raise naming unknowns); B2 parity fail-loud on
  unresolved skips; C1 `_SAMPLE_SCHEMA` derives from config; C2 `etl.run`
  coercion-persistence test + loader docstring fix; C3 derived/encoded
  read-flip coverage (narrowed); D1 rev-parse ref pre-check exit 2.
  Killed by panel: warning-absence pins, SHARED expansion/extraction,
  naming helper, dtype-policy constant, fixture migration, harness dedup,
  congruency-as-doctrine.
- **D2 · Order & isolation** — G0 → A → B → C → D (sequencer-revised):
  A-before-B removes generic.py double-churn; B validated at BUILD time
  (load-time crashes parity collection via the 8-pair cross-product); G0
  lands before any D edit of the checker file.
- **D3 · Authorization mode** — batch: sequential dispatch, each contract
  runs worker → gates → adversarial reviewer → arbitration → commit/push;
  actuals sheet at completion.
- **D4 · Canonical gate invocation** — root `uv run pytest -q` pinned in
  every brief (CI `pytest tests/` quoted secondarily; ±19 difference);
  counts measured at step-0 paste, never projected; suite-total warnings
  NEVER gated (25 occ/7 groups vs ledger "9" — irreproducible); per-path
  pins only.
- **D5 · configs/experiments/mlflow.yaml** — LEAVE DIVERGENT, document it:
  deliberately taxi-purged on main (`4657013`, `a2f26e9`); adding to SHARED
  would let main-day `--sync` clobber main's synthetic variant with ratecode1
  names that do not exist there. Parity only after a data-agnosticize pass.
- **D6 · graph_todo.md** — undocumented working-tree deletion occurred
  2026-08-23; restored by main agent; provenance unclaimed. Stays restored.
- **D7 · HANDOFF final-SHA bookkeeping** — batch SHAs written into HANDOFF
  at slate completion, not before.
- **D8 · Worker evidence format** — codify "paste the command alongside the
  tail line" in WORKER_CONTRACT (lesson: root vs tests/-scope invocation made
  both 783 and 764 honest numbers); rides G0b.
- **D9 · Backlog confirmed parked**: log_dataset in-memory frames when
  lineage wires into training; Option-C typed-source hard rejection;
  composite-key encoding-naming revisit iff a config declares multi-column
  encodings.
- **D10 · Artifact hygiene** — regenerate train_features.parquet via real
  taxi flow after the slate lands (pre-guard artifacts predate A/B changes).

Standing facts: baseline after FIX_4 = 783P/1S/0X root-scope (764P tests/-scope),
1 skip is the PARITY_MAIN_DAY gate; parity gate red-by-design until declared
main-day. Batch lineage: H `3db7b4b` → FIX_1 `c324583` → FIX_2 `3ee1ef5` →
FIX_3 `ca8c123` → FIX_4 `79ac26c` → governance `3ea88d1`/`c34710c`/`be34c30`.

*Authorized for publication by the human operator via GUI session,
2026-08-23 ("push the D1-D10 ... if its easier push on sklearn").*

## D11–D13 — slate v5 rulings (ratified post-ADV-trio)
- **D11 Tier-4 dead-code doctrine**: stub modules consolidate into INVENTORY.md
  (name / one-line intent / why unbuilt) then delete stubs + dead fields in ONE
  commit. Test-only exports and dead EnvironmentConfig fields: straight delete,
  no inventory entry.
- **D12 decision-moment leakage**: ENFORCE gating; drop dropoff_location_id from
  the shipped taxi experiment surface. A contract layer silently overridden by
  config teaches the wrong lesson; amend-the-contract is rejected absent a real
  use-case change.
- **D13 tier order**: Tier1 confirmed bugs → Tier2 config coherence (absorbs
  B1/B2) → Tier3 data-gate hardening → Tier4 (post-D11) → Tier5 bulk.
- **D14 T-bug-1 scope expansion (user spot-check)**: estimation_table() blindly
  trusts the handed model object — HC3_SE *and* CI_low/CI_high mislabel
  nonrobust fits whenever callers don't pre-fit cov_type="HC3". Fix derives HC3
  independently of input fit; landmine test added (plain-fit input ⇒ HC3 ≠ OLS).
- **D15 proof-carrying wrapper layer REJECTED**: scoping agent mapped all nine
  affected findings (T-bug-1, F1, F8/F11/F14/F16, T-bug-4, F4/F19) to
  equal-safety direct fixes inside the existing idiom — pydantic parse
  validation, pandera schemas derived from DatasetContract/LookupSpec,
  config-threaded thresholds with required params (runners.py precedent).
  mypy IS CI-enforced but only sees type structure, not construction-site
  invariants, so wrappers would add a drift-prone second declaration surface
  (contra D5/D6) and nothing else. Tier 3 fixes are written against the
  existing machinery; new sub-finding adopted: stats/module.py silently skips
  empty groups — min_rows_for_sampling exists in config but is unenforced at
  the stats entry point.

## D16 — era-aware branch parity (senior ruling: design C8, adversary-amended)

Adversarial amendments ratified into D16b: (F1a) era file added to SHARED so
main-day --sync delivers it; (F1b-residual) pushes TO main execute main's
legacy checker until main-day — a frozen-line push is itself the violation,
so legacy red is correct there; (F2-revised) custody anchors on
PARITY_MAIN_ANCHOR (seeded at the frozen tip, updated ONLY in the same commit
as any ratified main-day sync/flip-back) + blob-provenance; merge-base
anchoring rejected — 21/24 SHARED entries legitimately diverge between
merge-base 7758d1a and the sanctioned main tip; (F3) pytest gates on the
committed era file, never os.environ; (F4) GITHUB_REF_NAME-first branch
detection with HEAD/empty handling; (F5) guarded allowlist expansion +
case-whitelist era validation.

- **D16a single era vocabulary** — `.github/parity-era.env` is the only era
  declaration (`PARITY_ERA=dev|main`); the `PARITY_MAIN_DAY` os.environ
  dialect is deleted everywhere.
- **D16b dev-era semantics** — sklearn ⇒ taxi may lag, never fork;
  taxi ⇒ fast-forward byte-equality required; every event runs custody
  (anchor drift guard + blob provenance) incl. ROGUE MAIN WRITE;
  verified 0-novel/261 blobs ~95ms.
- **D16c main-day playbook** — human ratifies; main agent flips
  `PARITY_ERA=dev` → `main` in ONE commit citing this ruling; stock loop
  resumes verbatim; revert = `git revert` of the flip. The SAME commit also moves `PARITY_MAIN_ANCHOR` to the post-sync main tip. Run `--sync` from a checkout holding LOCAL `main` and `taxi` branches — the stock `sync_to_main` checks out local refs, not remote-tracking ones.
- **D16d rejected candidates** — C7 (dual vocabularies: two sources of era
  truth drift apart); C6 (os.environ gating: CI sets no env vars, hole
  re-opens); C2 (per-worker flags: the invariant is suspended, not false —
  a per-worker suppression flag deletes the red and buys silence);
  C1 (do-nothing: same trade — the known hole stays open, and removing or
  ignoring the failing gate buys silence rather than conceding the
  invariant is suspended, not false); C4 (timestamp eras: kills the
  legitimate lag window — every sklearn push would sit in a forbidden/red
  interval until a clock crosses the declared instant, garble included);
  C5 (branch-name-as-era: spoofable, no main-day act);
  C3 (third long-lived branch: surface bloat, more parity pairs);
  C0 (hardcoded main-day date: unfixable without an edit + redeploy).

## D17 — Gate single-source-of-truth; coverage floor raised to 95% (adversary-panel ruling)
- **D17a root cause**: the platform gate list lived only as inline ci.yml YAML
  while contracts assembled verification ad hoc from a one-invariant template.
  Of nine locally-runnable CI checks, exactly one (parity) was shared fully;
  pytest ran at wrong scope with no coverage floor. Thirteen contracts of
  ruff-luck masked the gap until eee33d7's orphaned numpy import went red on
  remote CI — luck had been mistaken for protection.
- **D17b SSOT script**: `scripts/run_local_ci.sh` (parity, ruff, mypy,
  config-parse, pytest+cov≥floor) is the ONLY owner of the shared platform
  gate list; ci.yml invokes it and retains docker-only checks BY NAME.
  Editing YAML alone reopens two-source drift (D16d C7); do-nothing was
  D16d C1. Push authorization: local tiers green AND last CI-on-tip green.
- **D17c floor**: coverage floor raised 85→95 once the COV-95 campaign lands
  tests for reachable gaps; unreachable-by-design lines require documented
  pragma justification or deletion, never silent exclusion.
- **D17d standing adversaries**: every review mandate now includes
  static-hygiene, gate-divergence-watch, and coverage-gaming vectors
  (codified in MAIN_AGENT_CONTRACT.md §6). Facade modules are never
  autofixed; experiment batches outside CI's lint whitelist carry a recorded
  44-finding debt batched for Tier-4/5 cleanup.

## D18 — Arbitration verdict-class table (scaling the serial stage)
Standing rulings reviewers cite directly; novel cases escalate by default.
- Vacuous/tautology test ⇒ reject-and-fix (test author rewrites with
  falsifiable assertion).
- Missing `match=` pin on a new raise site ⇒ rider (batch at Tier-5).
- Doc-drift vs shipped behavior ⇒ commit-as-is-with-rider (fix folds into
  next governance touch).
- New branch-level dead arm ⇒ pin-or-pragma with justification comment.
- Coverage-gaming (line-execution without meaningful assert) ⇒ reject.
- Custody deviation of ANY kind ⇒ STOP, log incident, bar pending review.

## D19 — Custody amendment + decision-pipeline adoptions (human-ratified)
EVENT: issues/4#issuecomment-5398092508 event-id 3afcd9b1
Full pipeline ran: 3 investigators + 3 adversaries + senior synthesis
(b983b5a5). Rulings: P1 risk-tiered review ADOPTED as amended (TIERREV-1);
P2 custody isolation ADOPTED as three-tier architecture (CUSTODY-1) — naive
uniform worktrees REJECTED after demonstrated shared-store destruction
(gc --prune=now from a "read-only" worktree dropped objects 7853→5); P3
xdist ADOPTED as amended with blocking uv-0.5.x lock-parse precondition
(XDIST-1, real speedup ~2×).
**Authority grant (this ruling):** the custody line "workers run no git
operations at all" is AMENDED to permit harness-WRAPPED git writes inside
ephemeral detached worktrees for the trusted-executor tier ONLY; main agent
retains sole commit-to-shared-refs/push custody. Mitigations binding this
grant: wrapper operation ALLOWLIST (gc/reflog-expire/update-ref/tag-delete/
foreign-prune structurally unreachable), 2-week private-clone quarantine
for executors before any wrapped worktree is issued, mechanical integrity
probe at every arbitration (git count-objects + reflog depth +
PARITY_MAIN_ANCHOR blob census vs pinned baselines, alarm-on-drift).

## D20 — Micro-contract decomposition (human-directed rethink)
EVENT: issues/4#issuecomment-5398092820 event-id 7595cb13
Motivated live: a 4-deliverable worker ran >15 min on 1/N visible progress.
Long single-worker contexts are hallucination + context-exhaustion risk;
fat contracts also hide the parallelism the pipeline could exploit.

### Size classes (hard caps; LARGE exists only as a plan, never a dispatch)
- **MICRO**: ≤2 files OR ≤30 changed lines, ONE deliverable, ≤10 min expected,
  no investigation needed beyond main-agent scoping.
- **MEDIUM**: ≤4 files AND ≤150 changed lines AND ≤3 deliverables, ≤25 min.
- **LARGE**: anything larger ⇒ MUST decompose into MEDIUMs before dispatch.

### Two-phase mandate (INVESTIGATE ≠ EXECUTE)
- **INVESTIGATE contracts** are read-only: deliver an exact FACT SHEET
  (current behavior, line numbers, edge inventory, proposed-diff sketch).
  No production changes ever. ≤10 min each.
- **EXECUTE contracts** consume the fact sheet VERBATIM — pasted INTO the
  dispatch prompt, never re-derived from memory. They touch only files the
  sheet names and implement exactly the sketched diff. Reality contradicting
  the sheet ⇒ stale-on-arrival STOP (existing gate).
- MICRO contracts may fold both phases into main-agent scoping when the
  surface is already known.
- Anti-fabrication binding: an EXECUTE worker may not assert any fact absent
  from its sheet; evidence pastes mandatory per deliverable.

### Sequential vs parallel delegation
- PARALLEL-ELIGIBLE only when ALL hold: (a) zero file-custody intersection,
  (b) no producer/consumer dependency between outputs, (c) combined tree
  state remains fully describable in STATE.md's lanes table.
- SEQUENTIAL whenever: shared files, data flow between outputs, gate or
  lock dependencies, or ordering is load-bearing.
- DEFAULT WHEN UNSURE: sequential (single-writer discipline).

### Context bounding
- Dispatch prompts are self-contained by construction (no conversation
  inheritance); investigation results cross to executors as pasted text,
  not memory.
- Heartbeat rule (>20 min dispatches) applies FROM LAUNCH, not after the
  first silence; two missed/stale beats ⇒ interrupt-and-reconcile.
- Worker context is bounded by scope caps above; a worker approaching its
  cap reports partial delivery rather than improvising onward.

## D21 — Parity era relocation + F1b pin-guard (human-found gaps, closed)
EVENT: issues/4#issuecomment-5398093134 event-id 555b6fb8
Human review caught two deviations in D16's landing: (a) D1's finalized
"relocate under scripts/, zero array lines forever" was NOT followed —
parity-era.env shipped as separate file + SHARED entry instead; (b) F1b
was never landed — run_local_ci.sh executed tree-local checker, so CI on
frozen main would silently run the pre-D16 legacy checker.
RULING: era/track/allowlist/anchor now INLINE in scripts/check_branch_parity.sh
(PARITY_ERA=dev · TRACK=sklearn · ANCHOR=18607091…c4b); .github/parity-era.env
DELETED, SHARED entry removed. F1b closed: gate_parity() pins execution to
refs/remotes/origin/sklearn and refuses any checker lacking ^PARITY_ERA=
(legacy marker test). Playbook amendment to D16c: the main-day "flip" act
is now editing the inline constants + ANCHOR in ONE commit on sklearn.
Caveat on record: shallow-checkout PR runs may lack refs/remotes/origin/
sklearn ⇒ gate fails loud ("unavailable") rather than silently passing;
pushes-to-sklearn (our authorization path) always have the ref.

## D22 — gh-comms-as-event-store rejected; doctrine set + SECRET-1 rider
MOTIVATION: phantom-relay incident (agent claimed "message e5e41a7a
accepted by senior"; registry forensics showed no such counterpart) plus
human question "can gh comms be the sole traceability storage?".
ADVERSARY af2289bb inverted the proposal: repo is PUBLIC; ledgers carry
gate mechanics + bar reasons; unlocked public issues are stranger-
writable ⇒ mandatory gh-URL citations would TRAIN agents to trust
planted comments (worst-case: audit trail launders fabrication).
RULING (senior d4a86cf9, adopt-as-trimmed):
- In-repo git-anchored authority AFFIRMED: commit hashes are the tamper
  chain at single-writer scale; events derive from git + trailers +
  this ledger per STATE.md GIT-WINS.
- Three-class publication rule for NEW prose rows only:
  PUBLIC→contract IDs/verdict classes/SHAs/suite tails/outcomes;
  LOCAL→agent IDs/bar reasons/custody forensics/gate internals/
  dialogue/pending rows; REDACT-ALWAYS→hostnames, credential-shaped
  strings, detection-window phrasing. Enforced as prose discipline now,
  gate later only on first RED near-miss.
- DOCTRINE: no gate ever requires github.com online.
- CITATION VALIDITY: coordinator-dialogue claims must resolve to
  main-agent registry records or in-repo artifacts; unresolved = void
  (fabrication-class). gh-URL form deferred until locked issues AND
  sha-anchored citations both exist.
- TRIGGER-GATED DEFERRALS: events.ndjson ← second writer or >50
  events/day; gh mirror ← first non-cloning consumer; classification
  gate ← first near-miss; citation tooling ← first gh citation attempt.
- MACHINERY (sha256-chains, ULIDs, pending buffers) REJECTED:
  disproportionate vs git history at current scale.
PROCESS HAZARD ON RECORD: first landing attempt died mid-write
(python substring miss) yet committed+pushed under a Governance-D22
message carrying only the routing-table amendment (4133e45) — a live
instance of mislabeled provenance, caught by re-reading the tool output
before declaring victory. This entry is the corrective second act.
| D22-rider | secret.yaml ships real high-entropy DB password PUSHED PUBLIC ("local-kind" = comment, not enforcement). SECRET-1 rotate-and-remove IMMEDIATELY post-CIADPT-d arbitration, before Axis-2: template manifest + local-override/generated fallback in lifecycle.sh (~2 files, rides shellcheck/kubeconform gates); NO history rewrite (rotation kills value; force-push mid-ladder risks more than a dead local credential). | senior d4a86cf9 |
  AMENDMENT (USER-MVP pilot, human-ratified): TRIGGER "citation tooling ← first gh citation attempt" FIRED early by sovereign approval of the ledger pilot; gh-URL citations UN-DEFERRED solely for events on locked fbarulli/broadway ledger issues under merged conditions (conversation lock + resolution-time actor check + sha8-of-content ids); all other deferrals unchanged.

## D23 — AXIS-2 v2: static-gate split rule (evidence-based replacement)
SUPERSEDES the D22-adjacent Axis-2 provisional rule (p50≥10min / ≥2-cancellations).
EVIDENCE (adversary ADV-B e3f1c4b0, 263-run audit): platform p50 194s, max-ever 336s over 138 green
jobs — old threshold unreachable short of ~×5 suite growth; split benefit capped by the
static slice (~24s @p50) and insensitive to test duration ⇒ platform-duration was the wrong
trigger variable; cancellation clause saturated (60/60 superseded-within-15min ≈4.6/day,
zero signal); warm-bias proven (cold b&b sample size ZERO post-CIADPT-c; cache pool at
10.42 GB vs 10 GB ceiling).
RULE (verbatim, ADV-B text + senior clauses): Measure weekly from Actions API over rolling
20-green sklearn runs. Cancelled runs excluded from all timing statistics, logged separately
(ref, superseding sha, lag) — never a trigger. Track M1=static-gates s, M2=pytest+cov s,
M3=duplication-overhead s (median runner+checkout+sync prefix), M4=b&b s on cache-miss runs.
Split static gates into their own job only when, for two consecutive weekly reviews, BOTH:
(i) M1 > M3+60; (ii) platform p90>360s AND job queue p50<60s. Otherwise keep monolith.
Re-evaluate immediately on XDIST-1b landing or ≥30s static-gate growth.
SENIOR CLAUSES: recorder = senior agent, rows into STATE.md telemetry; cold-sample floor n≥3
before cold figures may drive split decisions; COLDPROBE-1 = one-byte src push on sklearn tip
after XDIST-1b acceptance + ≥5 post-landing greens (~5 runner-min), recorded as {push sha8,
cold_bb_s, cache_hit=false, queue_s} in STATE.md + this entry's addendum — mandatory.
CACHE HYGIENE: no manual deletes during flight; LRU purge is the working mechanism;
safe-delete class only entries unreferenced across the window; act on ceiling only after ≥3
consecutive degraded reviews. | Ruling: senior 3813c37c | Evidence: adversary ADV-B e3f1c4b0 |

## D24 — TAXI-SYNC B: single reconciliation pass to taxi after FIX-WAVE-1 (human-ratified)
HUMAN QUESTION: why did the sweeps land only on sklearn? RECORD: sklearn is the sole active line
(MAIN_AGENT_CONTRACT §2 branch model; README pointer; D16/F1b parity gate anchors origin/sklearn;
main=demo contents, taxi=legacy data-smoke line). Human ratified propagation afterward.
RULE: exactly ONE reconciliation pass carries this session's landed commits (SWEEP-MICRO 619f069,
SWEEP-KNN 953c727, SWEEP-SAMPLER d53f7b9, plus DOCS-TRUTH and FIX-WAVE-1 as they stand when it
runs) onto branch taxi via cherry-pick. No incremental per-batch syncs.
CONSTRAINTS: execute only when zero worker contracts are in flight (worktree quiescent); taxi-side
gate = full `scripts/run_local_ci.sh` GREEN before push, shipped with ship.sh semantics using
refspec `taxi:taxi`; sklearn parity expectations unchanged. If conflicts exceed mechanical
resolution, STOP and re-present options instead of force-merging.
Rationale: one gate cost instead of two; D-register rulings may still reshape taxi-relevant
surfaces. | Ruling: human, this session | Executor: main agent only |
ADDENDUM (ADV-3 findings adopted by main agent): origin/taxi verified STRICT ANCESTOR of the
sklearn line with ZERO unique commits (rev-list counts sklearn-only>0, taxi-only=0) — the
cherry-pick machinery above is superseded: the single reconciliation pass IS one non-force
`git push origin sklearn:taxi` (fast-forward). Procedure: execute only at worktree quiescence;
pre-step asserts and logs `HEAD == sklearn tip`; acknowledge CD side effect (:taxi image
build/publish fires on the ref update). ship.sh tree-vs-refspec gap (F-2) covered procedurally
by that assert — ship.sh itself untouched (gate-list law). Event-id: EVT-D24-TAXISYNC-B
(ledger-canonical; gh-comms rejected per D22).
PRECISION (ADV-3 N-7/N-8): the pass carries EXACTLY the commit list {619f069, 953c727,
d53f7b9, 5016e93} plus any commit bearing tag FIX-WAVE-1 or NOTES-CONSOLIDATE recorded in
this log before execution begins — no open-ended "whatever stands" reading. Quiescent :=
`git status --porcelain` empty except agents/ledger/** and the foreign untracked
experiments/more_modeling/{16..22}* batch. Mechanical resolution := conflict hunks whose
changed lines were all introduced by the listed commits themselves; anything else STOPs
and re-presents.

## D25 — PUSH-STANDING-GO: ship every green batch immediately, no per-push human go (human-ratified)
Human directive: "don't wait for me, push already, always." Amends the per-push-human-go clause of
the push-custody law for THIS working style effective immediately and for the rest of the session.
RULE: main agent ships via `scripts/ship.sh` the moment a batch is committed and LOCAL-CI GREEN —
no confirmation round-trip. The gate remains absolute: RED means refused, never overridden;
quiescence rule (D24) still applies before cross-line reconciliation; foreign/untracked surfaces
still never ship. | Ruling: human, this session |
ADDENDUM (ADV-3 findings adopted by main agent): D25 SUPERSEDES the second push-custody conjunct
("last CI run on the branch tip green") for this session — ci.yml cancel-in-progress makes it
structurally unreachable under push-always. Compensating controls, mandatory on every push:
(i) ship.sh LOCAL full-tier GREEN immediately pre-push (stricter than any stale remote run);
(ii) the post-push Actions run on the new tip is monitored; a remote RED triggers same-session
fix-forward or revert, ruled and recorded in FIXES.md — a red remote tip is never left
unaddressed across a session boundary. Event-id: EVT-D25-PUSHGO (ledger-canonical; gh-comms
rejected per D22).
PRECISION (ADV-3 N-6/N-9): the gate verdict certifies the pushed tip only when, at push time,
the worktree's dirty set contains no tracked code file outside the just-committed batch(es);
if sibling WIP landed in between, gates re-run after the tree settles before the push fires.
PUSH-STANDING-GO sunsets automatically at session end or on explicit human revocation,
whichever comes first.

## D26 — ARBITER-DELEGATION: senior stage rules the open decision register (human-ratified)
Human directive ("lets see them then, lets go … send all these decisions to the senior") delegates
ratification of the accumulated open-decision register — FX-A D1–D7, FX-B DR-1…DR-14, CODELAW
automation candidates A1–A9, adversary BM-/N-findings routing, test-hardening remainder W4–W6,
project/tests blind-spot scope (old D3), doc-truth riders N-1/N-2/N-12 — to the SENIOR arbitration
stage operating under agents/contracts/SENIOR.md + the D18 verdict-class table.
EFFECT: senior verdicts (ADOPT/MODIFY/REJECT, rationale mandatory) execute immediately as
standard implementation contracts through zero-write workers; run_local_ci.sh gate-list edits
become executable upon senior adoption rather than separate per-item human rounds; five-banner
gates remain absolute before any commit; D25 push-always applies to every green result. The human
retains session-level veto and receives the full verdict ledger as briefing. Items whose evidence
packets are still undelivered (FX-A/FX-B second halves) are ruled provisionally from compliance
addenda + banked summaries and flagged PROVISIONAL pending those packets. Event-id:
EVT-D26-SENIORDELEGATE. | Ruling: human, this session |

## D27 — PIN-LINEAGE ARCHIVAL: tier-1-complete / 8e7d51a are archived history
Tag tier-1-complete (rooted at 6be4ff7, incl. commit 8e7d51a) shares NO history with sklearn
(root 0561e93): its stamps are ARCHIVED HISTORY, never current branch state. The tag is
PUBLISHED on origin (H-infra DRIFT #4 correction; custody defers to GATE-INFRA-141 creation-time
law) and NO retirement is performed. Evidence citation remains valid via `git show 8e7d51a:<path>`
while the tag exists. No credential rationale attaches (HC-1 VOID per
arbitration/2026-08-24/00-resolutions.md). Cross-references: the STATE.md standing-hazard bullet
("Orphaned pre-rewrite lineage") and the SUPERSEDED header now on agents/contracts/G0B.md.
| Ruling: human, this session |

## D28 — SUITE RE-BASELINE 2026-08-25 + COUNT EXEMPTION RULES
A real measured run REPLACES all prior courtesy tails: command `UV_CACHE_DIR=$PWD/.uv-cache MPLCONFIGDIR=$PWD/.mplconfig uv run pytest -q` at HEAD c4f3018 → `1059 passed, 1 skipped,
11 warnings in 118.70s`, exit 0 (root scope).
EXEMPTION RULES (the human's three, verbatim intent): (1) failures recorded before 2026-08-25 do
not count against the current baseline; (2) warnings do not count — reaffirms D4: suite-total
warning counts are never gated, per-path pins only; (3) trailing text does not count — tail
matching pins exact pass/skip/fail counts, never surrounding prose.
Superseded figures: 846P/1S root scope, 827P/1S gated scope, stale TODO "516 passed"; STATE.md
suite-tail lines were refreshed in the same landing cycle. | Ruling: human, this session |

## D29 — CACHE-CEILING FALSE ALARM CLOSED (10.42 GB)
The 10.42 GB reading in D23 EVIDENCE (this file ~:274-275, "cache pool at 10.42 GB vs 10 GB
ceiling") was GitHub's SHARED Actions cache pool, NOT the local cache; the local uv cache is
confirmed nowhere near full. Alarm closed as false — do-not-re-raise.
DISTINCTION from the ≈10.39 GiB packet-H GitHub-side census encoded in GATE-INFRA-143
(gates.yaml ~:3770ff): that row records CI-side cache LAW and stays untouched by this ruling.
| Ruling: human, this session |

## D30 — F4′ RATIFIED (paperwork record of already-landed work)
F4′ = third event-id namespace ruling (senior authority 3813c37c): USER-MVP pilot event-ids are
valid iff a UNIQUE row exists in STATE.md ## EVENTS; role vocabulary grants no escape inside the
full-line EVENT grammar. Work ALREADY LANDED via linear single-parent commits 75cfe21 → 35c619f →
21e17cd → ed250be → b7086b0 → 38f649a (2026-08-24T16:57Z–17:18Z; both endpoints merge-base
--is-ancestor of sklearn, exit 0). This row is the ratification RECORD only — no code change.
Measured registry size at ratification: 40 data rows in ## EVENTS (a cited "41 rows" counted the
table's pipe-header row). Enforcement probes live in tests/test_governance_probes.py
(:15-18, :44-47, :250, :421). | Ruling: human, this session |

## D31 — REGISTRY-AUDIT DUTY + EVENTS-LOG TAMPER LOCK (three-part ruling)
(1) NAMED OWNER: a fresh read-only REGISTRY-AUDIT agent dispatched by the main agent on a fixed
cadence. TRIGGERS: every governance-file-touching landing, any registry-accuracy dispute, plus
one standing weekly sweep. METHOD: agents/tools/render_gates.py query/blast-radius modes,
cross-checking gates.yaml rows against the live tree; zero writes, drift reported to the main
agent. (2) TAMPER LOCK (lands with this governance-file touch per the ruling): any commit that
edits STATE.md ## EVENTS must reference its authorizing event-id within the same commit (message
body or adjacent same-commit row). (3) Board-mirroring proposal REJECTED — do not build.
| Ruling: human, this session |
