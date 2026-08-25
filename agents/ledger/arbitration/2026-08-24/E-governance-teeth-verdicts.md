# PACKET E — GOVERNANCE-TEETH · SENIOR REPORT (verbatim, 2026-08-24)

## Step-0

- Hash gate: `git rev-parse HEAD` → `5016e937e6aa67b301e3b54b78b1891f85784c67` = stamp `5016e93`. MATCH.
- Context gate: STATE.md read. Contradiction echo: none STOP-class. Tree carries 13 modified tracked files (R1–R5 pending batch) while "active lanes: none" — consistent with board-mirrored pending inventory, not a contradiction. Suite tails "re-derive if needed" → re-derived below (973P/1S). Note: `gates/`, `factsheets/`, `arbitration/` are **untracked** — citations from them are working-tree-only, no HEAD version exists to diff against.
- Custody: zero writes; one scratch cache dir created then deleted (`.cache-seniorE`, removed; `git status --porcelain | wc -l` → 40, unchanged); no git mutations; two read-only gh api calls.

## RULINGS (packet format, 29 rows / 28 blocks — #15≡GAP-3 merged)

**1. INFRA-90a** — no test executes run_local_ci.sh end-to-end
VERDICT: MODIFY(to: extraction-execution harness running the REAL script body with PATH-shimmed `uv`/`git` stubs, asserting tier parse :11-16, suppression guard :52, run() aggregation :20-25, banner vocab :56-60)
root: the D17 SSOT gate script's own behavior is guarded only by text-shape reads — a gate nobody tests is prose with a shebang.
rationale: verified live (`grep -rn "run_local_ci\|custody" tests/` → only `_gate_parity_source` body-extraction at test_branch_parity_scripts.py:123 + wiring assert :133; case-dispatch/run()/banner zero). Alternative: accept text-read tests — loses: any aggregation/suppression regression passes green. Precedent for the harness pattern exists in-repo (:166 executes checker under bash with stub `git`).
now-fix: DEFER → board-row **CI-SELFTEST** (~55-line new test file; exceeds this window alongside higher-teeth items).

**2. INFRA-90b** — `--static` skips pytest on nominal full tier, prose-only discipline
VERDICT: MODIFY(to: static run prints distinct banner + explicit `pytest+cov SKIPPED` line)
root: suppression state invisible in output vocabulary — an operator cannot distinguish static-green from full-green, so "static isn't for landing" is unverifiable at the moment of decision.
rationale: **assumption-audit finding** — GATES.md:1128 claims banner vocabularies "deliberately distinct per tier"; FALSE: run_local_ci.sh:57 distinguishes only fast vs everything-else, and `--static` leaves `TIER="full"` (:12), so a static run prints **LOCAL-CI GREEN**, byte-identical to full. Mitigation narrower than briefed though: hook (`.git/hooks/pre-push:5`) and ship.sh:17 both invoke bare full tier unconditionally — the mechanical landing path already ignores --static. Alternative: forbid --static by contract edit — that's prose again, the very disease.
now-fix: DEFER → folded into **CI-SELFTEST** row (banner assertions).

**3. INFRA-91** — F1b true network path never exercised in CI
VERDICT: REJECT(with: do-nothing; hermeticity is the design)
root: validation strategy deliberately pinned offline — making CI exercise `git show refs/remotes/origin/sklearn` against live upstream would violate D22 doctrine ("no gate ever requires github.com online", STATE.md hazards).
rationale: the shimmed negative tests own the decision logic (marker rejection, legacy refusal); what's untested is GitHub's byte-delivery, which is not our surface. Alternative: nightly networked job — loses: reintroduces the exact external-dependency flakiness D22 bans, to prove someone else's transport layer.

**4. INFRA-92a** — project/ lint/type-checks in no tier
VERDICT: MODIFY(to: fix-first then widen — clear the 12 findings, then add `project/**/*.py` to the ruff roster; mypy widening stays separate until annotation debt measured)
root: gate roster was written against src/-era layout and never followed the code.
rationale: probed live: `uv run ruff check project/ --no-cache` → **Found 12 errors** across project/scripts/{04..11}_*.py, project/etl/process.py, project/tests/test_ingest_contract.py (10 auto-fixable). Adopting roster-widening today ships an instant RED — violates green-at-HEAD. Alternative: whitelist-suppress the 12 and widen now — loses: manufactures a second lint vocabulary exactly like the experiments backlog debt (STATE.md backlog row).
now-fix: DEFER → board-row **PROJECT-LINT** (fix-forward contract, then one-line roster edit).

**5. INFRA-92b ≡ R11 project/tests scope** — project/tests executes in no tier
VERDICT: ADOPT — wire `project/tests` into the full-tier pytest line now.
root: pytest invocation enumerated one directory by habit, not by ownership rule ("every tracked test directory runs").
rationale: probed live: standalone `uv run pytest project/tests -q -p no:cacheprovider` → **26 passed, 4 warnings in 2.12s**; combined `uv run pytest tests/ project/tests -q -p no:cacheprovider` → **973 passed, 1 skipped in 101.96s** (background job bash-1, exit 0). Assumption-audit: brief blamed the 2 *untracked* ingest files — understated: `git ls-files project/tests/ | wc -l` → **4 tracked** test files also sit outside every gate today. Coverage floor untouched (`--cov=src/broadway` unaffected). Cost: ~+2s tier time.
now-fix: files=[scripts/run_local_ci.sh] · changed-lines≤1 · acceptance=`uv run pytest tests/ project/tests -n 4 --dist worksteal -q -p no:cacheprovider | tail -2` (expect ≥973 passed, 1 skipped). **SLATE A.**

**6. INFRA-92c** — nothing proves a sub-95 cov run fails
VERDICT: REJECT(with: do-nothing — the flag-deletion tripwire already exists; enforcement semantics are upstream pytest-cov's)
root: (none needed to build — the gap as stated misattributes ownership.)
rationale: probe_coverage_floor (test_governance_probes.py:287-302) raises `"gate script declares no --cov-fail-under floor"` when the flag vanishes (:290-291) — the locally-ownable drift is already pinned. What remains is pytest-cov's own fail-under machinery, third-party-tested. A behavioral sub-95 test would re-prove a library. Alternative considered seriously: end-to-end red-run fixture — loses: slow subprocess per suite run to test upstream code we don't maintain.

**7. INFRA-93a** — stock check() comparison era-gated out during dev
VERDICT: REJECT(with: do-nothing; the skip IS the declared semantics)
root: cross-era comparison is undefined by design until flip day (D16/D21) — sklearn-vs-main SHARED divergence is legitimate all dev-era long.
rationale: test_branch_parity_scripts.py:87-101 documents exactly this ("diverges by design until the human-declared main-day flip"). Forcing the comparison dev-era yields a permanently-red test, which trains everyone to ignore red — worse than silence. Alternative: compare against a recorded drift baseline — loses: second source of truth for expected drift, zero signal until main-day anyway.

**8. INFRA-93b** — sync_to_main() zero coverage
VERDICT: ADOPT(deferred) — destructive checkout/rm/deletion-mirror machinery must be rehearsal-tested before anyone runs it for real.
root: one-shot destructive code with no dry-run harness converts main-day into first-live-test day.
rationale: it runs on MAIN DAY under human playbook (D16c); a bug there mutates or deletes shared-surface files on main. Timing makes deferral honest rather than lukewarm: value concentrates pre-flip. Alternative: write it now anyway — loses: window budget better spent on custody teeth (see slate), and rehearsal harness needs tmp-clone fixtures worth doing properly.
now-fix: DEFER → board-row **MAIN-DAY-DRYRUN** with deadline condition: MUST land before `PARITY_ERA=dev→main` flip commit.

**9. INFRA-94** — anchor guards + pass-along guards asserted nowhere
VERDICT: ADOPT(deferred) — negative-path tests through the real dispatch for garbled anchor (:119-126), non-ancestor taxi push (:210), drifted taxi tip (:216).
root: only the happy path of a ref-state machine was ever fed through its own gate.
rationale: these guards were born from incidents (fork/drift taxonomy, D16b); unexercised guards rot silently exactly like F1b did pre-D21. Same extraction-execution pattern as the existing F1b negatives fits. Oversized for this window next to custody.
now-fix: DEFER → board-row **PARITY-GUARD-TESTS**.

**10. INFRA-95a** — custody() ZERO coverage, both layers shortcut-bypassed today
VERDICT: ADOPT — falsifiability tests now: novel-blob layer fires when main moves off anchor carrying foreign content; alarm path proven once in this repo's life.
root: the rogue-write alarm has never fired nor been forced to fire — an alarm unproven is decoration (GATES.md:1191 "correct-by-construction but unproven"; validated_by: []).
rationale: the novel-blob false-positive incident (FIXES.md:208-220) was closed by adding the freeze-intact shortcut — i.e., the last change to custody() shipped with zero ability to regress-test itself. Extraction-execution against a tmp disjoint-history clone matches existing suite patterns. Alternative: wait for main-day — loses: shortcut logic interacts with layer arming TODAY (any anchor edit re-arms it); cheapest moment to pin is while topology is trivially known.
now-fix: files=[tests/test_branch_parity_scripts.py] · changed-lines≤32 · acceptance=`uv run pytest tests/test_branch_parity_scripts.py -q`. **SLATE C.**

**11. INFRA-95b** — stale-pin failure direction unpinned
VERDICT: ADOPT — same fixture as #10: advance main past a stale anchor, assert layer-1 fires (documented safe direction: false-red, never false-green).
root: "fails loud not silent" was argued in prose, never demonstrated.
rationale: folds into the #10 harness as a second assertion — near-zero marginal lines. Alternative: separate test file — loses: shares fixture, pointless duplication.
now-fix: included in **SLATE C** arithmetic.

**12. INFRA-96a** — ship.sh zero automated validation
VERDICT: ADOPT(deferred) — execute refusal law via shimmed `bash` (RED→refuse exit 1, single-push invariant on GREEN).
root: ship law written in response to two push-on-red incidents is itself enforced only by reading it.
rationale: born-of-incidence code without a pin regresses free (FIXES.md:150-161 ×2). Modest harness; lost the window budget contest honestly.
now-fix: DEFER → board-row **SHIP-LAW-TEST**.

**13. INFRA-96b** — pre-push hook UNTRACKED machine-local; fresh clones have NO local gate
VERDICT: MODIFY(to: tracked `.githooks/pre-push` as writer-of-record + idempotent `scripts/setup_hooks.sh` regenerating `.git/hooks/pre-push` from it + equality probe; single-writer preserved — .git/hooks becomes derived artifact)
root: the strongest local gate lives in untracked machine-local state — the exact ROOT-CAUSE MANDATE example (§14:415-417) still standing.
rationale: verified: `git config core.hooksPath` → unset; hook content runs bare full tier. Naive "add a probe asserting hook exists" REJECTED by me — false-red on every fresh clone forever. Alternative: rely on remote CI as backstop — loses: GATE-SSOT incident proved remote catches what local missed, not vice versa; local gate is the push-moment tripwire.
now-fix: DEFER → board-row **HOOK-TRAVEL** (3 small files but setup-script correctness + writer/derived relationship deserves its own review; not squeezed into a full window).

**14. INFRA-98 ≡ GAP-3 (#20/#21)** — nothing forces commits to CARRY a computed Tier: trailer
VERDICT: ADOPT(deferred) — staging/pre-push rejection of trailer-less commits; FULL tier requires resolvable verdict id.
root: tiering was built as a classifier, never wired as a gate — classification without invocation is advisory.
rationale: demonstrated exploit: `7c03347` shipped `Reviewer: none` under pressure (FIXES.md:138-148). Verified: only Tier: mention in scripts/.github/tests is a docstring (tier_classifier.py:4) — zero enforcement. Packet screening agreed: half-gate worse than none; trailer-validation grammar (what resolves as verdict id) must be decided first.
now-fix: DEFER → board-row **TIER-GATE** naming the grammar decision as its precondition.

**15. INFRA-99a** — kubeconform hardcoded k8s/optuna/
VERDICT: REJECT(with: do-nothing until a second sibling directory actually exists under k8s/)
root: speculative generalization — there are currently zero siblings to miss.
rationale: discovery-loop rewrite trades a hypothetical for real false-red surface (scratch manifests in a future dir become CI failures). The hazard is already named where hazards belong: GATE-INFRA-99 if_changed + DEPLOY-DIFF lane. Alternative: textual probe pinning the scan scope string — loses: pins today's limitation as law instead of surfacing it at expansion time.

**16. INFRA-99b** — cancel-in-progress × D25 push-always: superseded tip verdicts structurally unobservable
VERDICT: REJECT(with: accepted-risk, ratified)
root: none — this is D25 ADDENDUM working as designed (DECISIONS.md:327-333 explicitly supersedes tip-green conjunct and names compensations (i)/(ii) as mandatory-on-every-push).
rationale: serialization (cancel-in-progress:false on platform job) taxes every push to insure an edge case the human already accepted with procedural compensation. Alternative: queued concurrency — costs minutes per push for information about a superseded tip nobody can act on (the tip is gone).

**17. INFRA-99c** — no test validates ci.yml
VERDICT: ADOPT — textual delegation-law probe now; execution-level coverage stays deferred.
root: D17 made the script SSOT but left the YAML side of the delegation unwitnessed — editing YAML alone to add gates (the D16d C7 failure mode) would pass every gate today.
rationale: screened candidate is sound and cheap. **Assumption-audit finding:** GATES.md prose says "action-pinned SHAs" unguarded — reality checked: actions are TAG-pinned (`actions/checkout@v4` etc.), so any SHA-pinning assertion would ship RED; probe scoped to what's true. Execution-level (running compose validation etc.) is disproportionate.
now-fix: files=[tests/test_ci_yml_probe.py (new)] · changed-lines≤22 · acceptance=`uv run pytest tests/test_ci_yml_probe.py -q`. Asserts: exactly one `run:` line invoking `bash scripts/run_local_ci.sh`; no other `run:` line contains pytest/ruff/mypy/cov (delegation law, comment-safe line-level parse). **SLATE B.**

**18(#19). GAP-1** — rogue staging guarded by prose only
VERDICT: MODIFY(to: index-delta-vs-declared-adds comparison — gate diffs current index against the acting contract's sanctioned exact `git add` lines, ack-token allowlist for multi-lane trees; hard non-empty-index fail REJECTED)
root: read-only custody was defined as role prose instead of a checkable index invariant.
rationale: packet's own DESIGN NOTE is correct and dispositive: sanctioned contracts DO stage exact lines (STATE.md hazards; three lawful-strike contexts), so hard-fail criminalizes obedience. Alternative: trust-plus-audit after the fact — loses: strikes happened ×3 post-prose (FIXES.md:58-80); audit finds corpses.
now-fix: DEFER → board-row **STAGE-GUARD** (allowlist/ack design >window).

**19(#20). GAP-2** — fabricated confirmation channels prose-only
VERDICT: ADOPT(deferred) — extend probe family: claims matching user-confirmed/ordered/approved vocabulary must carry a probe-C-resolvable EVENT citation within the paragraph or the document fails.
root: fabrication was cheap because confirmation had no required FORMAT — grammar is what made event-ids verifiable; absence-of-citation is mechanically detectable the same way.
rationale: probe C's grammar+registry machinery already exists (test_governance_probes.py:48-56,322-344) — this rides it. False-positive control via existing HISTORICAL_MARKERS exemption. Alternative: adversarial-review-only — loses: TIERREV-1 lesson (FIXES.md:142-147): mechanical ~1s probes catch what adversarial reading missed.
now-fix: DEFER → board-row **CONFIRMATION-CITE** (~25 lines, next probe window).

**20(#21). GAP-3** — see block 14 (single ruling surface; enforcement twin of INFRA-98).

**21(#22). GAP-4 ≡ store-then-hash verifier** — no mechanical recomputation of registry ids
VERDICT: MODIFY(to: make post-posting recomputation a REQUIRED owner step in §14 recipe language — 2-line contract amendment; network-fetch probe REJECTED as a gate)
root: the recipe existed but nothing marked verification mandatory-vs-advisory, so the genesis rows were hashed from drafts and nobody had to notice.
rationale: an always-on gh-api probe violates D22 (no gate requires github.com online); the §14 recipe commands already exist (:363-386) — the missing piece is obligation wording. Genesis anomaly proves drafts diverge from stored bytes (13 variants, zero matches — FIXES.md:177-193).
now-fix: files=[agents/contracts/MAIN_AGENT_CONTRACT.md] · changed-lines≤2 · acceptance=`sed -n '343,352p' agents/contracts/MAIN_AGENT_CONTRACT.md` shows amended recipe paragraph (recomputation declared mandatory, unverified posting void-by-policy). **SLATE D.**

**22(#23). GAP-5** — ruff scans tree not commit; foreign-WIP false-red/green
VERDICT: MODIFY(to: opt-in `--clean-lint` mode scoped to ruff+mypy only, executing against a `git worktree add` snapshot of HEAD; default scan unchanged, never silent)
root: gate input ambiguity — the gate's subject (the commit) and its input (the worktree) differ whenever lanes share a tree, and nothing names which is being judged.
rationale: packet screening right to warn: interacts with --static discipline and parity's remote-ref inputs (which must NOT snapshot). Partial snapshot (lint/typecheck only) is the honest scope; pytest stays tree-bound (env-dependent). Alternative: attribute-and-wait discipline only (current FIXES.md remediation) — loses: it failed twice before being written down; it's the compensating control, not the fix.
now-fix: DEFER → board-row **SNAPSHOT-LINT** with that scope spec.

**23(#24). GAP-6** — post-landing conformance unchecked (D16 drift class)
VERDICT: ADOPT(deferred) — machine-readable `accepts:` line on NEW ratifications + asserting probe; explicitly NO retro-backfill.
root: rulings carried acceptance criteria in prose only, so "did the diff match the ruling?" had no mechanical oracle.
rationale: D16-PARITY shipped wrong (.github/parity-era.env + F1b missing) despite a finalized design (FIXES.md:163-175) — the class is real and recurring under session length. Backfill rejected: rewriting 26 ratified prose rows creates churn + new probe-C exposure for zero forward protection. Alternative: post-landing human spot-check — loses: that was the process that missed D16.
now-fix: DEFER → board-row **ACCEPTANCE-GRAMMAR**.

**24(#25). GAP-7** — self-directed initiative after mandate (phantom-channel)
VERDICT: REJECT(with: do-nothing — self-attested compliance is theater)
root: none addressable here — the proposed control asks the untrusted party to attest their own restraint.
rationale: the phantom-channel adversary self-certifying "post-mandate actions: none" manufactures a compliance SIGNAL with zero constraint — worse than nothing, because green checkboxes erode vigilance. Real controls exist and worked: registry forensics caught it, interrupt + lane closure followed (STATE.md hazards). Alternative considered: registry-diff watchdog automation — loses: that is main-agent monitoring posture, already demonstrated effective; codifying it belongs to no report schema.

**25(#26). R11-teeth-map** — gaps 1–7 need explicit threshold mapping
VERDICT: MODIFY(to: the mapping IS this packet — delivered threshold-per-gap below)
root: **assumption-audit finding:** no D1–D7 enumeration exists anywhere in-tree (grepped factsheets/, DIGEST.md, DECISIONS.md — absent); the slots live only in R11 board row a682f9f5 prose ("teeth decisions D1-D7 … await human threshold calls"). The register couldn't be mapped because it was never written down — thresholds outran documentation, exactly the row's own root cause.
rationale — the requested mapping (gap → threshold class):
| Gap | Threshold ruling |
|---|---|
| G1 rogue staging | enforcing gate AFTER design (**STAGE-GUARD**) |
| G2 fabricated confirmations | probe-level now-design (**CONFIRMATION-CITE**) |
| G3 review bypass | enforcing gate, grammar-first (**TIER-GATE**) |
| G4 event-id integrity | procedure-pin now (SLATE D) + HUMAN-CALL on pilot-row repair (#29) |
| G5 foreign-WIP scan | opt-in tool mode, not default (**SNAPSHOT-LINT**) |
| G6 ruling conformance | probe over NEW rows only (**ACCEPTANCE-GRAMMAR**) |
| G7 self-direction | accepted-risk-with-documentation (REJECT #24) |
now-fix: n/a (this block's deliverable is the table; main agent transposes it into the ratification packet R11 asks for).

**26(#27). R11-queued-tripwires** — dead-code census · ignore-file pin · tamper lock
VERDICT: mixed, ruled individually:
- dead-code census — MODIFY(to: advisory report script, output filed to backlog, NEVER a red gate) · root: census tools measure suspicion, not guilt; gating on them burns trust like the 44-finding experiment lint backlog did · alternative: vulture-in-CI — loses: false-positive tax on every run · DEFER → **DEADCODE-CENSUS**.
- ignore-file pin — REJECT(with: redundant — `.gitignore` is already byte-pinned cross-branch as a parity SHARED entry, scripts/check_branch_parity.sh:63; within-branch edits are reviewed diffs like every tracked file) · root: a guard proposed for drift that an existing stronger guard already covers · alternative: semantic-drift probe (no new ignores without comment) — loses: pure style law, zero incident behind it.
- tamper lock — MODIFY(to: append-only COMMIT-TIME enforcement design (ledger diffs must cite authorizing contract/event), NOT a worktree lock) · root: records-of-record had no writer-side authorization channel, so any tree-write indistinguishable from tamper · rationale: naive lock breaks the concurrent-lane model operating RIGHT NOW (this tree carries simultaneous ledger edits; STATE.md hazards bless them) · DEFER → **TAMPER-LOCK**.

**27(#28). R11-priorities** — mechanization order vs GAP items
VERDICT: MODIFY(to: this ordering, by demonstrated-harm-first): ① TIER-GATE (exploit shipped: 7c03347) ② STAGE-GUARD (×3 strikes) ③ ACCEPTANCE-GRAMMAR (drift class, D16) ④ CONFIRMATION-CITE ⑤ SNAPSHOT-LINT ⑥ DEADCODE-CENSUS ⑦ TAMPER-LOCK. Rubric Q1–Q3: mechanize ONLY the mechanical subsets (root:-line presence, simpler-form greps, sizing arithmetic) as ASSISTS; the three kill-questions themselves are judgment acts and stay human/senior — full rubric automation is a category error, ~5/32 mechanical is the honest ceiling, not a deficiency.
root: priority inversion risk — automating cheap checks ahead of ones guarding demonstrated exploits.
now-fix: n/a (ordering is the deliverable; feeds the ratification packet).

**28(#29). BOARD-genesis** — six pilot event-ids void-until-repinned
VERDICT: HUMAN-CALL(owner: human, ask: choose supersede-and-repin vs recipe-amendment for pilot rows b16fb9ca/e1f7cc62/493e21ce/3afcd9b1/7595cb13/555b6fb8 per anomaly row ae44dbfd — this affects whether ALL pre-board citations resolve or only genesis-era ones)
root: ids were computed over pre-posting artifacts — the hash protocol predates the store it must hash.
rationale: routing here was pre-flagged by the packet as the standing HUMAN-CALL example and I concur: both options rewrite ledger history semantics, a human-trust decision, not derivable. Interim control stands: store-then-hash posting (mechanically pinned by SLATE D). Note: probe C passes today DESPITE void ids — it checks shape/uniqueness, not cryptographic validity, so nothing is red while six foundational citations are technically void. That silent-validity illusion is why this must stay flagged, not deferred quietly.

## SLATE (ready-to-execute; 4 files, ≤60 lines, reversible, test-first)

| # | File | Δ | Item |
|---|---|---|---|
| A | `scripts/run_local_ci.sh` | +1 token (line 53: `tests/` → `tests/ project/tests`) | #5/92b |
| B | `tests/test_ci_yml_probe.py` (new) | ≤22 lines | #17/99c |
| C | `tests/test_branch_parity_scripts.py` | +≤32 lines (custody falsifiability: novel-blob fires off-anchor; stale-pin fires false-red) | #10+#11/95a+95b |
| D | `agents/contracts/MAIN_AGENT_CONTRACT.md` | +2 lines (§14 recipe: recomputation mandatory, unverified posting void-by-policy) | #21/GAP-4 |

Arithmetic: 1+22+32+2 = 57 ≤ 60 lines, 4 files. Acceptance commands: A `uv run pytest tests/ project/tests -n 4 --dist worksteal -q -p no:cacheprovider \| tail -2` (proven green today: 973 passed, 1 skipped, 101.96s — job bash-1) · B/C `uv run pytest <file> -q` · D `sed -n '343,352p' agents/contracts/MAIN_AGENT_CONTRACT.md`. All additive except A's one-token edit; revert = git checkout.

## DEFER (one line each → board row)

1. **CI-SELFTEST** [90a+90b] — execution harness for tier parse/suppression/aggregation/banner incl. distinct STATIC vocabulary.
2. **PROJECT-LINT** [92a] — fix 12 project/ ruff findings (10 auto-fixable) → widen ruff roster next window.
3. **MAIN-DAY-DRYRUN** [93b] — sync_to_main() throwaway-clone rehearsal; **deadline: before PARITY_ERA flip**.
4. **PARITY-GUARD-TESTS** [94] — anchor shape/resolution + non-ancestor-taxi + drifted-tip negatives through real dispatch.
5. **SHIP-LAW-TEST** [96a] — ship.sh refusal/single-push law via shimmed bash.
6. **HOOK-TRAVEL** [96b] — tracked `.githooks/pre-push` writer-of-record + setup script + equality probe (no false-red on fresh clones).
7. **TIER-GATE** [98≡GAP-3] — Tier-trailer staging/pre-push rejection; precondition: ratify trailer-validation grammar (FULL ⇒ resolvable verdict id).
8. **STAGE-GUARD** [GAP-1] — index-delta-vs-declared-adds allowlist design; hard empty-index fail rejected.
9. **CONFIRMATION-CITE** [GAP-2] — confirmation-vocabulary claims require probe-C-resolvable citation.
10. **SNAPSHOT-LINT** [GAP-5] — opt-in `--clean-lint`: ruff+mypy over HEAD worktree snapshot; defaults unchanged.
11. **ACCEPTANCE-GRAMMAR** [GAP-6] — machine-readable `accepts:` on new ratifications + probe; no backfill.
12. **DEADCODE-CENSUS** [#27] — advisory census to backlog, never a gate.
13. **TAMPER-LOCK** [#27] — append-only commit-time ledger enforcement honoring concurrent-lane model.

## Assumption audit (≥3 re-verified + beyond-brief finds)

Re-verified: hook untracked/unconfigured (J); ci.yml:47 bare delegation + kubeconform scope :71-80 + cancel-in-progress :13; zero sha8-verifier machinery; classifier classify():104; D25 ADDENDUM :327-333; FIXES.md incidents as briefed. Beyond-brief finds: (1) 4 of 6 project/tests files are TRACKED yet outside every gate — blind spot bigger than briefed; (2) GATES.md:1128 "distinct banners per tier" claim false — --static prints LOCAL-CI GREEN identical to full (sharpened #2); (3) ci.yml actions are TAG-pinned not SHA-pinned — SHA-probe would ship red (scoped #17); (4) no in-tree D1–D7 enumeration exists (#25 root); (5) hashlib exists in conftest.py/samples — factsheet's "zero hits" imprecise, correct claim is "zero event-id verifier"; (6) combined suite 973P/1S green WITH foreign WIP present — attribution caveat recorded.

OPEN QUESTIONS: (a) should STATE.md suite-tail expectations refresh to a 973P baseline after SLATE A lands (main-agent call, WIP-dependent)? (b) does any second deployed clone rely on .git/hooks/pre-push today — scopes HOOK-TRAVEL? (c) who owns writing the canonical D1–D7 enumeration the human status review referenced?

`VERDICTS: 10 adopt, 11 modify, 7 reject, 1 human-call`

## Judgment

The missing tooth that bites hardest is **TIER-GATE** (INFRA-98≡GAP-3): it is the only deferred tooth guarding against an exploit that has ALREADY succeeded — `7c03347` shipped `Reviewer: none` ungoverned while every piece of enforcement machinery sat fully built and fully tested in-tree (classifier :104, seven classifier test nodes, probes); the sole missing weld is the invocation that turns classification into refusal. Every other gap awaits a first recurrence; this one documents a demonstrated one, and until the weld lands the entire FULL/CHECKLIST edifice — reviewer depth, tier honesty, verdict traceability — remains advisory grammar that pressure can bypass exactly as it did before. Runner-up: the six cryptically void genesis rows (#29), which corrode the trust substrate every OTHER tooth verifies against, but that wound is already human-flagged and fenced with interim controls. I killed seven items outright (network-path CI, behavioral cov-floor testing, dev-era comparison, kubeconform generalization, cancel-in-progress insurance, ignore-file double-pinning, self-attestation theater) because a teeth packet that only ever adds teeth ends up with a mouthful of decorative enamel — several of those "gaps" are doctrine working correctly, and saying so is the job.
