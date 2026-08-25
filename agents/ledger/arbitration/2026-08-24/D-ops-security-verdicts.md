# PACKET D — OPS-SECURITY · SENIOR RULING (verbatim, 2026-08-24)

## Step-0 echo

- `git rev-parse --short HEAD` → **5016e93** == dispatch stamp ✓ · branch **sklearn** ✓
- Tree: 40 dirty/untracked paths, all unrelated WIP as briefed — custody-relevant hot files flagged below: `agents/ledger/FIXES.md`, `agents/ledger/DECISIONS.md`, `agents/ledger/STATE.md`, `src/broadway/cleaning/structural.py`, `tests/test_assumptions.py`, `project/etl/process.py`, untracked `project/tests/test_ingest_*.py`, untracked `experiments/more_modeling/{16..22}_*.py`
- STATE.md contradiction echo: **none** (lanes "(none) in flight"; board rows R6–R11 match this packet's five finding lanes; hazards honored — zero writes performed)

## RULING TABLE

| # | FINDING | VERDICT | root | rationale (alt considered → why it loses) |
|---|---|---|---|---|
| 1 | SEC-S1 cred live at PUBLIC refs | **HUMAN-CALL**(owner: human; ask: rotate upstream + evidence row) | revocation lives at provider, not git history (D22) | history rewrite loses per D22 (force-push mid-ladder); verified old cred byte-identical at `origin/main` 1860709 + `origin/taxi_work` 5c6370f HEADs |
| 2 | SEC-S2 rotation unevidenced; false `Ledger:` citation | **HUMAN-CALL**(owner: human/board; ask: ratify D24 propagation ordering + land corrected backfill) | no standing rule forced propagation/evidence check after branch-local merge | trailer verified @8b73645:13 vs `grep -c SECRET-1`=0 @HEAD FIXES.md; backfill exists **uncommitted** (WIP lane custody) |
| 3 | SEC-S3 D22 rider path wrong | **MODIFY**(to: rider disambiguated to `k8s/optuna/secret.yaml`; WIP FIXES.md:202 misquote fixed pre-landing) | ledger prose lacks canonical-path discipline | **premise killed**: string "configs/secret.yaml" exists in NO ref/artifact — born in-session; rider wrote bare "secret.yaml". DEFER→LEDGER-D22-PATH-DISAMBIG (both files WIP-hot) |
| 4 | SEC-S4 mechanics closure (context) | **ADOPT** — closure claim VERIFIED | template+lifecycle makes untracking structural | sklearn/taxi carry 606B/3×CHANGE_ME template; .gitignore:36-38 covers secret.local.yaml; zero-value sweep incl. untracked caches clean |
| 5 | SEC-S5 .env drift both directions | **ADOPT** → SLATE A | `.env.example` not treated as schema SSOT | key sets fully **disjoint** (5 API-only vs 6 infra-only); `.env` never committed in any ref (verified) |
| 6 | SEC-S6 dev literal creds | **MODIFY**(to: probe asserts staging/production interpolate-only; dev literal = documented accepted deviation) | no policy line separates dev-local literals from interpolated tiers | packet sketch ("no literal in development.yaml") rejected: removal breaks out-of-box boot — loader.py has no default injection (`CONFIGS_DIR` getenv :34 only); value is localhost-scoped, ≠ leaked prod cred |
| 7 | SEC-S7 lookup symlink undocumented | **ADOPT** → SLATE B | data-path environment assumption undocumented | symlink→`/home/opc/ONE/learning/...` confirmed live; README/docs grep empty |
| 8 | DEP-F1a stub + wrong uvicorn target | **MODIFY**(to: one F-1 lane implements app object + target `broadway.inference.api:app`) DEFER→DEPLOY-F1-API | manifests written against an aspirational contract nothing validates | stub=1-line docstring ✓; bare `inference.api:app` + PYTHONPATH=/app/src ✓; fastapi/uvicorn present pyproject:38-39 |
| 9 | DEP-F1b api lacks MLflow URI | **MODIFY**(fold into #8 lane: env + Secret mount) DEFER→same row | runtime dependency with no deployment source | no env block in api-deployment ✓ |
| 10 | DEP-F2 train-job vs 7 ${VARS} | **MODIFY**(to: env block + secretRef + volumeMounts + PVC) DEFER→DEPLOY-F2-TRAINJOB | prod config requires ambient env contract no surface fulfills; local gates blind via concrete dev defaults | 7 ${VARS} ✓; int-coercion crash path schema.py:73 ✓; resolver root owned by C#21 |
| 11 | DEP-M1 configMap `environment` uncreated | **MODIFY**(to: create it or drop refs; extend kubeconform scope k8s/optuna/→k8s/) DEFER→K8S-CONFIGMAP-ENV | CI structurally blind to top-level manifest semantics | 3×configMapKeyRef postgres:24,29,34 ✓; zero ConfigMap creators |
| 12 | DEP-M2 train env MISSING | **REJECT**(with: duplicate consequence of #10 — one fix surface) | row splits one defect into two board entries | dedupe per packet note |
| 13 | DEP-M3 api env MISSING | **REJECT**(with: identical fact+fix as #9 — fold) | duplication, not a distinct finding | same env gap counted twice in register |
| 14 | DEP-M4 HPA 70 vs config 80 | **REJECT**(with: DO NOT wire; delete dead `api_hpa_cpu_threshold` [C#25]; manifest `averageUtilization: 70` becomes declared SSOT) | two owners for one number — wiring preserves duplication with new machinery | 70 (:56 area) vs 80 (all 3 env yamls) ✓; key unwired anywhere; alt "wire config→manifest" loses: platform logic to rescue a dead key |
| 15 | DEP-M5 broadway-mlflow:latest unbuilt | **MODIFY**(to: consume `mlflow-server:<sha>` or build that tag) DEFER→IMAGE-TAG-COHERENCE | image names unbound to build outputs | CI tags mlflow-server:<sha>:186 only ✓ |
| 16 | DEP-M6 broadway-postgres:latest no source | **MODIFY**(to: official postgres image + init.sql mount, or add Dockerfile) DEFER→same row | tag references nonexistent build | docker/postgres = init.sql only ✓ |
| 17 | DEP-M7 compose builds Dockerfile-less ctx | **MODIFY**(fix/remove context) DEFER→same row | ungated compose path rots independently | `build: ./docker/postgres` w/o Dockerfile ✓ |
| 18 | DEP-M8 artifacts w/o PVC | **MODIFY**(rides #10 lane) DEFER→DEPLOY-F2 | cwd-relative writes assume node-local disk | zero volume keys in train-job ✓ |
| 19 | DEP-M9 broadway:latest never built by CI | **MODIFY**(ci.yml builds/pushes it in build-and-boot) DEFER→CI-BUILD-BROADWAY | consumed image outside validated set | CI builds base/optuna-worker/mlflow-server only (:165,:179,:186) ✓ |
| 20 | DET-C1 ≥3 RNG families | **ADOPT** (context confirmed) | seeded surfaces fine; hazard class = literals | carriers spot-verified (yaml random_state / sample seed=42 / rng(0) / onboard 42) |
| 21 | DET-C2 dual-numpy lock | **ADOPT**, DEFER→ENV-NUMPY-SINGLE-PIN | resolver markers split numpy by platform → divergence class | uv.lock:584-585/:1199-1200 exact markers ✓; fix=`uv add 'numpy==2.3.5' && uv lock`; complements #27 (libm still varies) |
| 22 | DET-C3 shapiro literal seed | **ADOPT** — cross-ref only, B#18 owns | shared literal class | `default_rng(0)` assumptions.py:29 ✓ |
| 23 | DET-a unpinned subsample indices | **ADOPT**, DEFER→DET-A-SUBSAMPLE-PINS (didn't fit slate budget) | silent seed/generator drift undetectable | new cold test file pins choice indices |
| 24 | DET-b onboard literal 42 | **MODIFY**(carrier decision rides C#19; interim named constant + doctrine comment) DEFER→C#19 rider | YAML-SSOT doctrine lacks bootstrap exception | module LIVE (cli.py:148 imports init) so deletion rejected; onboard precedes configs existing — chicken-and-egg |
| 25 | DET-c qq.py global RNG mutation | **REJECT**(with: do-nothing) | none — `np.random.seed(42)` sits inside `if __name__ == "__main__":` (qq.py:819-820): zero library-path global mutation | brief overclaimed "process-global side effect"; local-RandomState rewrite rejected as editing working demo code to fix an effect impossible on import |
| 26 | DET-d wall-clock bytes, no freeze flag | **MODIFY**(deletion-first eval inside freeze design: drop created_at from hashed artifacts or add BROADWAY_ARTIFACT_CLOCK) DEFER→REPRO-TIME-FREEZE | artifact bytes keyed to wall clock, no injection seam | writers baseline/module, trace.py, samples/generate ✓; zero freeze machinery repo-wide |
| 27 | DET-e golden bare `==` | **ADOPT** → SLATE C | golden equality assumes single numpy+libm universe | committed :153-163 floats/lists/records exact-eq ✓ |
| 28 | DET-f experiment scripts literal seeds | **MODIFY**(attach seed-policy condition to foreign-experiments ratification rows) DEFER→R-lane rider | exploratory scripts sit outside config law pending ratification | literals ✓ (10 files, e.g. 16:133); scripts are untracked read-only WIP this session |
| 29 | DET-g+h timing persist / unique() order | **MODIFY**(g **CLOSED-POSITIVE**: persists via TrainingResult→artifacts/training/training_result.json; evaluate reads it back :64,:91 — feed into recorder spec. h DEFER→DET-H-ORDER-DOC post-WIP) | timing lacked named sink (now found); warning examples inherit input order via `unique()` :51,:71,:98 | structural.py WIP-hot blocks h now |
| 30 | PERF-P1..P5 recorder absent / ≥8× re-reads | **MODIFY**(split TELEMETRY-RECORDER lane [P1,P5,P4-fact] + PERF-IO lane [P2,P3]) both DEFER | D23-mandated recorder never built; stage interfaces path-keyed so the same frame re-crosses disk ≥8× | STATE.md has NO ## telemetry (headers enumerated live) ✓; contracts.run + baseline PREDICTION both `df = load(cfg.dataset)` full-frame for near-zero compute ✓ |

## SLATE (ready-to-execute · 4 files · ~60 lines · reversible · test-first)

1. **NEW `tests/test_config_secrets_probe.py`** (~38 lines): (i) parity — `set(.env keys) == set(.env.example keys)`, skipif no `.env`; (ii) `.env` untracked guard (subprocess `git ls-files --error-unmatch .env` must fail) + `.gitignore` line-1 assert; (iii) staging+production yaml `database_*` lines must match `${…}` interpolation — dev exempt, deviation documented in docstring.
   + **EDIT `.env.example`** +5 placeholder lines (DEEPSEEK/NVIDIA/OPENROUTER/ROUTER/ORCA API keys).
   Acceptance: `uv run pytest tests/test_config_secrets_probe.py -q` — RED before the example edit, GREEN after (test-first order).
   Covers #5 + tripwire half of #6.
2. **EDIT `README.md`** +~5 lines under setup: fresh-clone bootstrap for `data/raw/taxi_zone_lookup.csv` (symlink or TLC source fetch). Covers #7. Acceptance: `grep -q taxi_zone_lookup README.md`.
3. **EDIT `project/tests/test_ml_pipeline.py`** ~12 lines: golden float lists → `pytest.approx(..., rel=1e-12)`; records compare numeric fields approx. File is tracked-cold ✓. Covers #27. Acceptance: `uv run pytest project/tests/test_ml_pipeline.py -q`.

Rider dropped from slate for file-cap: add `.env` + `k8s/optuna/secret.local.yaml` to `.dockerignore` (+2 lines) → attach to K8S lane.

## HUMAN-CALL BUNDLE (for the user — agents must NOT attempt these)

**HC-1 · ROTATE THE DB CREDENTIAL (prime exposure)**
Evidence: old credential lives as k8s-Secret `stringData.DB_PASSWORD`, len=24 pfx `Toz`, sha256pfx `3865079c5ad65ce7`; byte-identical 308-byte manifest (file sha256pfx `3a532996800e3ca5`) at HEAD of **PUBLIC** `origin/main` (1860709) and `origin/taxi_work` (5c6370f); fix commit 8b73645 contained by sklearn + origin/sklearn + origin/taxi ONLY (verified via `git branch -a --contains`); locals main/taxi/taxi_work/pr-1/pr-2/sp_probe + tag tier-1-complete also carry it (factsheet). The file's own header admits: "Values are for the local kind cluster; rotate for any real deployment."
Steps: ① identify operator account/host of that Postgres; ② **rotate the password at the provider** (the only value-killer); ③ update any live consumer via lifecycle.sh env/secret.local.yaml — never commit; ④ post rotation-evidence row on board #5 (supersedes); FIXES.md SECRET-1 row closes only on that evidence; ⑤ **NO history rewrite** (D22 ratified).

**HC-2 · PROPAGATION ORDERING + LEDGER TRUTHING**
Evidence: remediation landed sklearn-lineage only; main frozen until main-day (exposure window on record); WIP FIXES.md:195-206 backfill exists but repeats the misquote `"configs/secret.yaml"` (:202) — string never existed in any ref (verified: `git log --all -- configs/secret.yaml` empty; `git grep configs/secret 5016e93` empty); 8b73645's `Ledger: FIXES.md` trailer was false at push time.
Steps: ① ratify the D24 single reconciliation pass as the propagation vehicle; ② owning lane corrects :202 wording before landing the backfill; ③ supersede rows on the board.

## DEFER → board rows (#5)

#3→LEDGER-D22-PATH-DISAMBIG · #8+#9→DEPLOY-F1-API-LANE · #10+#18→DEPLOY-F2-TRAINJOB-LANE (resolver root C#21) · #11→K8S-CONFIGMAP-ENV (+kubeconform-scope rider + .dockerignore rider) · #15/#16/#17→IMAGE-TAG-COHERENCE · #19→CI-BUILD-BROADWAY-LATEST · #21→ENV-NUMPY-SINGLE-PIN (`uv lock` recipe) · #23→DET-A-SUBSAMPLE-PINS · #24→C#19-RIDER · #26→REPRO-TIME-FREEZE (deletion-first eval) · #28→EXPERIMENTS-SEED-POLICY (R-lane condition) · #29h→DET-H-ORDER-DOC (post-WIP) · #30→TELEMETRY-RECORDER-LANE + PERF-IO-LANE (spec already in factsheet).

**VERDICTS: 8 adopt, 16 modify, 4 reject, 2 human-call**

## Judgment

The most urgent exposure is unchanged and human-owned: a real-looking DB credential still sits at HEAD of public `origin/main` and `origin/taxi_work`, and nothing agents can do inside this repo kills its value — upstream rotation (HC-1) is the kill, which is exactly why it must be ruled HUMAN-CALL rather than deferred into a board queue. Among agent-scope items I would kill first **DET-c (#25)**: the brief's "process-global side effect" dissolves on inspection because the seed sits behind a `__main__` guard — fixing it would be negative-value churn. Second kill **DEP-M4 (#14)**: wiring the dead HPA key would spend platform machinery to preserve a duplication whose correct end-state is deletion of one side. The deploy register's deeper truth is that twelve mismatch rows share two roots — manifests written against contracts nothing validates, and a kubeconform scope that cannot see them — so they route as coherent lanes, not twelve whack-a-mole fixes.

Assumption audit: residue sweep beyond the brief (untracked `.pyc`, `.venv`) found the old credential NOWHERE — an initial binary-match scare was traced to stderr interleaving from the freeze-grep, disproven by direct `-cF` counts (0); `.env` never committed in any ref; `.env`/`.env.example` fully disjoint; trainer timing persist path resolved positive against the sheet's "UNVERIFIED"; loader has no env-default fallback (grounds #6's MODIFY). OPEN QUESTIONS: ① who operates the leaked credential's Postgres (blocks HC-1 execution)? ② should top-level k8s/ be declared documented-aspiration until deployment day, given CI structurally cannot validate it? ③ commit custody for WIP-hot ledger files (FIXES/DECISIONS) — owning lanes or board?

---
*POST-ARBITRATION HUMAN RESOLUTION (2026-08-24): HC-1 VOIDED by human ruling — the stringData.DB_PASSWORD matter disregarded entirely; no rotation, no evidence row; SEC-S1/S2 urgency framing dissolved. HC-2 APPROVED with three ordered conditions (see 00-resolutions.md).*
