# PACKET G — MODEL-PLANE (training · HPO · evaluate · stats-library · inference) · SENIOR RULING (verbatim, 2026-08-24)

## Step-0 echo

- **Hash gate:** HEAD `5016e937e6aa67b301e3b54b78b1891f85784c67` (short `5016e93`) == dispatch stamp ✓ == gates.yaml `meta.head: 5016e93` (line 4) ✓.
- **Tree state:** 48 dirty/untracked entries — the foreign WIP already flagged (slate §5.4: `trust/*` deletions, process.py/structural.py mods, WIP ledger files, untracked arbitration/). Every ruling below derived from HEAD-registry content + live source reads this session. **Zero writes performed except this file; zero gh ops.**
- **Scanner gate (load-bearing for §1):** `agents/tools/render_gates.py` read live — `owner_symbol_token()` (:446-453) keeps only the FIRST identifier token after the owner path's colon (leading line number stripped) ⇒ symbol matching is class/function-name granular; dotted `.method` suffixes are invisible to it. `dupe_scan()` NEAR-IDENTICAL grouping (:559-566) keys on `(owner_head, frozenset(output id-tokens))` via `entry_keys()` (:216-219: id-like tokens else raw string) — transforms text is NEVER consulted. These two mechanics decide O6 and N1; the slate's flag-retirement claims are measured against them, not against hope.

---

## §1 · TWO-OWNER ARBITRATION — rows owned under model-plane paths (evidence = pasted transforms text, verbatim from gates.yaml; code anchors re-read at HEAD)

### R1 · O6 — src/broadway/features/pipeline.py :: FeaturePipeline — FEAT-23 + FEAT-24 → **CONFIRM-LAND** (split already materialized; residual flag is tooling-owned)

Pasted transforms:
- FEAT-23: `'fit order = encodings config order; per column: type==''target'' → TargetEncoding(columns=[col], smoothing).fit(df, df[target]) (pipeline.py:26-28); type==''frequency'' → FrequencyEncoding(columns=[col]).fit(df) (pipeline.py:29-31)'`
- FEAT-24: `'PINNED ORDER: df.copy() → build_derived (pipeline.py:42-48) → ALL frequency encoders (pipeline.py:49-50, fill=freq_fill for unseen keys) → ALL target encoders (pipeline.py:51-52)'`

Live verification: `fit()` def pipeline.py:21 (state reset :22-23, target/frequency branches :26-31); `transform()` def :34 (copy :35, builder_kwargs :36-41, build_derived :42-48, freq loop :49-50, target loop :51-52). Both owner strings are **already method-level and disjoint at HEAD** — the slate's prescribed repair ("narrow each owner to method level") is a no-op.

**AMEND (disposition note only; strings stand):** the OWNER-COLLISION flag cannot clear by ANY registry edit — `owner_symbol_token` reduces both owners to the identical token `FeaturePipeline`. The residual flag is a scanner false-positive CLASS that re-fires on every future two-methods-of-one-class gate pair. **Rider DEFER→GATE-REGISTRY-TOOLING:** refine token extraction to carry the dotted method suffix (`FeaturePipeline.fit` ≠ `FeaturePipeline.transform`), riding the live-SSOT dupe pin tests/test_gate_registry.py:718. Do NOT contort owner strings to dodge the scanner.

### R2 · O8 — src/broadway/stats/robust.py :: estimation_table — TRAIN-38 + STATS-48 → **CONFIRM-LAND**

Pasted transforms:
- TRAIN-38: `'T-BUG-1 HC3 SELF-FIT: derives HC3 independently of how the input was fitted — robust = model.get_robustcov_results(''HC3'') internally (:54), reads bse/conf_int from THAT object (:56, :60) …'`
- STATS-48 (outputs): `'TypeError "estimation_table requires a fitted statsmodels regression results object exposing get_robustcov_results…" (:48-53) — the ONLY raising guard in this cluster; coef/HC3_SE/CI_low/CI_high table derived by INTERNAL get_robustcov_results("HC3") re-fit (:54-56) …'`

Live verification: robust.py:34 `def estimation_table(model, alpha: float = 0.05)`; :47-53 `hasattr(model, "get_robustcov_results")` TypeError gate naming the received type; :54 internal HC3 re-fit; :56/:59-62 reads conf_int/bse/params from that object. The split is real and disjoint: TRAIN-38 owns the primitive derivation law (:34/:54/:56/:60); STATS-48 owns the fail-loud guard (:47-53) plus the consumer-cluster framing (regression/diagnostics/time_series/baseline). Cross-ref: B#11/S10 (ungated siblings standardized_coefs/scenario_dollars) untouched by this split.

### R3 · N1 — src/broadway/training/hpo.py outs(1) :: HPO-80 + HPO-82 → **CONFIRM-LAND + two AMENDs**

Pasted transforms:
- HPO-80: `'storage_url None → optuna.create_study(direction, sampler) IN-MEMORY; study_name unused on this branch (:141-142)'` · `'else RDBStorage(url, heartbeat_interval=60, grace_period=300) (:144-148) + create_study(load_if_exists=True) reopen (:149-155)'` · TPESampler seeded at construction (:140)
- HPO-82: `'suggest dispatch: suggest_int iff BOTH bounds are ints, else suggest_float (:107-111; legacy twin optuna.py:95-99)'` · `'optimize(n_trials=n_trials, callbacks=…) with NO timeout, NO n_jobs (intra-study sequential), NO catch= tuple … (:159)'`

Live verification: `run_model_study` def hpo.py:120; TPESampler(seed) :140; in-memory branch :141-142; RDBStorage :144-148; load_if_exists reopen :149-155; **optimize site #1 :159** (inside run_model_study) and **optimize site #2 :177** (inside `_optimize_study` :163-177 — the bandit continuation HPO-82's own transforms claim). Ownership declaration CONFIRMED as stated: 80 = creation/resume/callback wiring; 82 = trial appends.

- **AMEND 1 (owner precision):** HPO-82's owner names :159 only while its transforms also own :177 — formally ownerless code. Corrected owner: `src/broadway/training/hpo.py:159+:177 study.optimize call sites (run_model_study leg + _optimize_study bandit continuation)`.
- **AMEND 2 (flag-retirement mechanism):** "declare output ownership; kills the near-identical flag" is mechanically FALSE — the flag keys on shared owner path + shared `ARTIFACT-OPTUNA-STUDY` output token and never reads transforms; it retires only by renaming or minting artifacts, which vocabulary closure forbids. Re-classify the residual NEAR-IDENTICAL flag as the W7 co-tenancy view (create/append/storage triad, documented benign) exactly as W2 did for the canonical parquet.

### R4 · W5 / W6 / W7 — artifact rows naming model-plane writers → **CONFIRM (co-tenancy legal)**

- **W6** namespaces disjoint (pasted): HPO-83 `'nested run per COMPLETE trial named f''{study_name} trial {n}'' with tags {trial, study} (:80-84)'` vs TRAIN-36 rounded metric dicts landing in the training/evaluate runs. Different RUN instances; CUST-149 tag vocabulary is the declared join — cross-ref recorded, no further action.
- **W7** triad 80-create / 82-append / 86-RDB-twin: confirmed by the same lines quoted in R3; the watch-item (80 vs 86 duplicate storage wiring) stands, riders under both.
- **W5** (evaluate plane): TRAIN-35 assembles/persists, TRAIN-36 supplies rounded payload dicts, TRAIN-37 records decision fields `'… REWRITTEN with the skip signal (:187-194)'` — one writer process, three field families; benign.

---

## §2 · SLATE ITEMS — dispositions

### D1 · GATE-SURF-100 ← PROPOSE-GH2-01 → **AMEND** (lands) · STANDING QUESTION answered: **(a)**

Live evidence: `src/broadway/inference/api.py` is ONE docstring line (88 bytes, `__init__.py` empty); interpreter probe this session: module imports clean, `hasattr(module, 'app') → False`; `k8s/api-deployment.yaml:18-24` commands `uvicorn inference.api:app --host 0.0.0.0 --port 8000` under `replicas: 2` (:6) with HPA `minReplicas: 2` (:48) ⇒ every apply CrashLoopBackOffs on ALL replicas. FINDING substance accurate.

**(a) Gate as-is with the CrashLoopBackOff FINDING — a documented aspiration whose deploy stays BLOCKED until implemented.** Reasoning, consistent with prior packets:
- Packet D #8/#9 ALREADY adjudicated this exact surface: MODIFY(to: one F-1 lane implements the app object + correct uvicorn target) DEFER→DEPLOY-F1-API-LANE, root "manifests written against an aspirational contract nothing validates". Re-opening it as deletion would overrule a standing senior ruling with no new facts.
- This is a WIRED half-aspiration, not a corpse: the producing side is built and gated (promote_candidate chokepoint → CUST-147; get_champion/classify_champion → TRAIN-39; check_champion_manifest.sh certifier). Unlike KIND_LABELS (C#9: zero production references → DELETE) or render_index (C#15: alive only through test imports → kill-before-wiring), the serving stub completes a chain whose every other half is live, pinned code. B#11's KEEP precedent (declared contract + active consumers defeat deletion-first) applies with MORE force here.
- Deletion-first is honored as the REFUSAL CONDITION, not the default: if DEPLOY-F1-API-LANE is not executed by the next arbitration cycle, the manifests die instead (drop the Deployment+Service+HPA set demanding the unbuilt app). That decision belongs to INFRA-130's MERGE-WITH-DEPLOY-F1-API-LANE row — no new gate minted, no duplicate custody.

**AMENDs (exact):**
1. Owner `src/broadway/inference/api.py:1 app` asserts a symbol that DOES NOT EXIST — violates precise-owner-symbols. Reword to demand form: `src/broadway/inference/api.py (module stub — NO app symbol at HEAD; gate demands its creation)`.
2. FINDING tightened: uvicorn fails AT STARTUP binding the ASGI target (attribute-not-found class on `inference.api:app`), container exits non-zero, BackOff on both replicas — not a per-request predict failure.
3. `validated_by: []` is lawful ONLY under the FINDING/root rule — satisfied today; the moment the app exists the gate MUST gain an import-contract pin (target importable; /health /predict /metrics declared) BEFORE DEPLOY-F1 unblocks — record as the gate's first `if_changed` entry.

### D2 · E2E-D1 → GATE-CFG-112 → **AMEND** (finding + LAND confirmed; owner path corrected)

Finding verified live: configs/sample/taxi_diagnostic.yaml:5-6 ships `column_mapping: {Borough: pickup_borough}` (source→logical) while ALL THREE consumers read logical→source — `sample.column_mapping.get(<logical>, <logical>)` at src/broadway/stats/module.py:42, src/broadway/stats/describe.py:132, src/broadway/timeline/runners.py:100 ⇒ the remap can never fire at HEAD; the three cited tests pin the consumers' direction, so the SHIPPED YAML is the inverted artifact. LAND confirmed.

**AMEND (owner path is WRONG):** `SampleSpec` does not exist in samples/loader.py — that file contains ZERO column_mapping references (it validates provenance digests/schema only, loader.py:51-88). True definition: `src/broadway/lineage/models.py:37 class SampleSpec` (field `column_mapping: dict[str, str] = {}` at :43, `extra="forbid"` at :38). Corrected owner: `src/broadway/lineage/models.py:37 SampleSpec.column_mapping + configs/sample/taxi_diagnostic.yaml:5`.

### D3 · E2E-D4 / E2E-D5 OPEN slots → **DROP** (unrecoverable; no number resurrection)

Un-recoverability evidenced: repo-wide sweep for `E2E-D` yields exactly D1/D2/D3 (slate §2 rows) and D6 (which survived because it was RE-DERIVED live and pinned inside the gates.yaml HPO-82 finding at clean HEAD); `/tmp/broadway-e2e/` holds step logs + `MANIFEST-run1.sha256` (sha256 lines only — head inspected) + mlflow db/server logs; zero findings notes; the tokens D4/D5 appear NOWHERE else in agents/, experiments/, or docs/.

Re-derive REJECTED as a slot-resurrection mechanism: without the original claims' text, anything written into D4/D5 would be NEW findings wearing recycled numbers — self-attestation by construction, and a vocabulary-closure breach (ids would denote content nobody can cite). Register-as-UNVERIFIED likewise rejected: a ledger row asserting existence of evidence nobody can produce is the same lie, softer-spoken. **Refusal condition:** the base-run operator may resubmit the original texts as FRESH proposals entering dense order after this cycle (≥154); they never inherit D4/D5. Note: the e2e *-r2 artifacts (logs-{evaluate,train,features}-r2.txt, 22:46) document the promotion-inversion probe ALREADY ruled as board item B#10/S9 — no lost content hides there.

### D4 · GATE-TRAIN-118 (PROPOSE-GH2-03) — baseline subprocess git-rev-parse purity → **AMEND** (lands, carries FINDING/root)

Live verification (src/broadway/baseline/module.py): `_git_commit()` :20-27 — `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)`; `except (subprocess.CalledProcessError, FileNotFoundError): return "unknown"`; consumed by `_build_trace` :30-36 → `ArtifactTrace.commit` (trace.py:12). The slate's "silent 'unknown' fallback :26-27" quote is exact.

Purity defects CONFIRMED:
1. **Silent degrade** — the module has `logger` (:17) but the fallback emits NOTHING: artifacts carrying `commit: "unknown"` are indistinguishable from provenance-bearing ones. Violates the warn-visible vocabulary.
2. **CWD-dependent** — rev-parse resolves against process CWD, never anchored to a repo root; the train-job image does not ship `.git`, so deployed runs record `"unknown"` EVERY time, silently. Provenance theater in precisely the lane the manifests provision.
3. The wall-clock twin (`created_at`, :32) is D#26/REPRO-TIME-FREEZE territory — cross-ref only, no duplicate custody claimed here.

**AMEND (anchors):** slate says `run()` is at :45 — actual `run()` def is **:75** (:39-45 is `load_persisted`). Corrected owner: `src/broadway/baseline/module.py:75 run() (+ _git_commit() :20-27 subprocess-provenance leg)`. Output anchors made precise: BaselineResult JSON saved :85; lineage node_id("baseline") + parents :87-92.

Disposition: LAND with `validated_by: []` + FINDING/root (lawful combination). Fix shape ≤2 files: on the fallback branch emit `logger.warning` naming the exception class + record the test pin (monkeypatched CalledProcessError → assert value AND warning; FileNotFoundError twin). Raise-on-missing REJECTED: containers legitimately lack `.git`; baseline must not die for metadata. Deletion-first applied honestly: deleting `_git_commit` strips a contract field consumed via ArtifactTrace — rejected (B#14 keep-with-rationale class).

### D5 · GATE-ETL-115 (PROPOSE-GH2-06) — download.py orphan fetcher → **OVERRULE→NOT-GATE · dead-code-DELETE-referral**

Census (live): production callers of `download()` = **ZERO** (src/project/experiments/scripts swept; sole importer is tests/test_download.py:17). Zero mentions in README/docs/pyproject.toml/.env.example. `import requests` occurs nowhere else in src/. This is a corpse maintained by its own test file — no API.md-style contract, no consuming experiment (the contrast that saved B#11 applies in reverse).

Why NOT-GATE rather than gate-the-hazard: the function derives the on-disk name from the URL basename (unquoted) and stream-writes `open(dest, "wb")` (download.py:17-25) — truncating, with NO collision/existence guard. Wired as-is it would SILENTLY OVERWRITE canonical raw files (a URL ending in `/taxi.csv` clobbers the exact raw frame GATE-INGEST-01 globs). A gate would bless that path; the proposal's own purpose line concedes the remedy is "wire-or-delete".

Referral arithmetic (pure deletion, reversible):
- PRIMARY: DELETE `src/broadway/data/download.py` (−26 lines) + `tests/test_download.py` (−101 lines) = 2 files, −127 lines, zero production blast radius.
- RIDER (must NOT bundle): `EnvironmentConfig.download_chunk_size` (config/schema.py:67) is load-bearing for the env int-coercion law (gates.yaml :2366) and constructed in ~8 test sites + experiments/mlflow/_common.py:156 + 3 environment yamls. Removing the field touches ≈11 files ⇒ separate CFG-owned row, sequenced WITH the yaml-key removals — under C#24's adopted `extra="forbid"` the keys and the field must move TOGETHER or boot dies. Until then the yamls keep the key harmlessly.

### D6 · adjacent Band-04 slate items (spot-verified at HEAD)

| item | disposition | evidence |
|---|---|---|
| GATE-TRAIN-119 causal design step | **CONFIRM-LAND** | anchors exact: `run()` causal/module.py:18; `save_design(design, out_path)` :35; `write_record(node_id("causal", …), parents [baseline, analysis])` :37-42 |
| GATE-TRAIN-120 MLflow connectivity teeth | **CONFIRM-LAND** | `_is_unreachable_http_store` marker net :66-72; RuntimeError + README hint :63/:75-76; validated_by file tests/test_mlflow_utils_unreachable.py exists |

---

## §3 · COUNTS

- Ruled: 10 items (SURF-100 · E2E-D1/CFG-112 · E2E-D4/D5 · TRAIN-118 · ETL-115 · FEAT-23/24 · TRAIN-38/STATS-48 · HPO-80/82 · TRAIN-119 · TRAIN-120) + 3 two-owner artifact rows (W5/W6/W7) + 1 standing question.
- Per class: **CONFIRM-LAND 5** (O6 split · O8 split · N1 split · TRAIN-119 · TRAIN-120) · **AMEND 3** (SURF-100 · CFG-112/D1 · TRAIN-118 — every correction enumerated above, each item otherwise landing) · **OVERRULE→NOT-GATE 1** (ETL-115 → dead-code-delete-referral, 2 files/−127 lines + coupled CFG rider) · **OVERRULE→DEFER-lane-X 0 whole-item** (1 rider: O6 scanner token granularity → GATE-REGISTRY-TOOLING) · **DROP 1** (E2E-D4/D5 pair, two slots) · co-tenancy CONFIRMs 3 (W5/W6/W7).
- Standing question on SURF-100: **option (a)** — see D1.

## §4 · Judgment

The SURF-100 ruling sets the cycle's precedent for aspirational surfaces: an existence gate may hold `validated_by: []` ONLY while its FINDING states plainly that the demanded symbol does not exist and the deploy lane is blocked — the moment the registry pretends the symbol is there (`api.py:1 app` as written), the SSOT starts lying, and lying registries are how CrashLoopBackOffs get documented as healthy. Hence (a) with amended owner wording, funded-or-dies refusal condition attached. Second lesson, from O6/N1: the hygiene scanner defines what "clean" means, and its granularity limits (class-name tokens; transform-blind near-identical grouping) must be recorded as tooling facts — otherwise every honest registry carries permanent red flags, and teams learn to silence flags by corrupting artifact vocabulary instead of fixing the probe. Third, D4/D5: findings without persisted text die with their session; the e2e lane's only survivors (D1–D3, D6) lived because someone re-derived them against live code. Drop the slots, keep the lesson.
