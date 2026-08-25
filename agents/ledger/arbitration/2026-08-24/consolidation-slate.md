# CONSOLIDATION SLATE — senior arbitration cycle #2

- Date: 2026-08-24 · Repo: /home/opc/ONE/broad-way · **step-0 VERIFIED: HEAD = `5016e937e6aa67b301e3b54b78b1891f85784c67` (short `5016e93`) ✓** matches gates.yaml `meta.head`.
- Read-only mission; this file is the ONLY file created. Zero gh ops. gates.yaml / render_gates.py / tests untouched.
- Inputs merged: (A) `render_gates.py --dupe-registry` → 9 OWNER-COLLISIONS + 9 ARTIFACT-TWO-WRITER + 1 NEAR-IDENTICAL (exit 3, DUPE(S) FOUND); `--dedupe-proposals agents/ledger/arbitration/2026-08-24` → 42 parsed · **40 UNIQUE / 2 OVERLAP-PATH / 0 DUP-OF-GATE**. (B) unparsed pools gap-cd-mlflow.md PROPOSE-GH5-01..06, gap-object-creators.md PROPOSE-GH6-01..06. (C) strays: E2E D1–D3 (D4/D5 not recoverable — see §5 flags), lifecycle.sh view_local backgrounding, tracked_run-wrapper + object-ledger proposal.
- Ruling vocabulary per contract: LAND · MERGE-WITH-\<gate\> · SPLIT-OWNER · REJECT-as-NOT-GATE · DEFER-to-lane-X.

---

## §0 · SLOT ARITHMETIC (why 100+)

Registry census (`gates.yaml`, 99 gates): every band is FULL to its dense ceiling —

| band | ids held | ceiling |
|---|---|---|
| 01-ingest | GATE-INGEST-01..09 | 9 |
| 02-etl-lookup | GATE-ETL-10..19 | 19 |
| 03-features | GATE-FEAT-20..29 | 29 |
| 04-training | GATE-TRAIN-30..39 | 39 |
| 05-stats | GATE-STATS-40..49 | 49 |
| 06-timeline-lineage | GATE-TLINE-50..59 | 59 |
| 07-surfaces | GATE-SURF-60..69 | 69 |
| 08-config-schema | GATE-CFG-70..79 | 79 |
| 80-hpo-optuna | GATE-HPO-80..89 | 89 |
| 09-infra | GATE-INFRA-90..99 | 99 |

**Zero free slots exist in bands 01–09 + 80–89.** Therefore all new ids continue the global dense order at **100…153**, prefixed by the band that owns the owner symbol. Exactly ONE new band is minted: **81-object-custody** — justified territory: write-side custody law over external stores (MLflow registry/aliases/experiments/tags; cross-plane object ledger) is owned by NO existing band's phase (TRAIN/HPO own compute legs that merely consume MLflow; INFRA owns repo/CI scripts; CFG owns YAML parsing). Density invariant preserved: order stays gapless 1–99 → 100–153; future insertions append ≥154; no id reuse; no renumbering of 1–99.

---

## §1 · TWO-OWNER ARBITRATION (all 19 dupe-registry groups; evidence = actual `transforms:` text)

### OWNER-COLLISIONS (9)

| # | path::symbol | gates | disposition | evidence (transforms excerpts) |
|---|---|---|---|---|
| O1 | project/etl/process.py :: select_and_clean_columns | INGEST-04 + ETL-17 | **PRECISE-SPLIT** | INGEST-04 owns transform semantics: "rename_columns applies cfg.rename_map (:94-95) · missing contract column(s) → ValueError listing sorted(missing) (:103-106) · df[cols].dropna() with ledger ('dropna', n) (:110-114)". ETL-17 owns the accounting grammar over the SAME lines: "ledger appended with ('dropna', rows_after) at :114 and ('duplicates', rows_after) at :119 · StageLedger stores ROWS-AFTER, not deltas". Narrow owners: INGEST keeps projection/drop semantics; ETL-17 owner string narrows to "process.py:110-119 ledger-append sites only". |
| O2 | src/broadway/cleaning/structural.py :: parse_numeric | INGEST-06 + ETL-15 | **PRECISE-SPLIT** | INGEST-06: "pd.to_numeric(errors='coerce') (:89) · _fractional_refusal … astype is SKIPPED (FX-A05) (:100-105)". ETL-15: same symbol but transforms are the persistence channel only: "collector append CoercionRecord{…} at :107-115; persisted by etl.run as \<name\>_coercion_audit.json (ARTIFACT-COERCION-AUDIT) + lineage node_id('coercion')". Split = repair semantics vs audit-persistence channel. |
| O3 | src/broadway/contracts/pandera.py :: build_raw_schema | INGEST-09 + CFG-75 | **PRECISE-SPLIT** | INGEST-09 enforcement leg: "CONTRACT enforcement: build_raw_schema(dataset).validate(df) AFTER coercion, coerce=False strict (etl/module.py:95)". CFG-75 schema-law leg (dtype whitelist mapping, pa.Object sentinel tail :14-27,:34-43). CFG owns dtype-vocabulary law; INGEST owns boundary enforcement call. |
| O4 | src/broadway/data/loader.py :: load_with_audit | INGEST-02 + ETL-11 | **PRECISE-SPLIT** | INGEST-02 admission leg: "extension whitelist READERS {.csv,.parquet,.xlsx,.xls}, else ValueError 'unsupported format' (:18-23, :126-128)". ETL-11 join leg: "per lookup: audit_join BEFORE merge, then df.merge(...suffixes=('','_lookup')) at :139 · merged_lookup_column_names() is the SSOT of the '_lookup' collision rename". Raw-read admission vs lookup-join semantics. |
| O5 | src/broadway/etl/module.py :: run (3-way) | INGEST-03 + INGEST-07 + ETL-10 | **CO-TENANCY-LEGAL** (+narrowed strings) | ETL-10 is the step orchestrator ("loader guard _assert_data_source_supported() … TransformAudit assembled; unexplained drops enforced"); INGEST-03 a named branch ("CI=true AND etl.ci_sample_size>0 → df.sample(...) reason 'CI sampling: -\<N\> rows' (:74-78)"); INGEST-07 a named sub-block ("persisted as StructuralCleanResult.parse_failures in \<dataset\>_clean.json (etl/module.py:129-137)"). Three genuine aspects of one function; keep all three with line-anchored owner strings. |
| O6 | src/broadway/features/pipeline.py :: FeaturePipeline | FEAT-23 + FEAT-24 | **PRECISE-SPLIT** (string repair) | Owners already point at distinct methods — FEAT-23 ":21 FeaturePipeline.fit()" ("fit order = encodings config order … TargetEncoding(columns=[col], smoothing).fit") vs FEAT-24 ":34 FeaturePipeline.transform()" ("PINNED ORDER: df.copy() → build_derived → ALL frequency encoders → ALL target encoders"). Collision exists only because the scanner matched the class name; narrow each owner to method level. |
| O7 | src/broadway/stats/module.py :: run (3-way) | STATS-40 + STATS-41 + STATS-43 | **CO-TENANCY-LEGAL** | Owners already disambiguated ":24 run()" guards ("None-guard both optional config sections BEFORE any IO"), ":31 run() data leg" ("existence checked BEFORE pd.read_parquet on whichever leg was selected"), ":50 run() floor binding" ("run_anova(groups, small_group_threshold=cfg.stats.min_rows_for_sampling)"). Distinct aspects, legal co-tenancy; strings stay as-is. |
| O8 | src/broadway/stats/robust.py :: estimation_table | TRAIN-38 + STATS-48 | **PRECISE-SPLIT** | TRAIN-38 anchors :34 primitive law: "T-BUG-1 HC3 SELF-FIT: derives HC3 independently of how the input was fitted — robust = model.get_robustcov_results('HC3') internally (:54)". STATS-48 anchors :47 consumer cluster "(regression/diagnostics/time_series/baseline cluster)". Narrow TRAIN-38 to the HC3 primitive; STATS-48 to the cluster call-sites (D14 scope). |
| O9 | src/broadway/timeline/walkthrough.py :: _write_timeline | SURF-60 + SURF-61 | **PRECISE-SPLIT** (string repair) | Same function, two render legs: SURF-60 "# renders via reports/index.py:61 render_dashboard() → reports/index.md"; SURF-61 "# renders via reports/timeline.py:16 render_timeline() → reports/timeline.md". Narrow each owner to its render call-site. |

### ARTIFACT-TWO-WRITER (9)

| # | artifact | writers | disposition | evidence |
|---|---|---|---|---|
| W1 | ARTIFACT-CANONICAL-FRAME | INGEST-05+06+08+09 | **CO-TENANCY-LEGAL (benign)** | Sequential refinement inside one canonicalize chain, single-threaded: 05 "pd.to_datetime(series, errors='coerce')" → 06 "astype(target_dtype)" → 08 "LOCKED ORDER (docstring :3-4): drop_duplicates → standardize_missing → parse_datetime → parse_numeric → dropna(subset=[target])" → 09 "schema.validate(df)". Each stage mutates then hands on; no competing writers. |
| W2 | ARTIFACT-CANONICAL-PARQUET | ETL-10 + ETL-16 | **CO-TENANCY-LEGAL (benign)** | ONE physical write site viewed by two gates: ETL-16 owns location law ("outputs \<data_dir\>/\<processed_subdir\>/\<dataset.name\>_canonical.parquet … written index=False at module.py:98"); ETL-10 orchestrates the step containing it. Writer-vs-location-law split; document, do not split gates. |
| W3 | ARTIFACT-CLEAN-EVIDENCE | INGEST-05+06+07+08 | **CO-TENANCY-LEGAL (benign)** | Append-only accumulation: 05/06 append ParseFailure entries ("failed mask = input.notna() & coerced.isna() → ParseFailure(column, count, examples[:5])"), 08 collects via canonicalize ("failures collected … inside canonicalize (cleaner.py:44, :59-69)"), 07 persists ("<dataset>_clean.json (etl/module.py:129-137)"). Producer-collector-persister chain. |
| W4 | ARTIFACT-COERCION-AUDIT | ETL-15 + INGEST-06 + INGEST-07 | **CO-TENANCY-LEGAL (benign)** | Same triad shape as W3: 06 produces CoercionRecords ("a CoercionRecord{column, declared_dtype, arriving_dtype, rows_affected} is appended to the caller-supplied collector"), 15 defines the channel contract, 07 persists ("<name>_coercion_audit.json + lineage record node coercion:\<dataset\>"). Consistent with O2 split. |
| W5 | ARTIFACT-EVALUATION-RESULT | TRAIN-35 + TRAIN-36 + TRAIN-37 | **CO-TENANCY-LEGAL (benign)** | 35 assembles/persists ("EvaluationResult … persisted BEFORE promotion ordering enforced by tests/test_evaluate_contracts.py:339-411"), 36 supplies payload dicts ("rounded metric dicts persisted in ARTIFACT-EVALUATION-RESULT"), 37 records decision fields ("(promote: bool, reason: str) recorded in ARTIFACT-EVALUATION-RESULT … REWRITTEN with the skip signal (:187-194)"). One writer process, three field families. |
| W6 | ARTIFACT-MLFLOW-RUN | HPO-83 + TRAIN-36 | **CO-TENANCY-LEGAL (disjoint namespaces)** | Different RUNS of the same class: HPO-83 "nested run per COMPLETE trial named f'{study_name} trial {n}' with tags {trial, study}"; TRAIN-36 metrics land in training/evaluate runs ("MLflow-logged metrics keys"). No shared instance; document namespace law (cross-ref CUST-149 tag vocabulary). |
| W7 | ARTIFACT-OPTUNA-STUDY | HPO-80 + HPO-82 + HPO-86 | **CO-TENANCY-LEGAL (create/append/storage)** | 80 creates/reopens ("storage_url None → optuna.create_study(direction, sampler) IN-MEMORY … else RDBStorage(url, heartbeat_interval=60…) + create_study(load_if_exists=True) reopen"), 82 appends trials ("optimize(n_trials=n_trials, callbacks=…)"), 86 is the k8s RDB twin lane. Watch-item: 80 vs 86 duplicate storage wiring — riders under both. |
| W8 | ARTIFACT-PIPELINE-PICKLE | FEAT-23 + FEAT-25 | **CO-TENANCY-LEGAL (state-owner vs serializer)** | 23 fits the state that goes INTO the pickle ("fitted encoder state _target_encoders/_freq_encoders … persisted later into ARTIFACT-PIPELINE-PICKLE (module.py:52-54)"); 25 performs the single dump ("pickle.dump (:52-54)"). One dump site. |
| W9 | ARTIFACT-RAW-FRAME | INGEST-02 + INGEST-03 | **CO-TENANCY-LEGAL (benign)** | Loader produces ("single pd.read_* of dataset.path (:129)"); CI-sample guard immediately mutates row count ("df.sample(n=min(ci_sample_size, len), random_state=…) (:74-78)"). Sequential same-pipeline mutation; folds under O5 narrowing. |

### NEAR-IDENTICAL (1)

| # | group | disposition | evidence |
|---|---|---|---|
| N1 | src/broadway/training/hpo.py outs(1): HPO-80 + HPO-82 | **PRECISE-SPLIT** (declare output ownership; kills the near-identical flag) | NOT duplicates: HPO-80 transforms are study creation/resume/callback wiring ("TPESampler seeded AT CONSTRUCTION … create_study(load_if_exists=True) reopen … callbacks=[_mlflow_callback]"); HPO-82 transforms are the optimize loop ("suggest_int iff BOTH bounds are ints, else suggest_float (:107-111) … optimize(n_trials=n_trials, callbacks=…) with NO timeout, NO n_jobs, NO catch="). Both list ARTIFACT-OPTUNA-STUDY as output → scanner collision. Rule: HPO-80 owns study creation/resume; HPO-82 owns trial appends. |

---

## §2 · CONSOLIDATED SLATE (54 proposed gate ids from 59 pool items)

Fields per item: purpose · owner path:symbol · inputs→outputs sketch · validated_by (existing node ids where known) · source · RULING-REQUEST.

### Band 07-surfaces

- **GATE-SURF-100** ← PROPOSE-GH2-01 · Serving-entry existence gate for the champion predict path · owner `src/broadway/inference/api.py:1 app` + `k8s/api-deployment.yaml:18 command` · inputs: [uvicorn target import, models:/taxi@champion] → outputs: [HTTP /health /predict /metrics :8000 or loud CrashLoop with named cause] · FINDING: api.py is a one-line docstring stub — no `app` symbol; uvicorn raises AttributeError ⇒ CrashLoopBackOff on all replicas · validated_by: [] · **RULING: LAND**
- **GATE-SURF-101** ← PROPOSE-GH2-07 · Gate the last ungated read-only CLI subparser (`ds-pipeline columns`) · owner `src/broadway/discover/columns.py:10 run()` · inputs: [raw CSV path] → outputs: [stdout dtype/null report] · validated_by: [] · **RULING: LAND**
- **GATE-SURF-102** ← PROPOSE-GH1-04 · experiments_ui FastAPI dashboard series-dispatch custody · owner `experiments_ui.py` FastAPI app · inputs: [results CSVs] → outputs: [dashboard series endpoints] · validated_by: [] · **RULING: LAND**

### Band 08-config-schema

- **GATE-CFG-103** ← PROPOSE-GH1-01 (OVERLAP-PATH w/ GATE-CFG-78 loader.py) · STEP_MODULES dispatch-map custody — the registry that decides which module each step name may load · owner `src/broadway/config/loader.py STEP_MODULES (composite)` · inputs: [step name argv] → outputs: [module binding or loud unknown-step error] · validated_by: [tests/test_gate_registry.py:691 (documents PROPOSE-GH1-01 as UNOWNED surface tripwire)] · **RULING: LAND** (PRECISE-SPLIT vs CFG-78: CFG-78 keeps resolve_full_steps merge law; 103 owns the dispatch map)
- **GATE-CFG-104** ← PROPOSE-GH3-02 (OVERLAP-PATH w/ GATE-TRAIN-32 project/data.py) · Single configs-root owner; kill the project-plane split-brain · owner `project/data.py` configs resolution sites · inputs: [configs/{dataset,project}/ trees] → outputs: [one resolved root, drift ValueError] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-105** ← PROPOSE-GH2-02 · Config-writing scaffolder custody (`ds-pipeline init`) — the only entry that mutates the contract SSOT tree · owner `src/broadway/onboard/module.py:215 init()` (_write_configs :178-198) · inputs: [stdin prompts + 13 argv flags] → outputs: [configs/dataset+analysis+experiment YAMLs, profile JSON, 1 lineage record] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-106** ← PROPOSE-GH2-05 · contracts-step checks parity gate (second "contract truth" surface) · owner `src/broadway/contracts/module.py:19 run()` (checks/checks.py rules) · inputs: [cfg.contracts.null_threshold, frame] → outputs: [pass/fail verdict, ValueError :30-33] · validated_by: [] · **RULING: MERGE-WITH-GATE-CFG-75** (duplicate contract-checking custody; seniors decide absorption vs narrow co-tenancy)
- **GATE-CFG-107** ← PROPOSE-GH3-01 · Bootstrap-existence gate for out-of-repo lookup symlink · owner `configs/dataset/taxi.yaml` lookup_tables path · inputs: [declared lookup paths] → outputs: [loud missing-artifact error pre-merge] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-108** ← PROPOSE-GH3-03 · Numeric env-parse teeth + dedup for threshold vars · owner `onboard/infer.py` env-parse sites · inputs: [env vars] → outputs: [typed values / named parse errors] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-109** ← PROPOSE-GH3-04 · Provenance tooth for the ambient CI switch · owner `module.py` ambient-switch site · inputs: [CI env flag] → outputs: [provenance-stamped behavior] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-110** ← PROPOSE-GH3-06 · One root-owner for the five output-tree env vars · owner `module.py` output-tree var cluster · inputs: [env vars] → outputs: [single resolved root law] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-111** ← PROPOSE-GH3-07 · Unify BROADWAY_MLFLOW_CONFIG resolution style · owner `mlflow_utils.py` config-resolution site · inputs: [BROADWAY_MLFLOW_CONFIG] → outputs: [single resolution dialect] · validated_by: [] · **RULING: LAND**
- **GATE-CFG-112** ← FINDING E2E-D1 · SampleSpec.column_mapping DIRECTION LAW (logical→source) + fixture fix · owner `src/broadway/samples/loader.py SampleSpec` + `configs/sample/taxi_diagnostic.yaml:5` · inputs: [column_mapping block, analysis group_column] → outputs: [validated direction or loud inversion error] · FINDING: shipped yaml writes `{Borough: pickup_borough}` (source→logical) while ALL three consumers read logical→source (`stats/module.py:42`, `describe.py:132`, `runners.py:100` use `column_mapping.get(group_column, default)`) ⇒ remap silently never fires at HEAD · validated_by: [tests/test_sample.py::test_load_sample_column_mapping_round_trip, tests/test_stats_module.py::test_stats_run_with_sample_column_mapping, tests/test_describe.py::test_describe_run_column_mapping] (tests pin logical→source; the SHIPPED YAML is what is inverted) · source: FINDING-D1 (verified against live code this session) · **RULING: LAND**

### Band 05-stats

- **GATE-STATS-113** ← FINDING E2E-D3 · Small-group reachability floor at sample scale (Staten-Island-unreachable) · owner `src/broadway/stats/guards.py:8 validate_groups()` + sample-size config · inputs: [sample parquet, declared group_values incl. Staten Island] → outputs: [loud unreachable-group verdict instead of vacuous comparison] · context: full-data SI n=84 (agents/notes/trust.md, LEARN.md:28); diagnostic samples cannot reach declared groups; walkthrough deliberately keeps size-0 arrays (STATS-42 finding) relying on guards · validated_by: [tests/test_walkthrough.py::test_run_describe_flags_imbalance_and_absent_groups] · source: FINDING-D3 · **RULING: LAND**

### Band 06-timeline-lineage

- **GATE-TLINE-114** ← FINDING E2E-D2 · Flow↔lineage profile-parent completeness gate · owner `src/broadway/lineage/state.py:5 EXPECTED` + `src/broadway/baseline/module.py:88 parents` · inputs: [flow step sequence] → outputs: [loud graph incompleteness when declared parent kind never ran] · FINDING: baseline declares parents ['analysis:taxi','profile:taxi'] (verified live in e2e record) and EXPECTED hypothesis/prediction/causal sequences all include "profile", yet flows run ingest→etl→baseline without ever invoking the profile step (profile ran separately at 22:45 vs baseline 22:38) · validated_by: [] · source: FINDING-D2 · **RULING: LAND**

### Band 02-etl-lookup

- **GATE-ETL-115** ← PROPOSE-GH2-06 · Orphan fetch-helper ruling: wire-or-delete download() before URL-controlled filenames enter raw_dir · owner `src/broadway/data/download.py:16 download()` · inputs: [arbitrary URL] → outputs: [`<raw_dir>/<URL-basename>` stream-written file feeding GATE-INGEST-01 glob if ever wired] · validated_by: [tests/test_download.py::test_download_writes_chunks_under_raw_dir] · **RULING: LAND**
- **GATE-ETL-116** ← PROPOSE-GH1-06 · generate_samples producer gate (immutable artifact + provenance sha256 pair) · owner `src/broadway/samples/generate.py generate_samples()` · inputs: [SampleSpec definition] → outputs: [`<name>@<version>.parquet` + provenance json {artifact_sha256, definition_sha256, row_count}] · validated_by: [tests/test_sample.py round-trip pins] · **RULING: LAND**
- **GATE-ETL-117** ← PROPOSE-GH1-10 · Legacy twin config-loader retirement/fork guard · owner `project/etl/process_config.py legacy twin config path` · inputs: [configs/project/taxi.yaml knobs] → outputs: [raw_dir/processed_dir/processed_file for LEGACY pipeline] · validated_by: [] · **RULING: LAND**

### Band 04-training-eval

- **GATE-TRAIN-118** ← PROPOSE-GH2-03 · Baseline step gate (subprocess git-provenance + artifact writer) · owner `src/broadway/baseline/module.py:45 run()` · inputs: [PipelineConfig baseline section, dataset frame] → outputs: [BaselineResult JSON :82-92, lineage node_id("baseline") :92; commit provenance or silent "unknown" fallback :26-27] · validated_by: [] · **RULING: LAND**
- **GATE-TRAIN-119** ← PROPOSE-GH2-04 · Causal step design-semantics gate · owner `src/broadway/causal/module.py:18 run()` · inputs: [cfg.causal design params, canonical frame] → outputs: [causal design artifact save_design(:35), lineage record :37] · validated_by: [] · **RULING: LAND**
- **GATE-TRAIN-120** ← PROPOSE-GH3-05 · MLflow connectivity teeth: widen marker net + preflight precedent · owner `src/broadway/training/mlflow_utils.py setup_mlflow/connectivity markers` · inputs: [tracking URI] → outputs: [named RuntimeError with README hint :60-76] · validated_by: [tests/test_mlflow_utils_unreachable.py::test_unreachable_http_store_raises_clear_error] · **RULING: LAND**

### Band 09-infra-meta (entrypoints · containers · git/cache custody)

- **GATE-INFRA-121** ← PROPOSE-GH2-08 · qq demo `__main__` stray-write block · owner `src/broadway/discover/qq.py __main__ block` · inputs: [none — synthetic seeded frame] → outputs: [qq_demo_output/ + json into whatever CWD] · validated_by: [] · **RULING: REJECT-as-NOT-GATE** (remedy is deleting the demo block, not a gate)
- **GATE-INFRA-122** ← PROPOSE-GH1-05 ≡ PROPOSE-GH2-09 (**internal pool merge**: both own experiments.py dispatcher) · Root experiments CLI dispatcher custody · owner `experiments.py main argparse dispatcher (ols|diagnostics|qq_legend|verify)` · inputs: [argv subcommand] → outputs: [plots/results CSVs/verification JSON; self-auditing verify] · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-123** ← PROPOSE-GH2-10 · experiments_ui `__main__` uvicorn entry custody · owner `experiments_ui.py __main__ (uvicorn.run)` · inputs: [host/port argv] → outputs: [dashboard server] · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-124** ← PROPOSE-GH2-11 + stray-FINDING-C (**merged**) · lifecycle.sh command-surface gate incl. subcommand symmetry · owner `k8s/optuna/lifecycle.sh train|up|view|dump|down case dispatch` · inputs: [subcommand] → outputs: [cluster/state/view lifecycle actions] · FINDING absorbed: view_local backgrounds mlflow server `( cd "$ROOT" && uv run mlflow server … & )` at :205-208 with NO stop subcommand — cleanup is only an echoed hint `kill $(pgrep -f 'mlflow server.*…')` :215; add `stop` subcommand owning the child PID · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-125** ← PROPOSE-GH2-12 · optuna-init.yaml container command custody · owner `k8s/optuna/optuna-init.yaml command ["python",…]` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-126** ← PROPOSE-GH2-13 · Dockerfile.worker CMD custody · owner `Dockerfile.worker CMD ["python","/app/experiments…"]` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-127** ← PROPOSE-GH2-14 · Root Dockerfile bare `CMD ["ds-pipeline"]` custody · owner `Dockerfile CMD` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-128** ← PROPOSE-GH2-15 · compose mlflow server command custody · owner `docker-compose.yml command mlflow server --back…` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-129** ← PROPOSE-GH2-16 · experiments/** `__main__` lane (~60 runnable scripts) · owner `experiments/** __main__ lane` · validated_by: [] · **RULING: SPLIT-OWNER** (seniors carve teaching-surface vs model-battle vs optuna-worker sublanes)
- **GATE-INFRA-130** ← PROPOSE-GH4-01 (+ GH1-07 rider) · api-deployment image/command coherence · owner `k8s/api-deployment.yaml Deployment broadway-api` · validated_by: [] · **RULING: MERGE-WITH-DEPLOY-F1-API-LANE** (already routed in D-ops verdicts mapping #8/#9; avoid duplicate custody per packet-C #27 precedent; carries GH1-07 image-expectation rider)
- **GATE-INFRA-131** ← PROPOSE-GH4-02 · api HPA bounds custody · owner `k8s/api-deployment.yaml HorizontalPodAutoscaler` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-132** ← PROPOSE-GH4-03 (+ GH1-07 rider) · train-job Job template custody · owner `k8s/train-job.yaml Job train-{{ .Values.runId }}` · validated_by: [] · **RULING: MERGE-WITH-DEPLOY-F2-TRAINJOB-LANE** (#10/#18 routing; GH1-07 rider)
- **GATE-INFRA-133** ← PROPOSE-GH4-04 · postgres StatefulSet custody · owner `k8s/postgres-deployment.yaml StatefulSet postgr…` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-134** ← PROPOSE-GH4-05 · ABSENT-OBJECT manifest gate (demanded-by exists, declaring manifest does not) · owner `ABSENT-OBJECT — no manifest in the repo declare…` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-135** ← PROPOSE-GH4-06 · mlflow-deployment name mismatch (`broadway-mlflow` demanded, CI builds `mlflow-server`) · owner `k8s/mlflow-deployment.yaml Deployment mlflow` · validated_by: [] · **RULING: MERGE-WITH-IMAGE-TAG-COHERENCE** (#15/#16/#17 routing)
- **GATE-INFRA-136** ← PROPOSE-GH1-09 ≡ PROPOSE-GH4-07 (**merge**: same owner path root ./Dockerfile) · Root multi-stage builder ↔ unbuilt `broadway` image economy · owner `Dockerfile multi-stage build (uv sync --frozen…)` · validated_by: [] · **RULING: MERGE-WITH-CI-BUILD-BROADWAY-LATEST** (#19 routing: `broadway` demanded by train-job/api but built by NO workflow)
- **GATE-INFRA-137** ← PROPOSE-GH1-08 ≡ PROPOSE-GH4-08 (**merge**: compose services/build-context overlap) · compose-stack build/volume law (postgres broken context M7 fact rides here) · owner `docker-compose.yml services mlflow/postgres (bu…` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-138** ← PROPOSE-GH1-02 · conftest snapshot-dir hygiene gate · owner `tests/conftest.py _SNAPSHOT_DIRS snapshot hygie…` · validated_by: [] · **RULING: LAND**
- **GATE-INFRA-139** ← PROPOSE-GH1-03 · uv editable-rebuild probe guard formalization · owner `tests/test_uv_probe_guard.py uv editable-rebuil…` · validated_by: [tests/test_uv_probe_guard.py suite] · **RULING: LAND**
- **GATE-INFRA-140** ← PROPOSE-GH1-11 · ci-fixtures k8s-config orchestrator-fixture custody · owner `.github/ci-fixtures/k8s-config.yaml orchestrato…` · validated_by: [] · **RULING: DEFER-to-K8S-CONFIGMAP-ENV lane** (fixture-side rider of the routed configmap-env lane)
- **GATE-INFRA-141** ← PROPOSE-GH6-01 · Git-tag minting custody (`tier-1-complete` local-only, credential-bearing, no minter law) · owner `git tag -a manual path` (no scripted minter exists) · validated_by: [] · source: gap-object-creators Plane 1.1 · **RULING: LAND**
- **GATE-INFRA-142** ← PROPOSE-GH6-02 · Stale-ref retirement law (pr-1/pr-2 closed-unmerged, credential-bearing snapshots) · owner `refs/heads/pr-1, pr-2` · validated_by: [] · source: Plane 1.2–1.3 · **RULING: LAND**
- **GATE-INFRA-143** ← PROPOSE-GH6-03 · CI cache retention/dedupe law (19 keys ≈10.39 GiB AT LRU ceiling; setup-uv key saved twice) · owner `.github/workflows/ci.yml:145 actions/cache@v4` · validated_by: [] · source: Plane 2.3 · **RULING: LAND**
- **GATE-INFRA-144** ← PROPOSE-GH6-04 · GHCR registry-write ownership + version cap (state honestly UNVERIFIABLE: token lacks read:packages, HTTP 403/404 recorded) · owner `ci.yml CD job Push verified images to GHCR (:291-301)` · validated_by: [] · source: Plane 2.4 · **RULING: LAND**
- **GATE-INFRA-145** ← PROPOSE-GH6-05 · Container-image retention law (teardown.sh rm's cluster but never rmi's 6 × ~4.61 GB images) · owner `k8s/optuna/teardown.sh` + docker rmi sites · validated_by: [] · source: Plane 3.1 · **RULING: LAND**
- **GATE-INFRA-146** ← PROPOSE-GH6-06 · Cache-root fork law — ship.sh:10 `export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"` forks an 11.45 G second cache root · owner `scripts/ship.sh:10 UV_CACHE_DIR/MPLCONFIGDIR exports` · validated_by: [] · source: Plane 4.2 · **RULING: LAND**

### Band 81-object-custody (NEW — justification §0)

- **GATE-CUST-147** ← PROPOSE-GH5-01 · Champion-alias single-writer custody — alias writes originate solely from the promotion chokepoint (import-lint on set_registered_model_alias/MlflowClient outside training.mlflow_utils + registry audit trail) · owner `src/broadway/training/mlflow_utils.py:162 promote_candidate()` · inputs: [model_uri, alias="champion"] → outputs: [registered version + alias move; elsewhere loud refusal] · context: worker pods hold full tracking reachability (configmap `allowed_hosts: "*"`); check_champion_manifest.sh declines to be a gate · validated_by: [adjacent pins: tests/test_evaluate_contracts.py::test_module_run_promotes_only_after_persistence] · **RULING: LAND**
- **GATE-CUST-148** ← PROPOSE-GH5-02 · Registered-model-name pin (= cfg.dataset.name / configmap dataset.name) — ad-hoc names would silently fork the champion namespace scanned by get_champion/list_champions · owner `mlflow_utils.py register_model call sites` · validated_by: [] · **RULING: LAND**
- **GATE-CUST-149** ← PROPOSE-GH5-03 · Run-tag closed vocabulary {model, study, seed} across the five independent writer sites · owner `hpo.py _mlflow_callback tags + experiments/mlflow writers` · validated_by: [] · **RULING: LAND**
- **GATE-CUST-150** ← PROPOSE-GH5-04 · Experiment-name pin (lawful namespaces: ratecode1_model_battle family + dataset-name experiments); setup_mlflow refuses orphans · owner `mlflow_utils.py:54 setup_mlflow()` · validated_by: [] · **RULING: LAND**
- **GATE-CUST-151** ← PROPOSE-GH5-05 · Artifact-retention/lifecycle law (0 deletion calls repo-wide; versions + lifecycle.sh tarballs grow unbounded) · owner `mlflow registry prune policy + k8s/optuna/lifecycle.sh BACKUP_DIR generations` · validated_by: [] · **RULING: LAND**
- **GATE-CUST-152** ← PROPOSE-GH5-06 · Run-name convention gate (optuna_{model}, {study} trial {n}, battle names) so determinism-diff and champion manifest can rely on run identity · owner `run-name emit sites (sites 6/9/10 of census B1)` · validated_by: [] · **RULING: LAND**
- **GATE-CUST-153** ← stray proposal (mission text; NO in-repo record found — flagged) · tracked_run chokepoint wrapper + append-only object-ledger.jsonl (human-endorsed direction): subprocess chokepoint wrapper around mlflow/optuna/git-tag/cache-root object creators; context-managed planes; constructor wrappers (create_study / git tag / cache roots); every created object appended to object-ledger.jsonl · owner `new wrapper layer over training/optuna.py create_study, mlflow_utils constructors, lifecycle.sh, ship.sh cache exports` · inputs: [constructor calls] → outputs: [ledger jsonl append per object + wrapper-scoped teardown] · validated_by: [] · **RULING: LAND** (design-direction ratification requested)

---

## §3 · BAND MAP (zero collisions; dense-order invariants intact)

| id range | band | slots | count |
|---|---|---|---|
| GATE-SURF-100..102 | 07-surfaces | 100,101,102 | 3 |
| GATE-CFG-103..112 | 08-config-schema | 103–112 | 10 |
| GATE-STATS-113 | 05-stats | 113 | 1 |
| GATE-TLINE-114 | 06-timeline-lineage | 114 | 1 |
| GATE-ETL-115..117 | 02-etl-lookup | 115–117 | 3 |
| GATE-TRAIN-118..120 | 04-training-eval | 118–120 | 3 |
| GATE-INFRA-121..146 | 09-infra-meta | 121–146 | 26 |
| GATE-CUST-147..153 | 81-object-custody (NEW) | 147–153 | 7 |

Invariants: numeric order globally unique and gapless 1…99,100…153 · no slot reused from exhausted ranges 01–09/80–89 · new-band number 81 chosen OUTSIDE the 01–09 fragment range to avoid clashing with the `NN-` prefix convention of existing fragments (80 taken by hpo-optuna) · future gates append ≥154 regardless of band.

---

## §4 · DEPENDENCY ORDER (land-first sequence)

1. **Dupe-registry PRECISE-SPLIT owner-string repairs (§1 O1–O9, N1)** land FIRST — id stability before any new ids attach to those bands.
2. **GATE-CFG-103 (STEP_MODULES dispatch)** before ALL other step-entry gates (105, 106, 114, 118, 119) and before any HPO suggest-dispatch refinements reference step dispatch.
3. **GATE-CFG-112 (D1 column-mapping direction law)** before sample-consuming rulings (113 stats floor; walkthrough lanes) and before worker/HPO gates that read samples.
4. **GATE-TLINE-114 (profile-parent completeness)** before worker-plane/HPO gates assert lineage/graph completeness.
5. **Band-81 custody core 147→148→149** before worker-plane infra gates (125, 126) — custody defines what worker pods may write before their entrypoints are gated.
6. **GATE-TRAIN-120 connectivity teeth** before GATE-INFRA-135 mlflow-deployment boot checks.
7. **Image economy 136/137** before **144 (GHCR version cap)** — capping publishes presumes the built images exist.
8. **GATE-CUST-153 (object-ledger wrapper)** LAST among custody items — it wraps constructors whose laws 147–152 define.

---

## §5 · COUNTS SUMMARY + FLAGS

**Pool intake:** 42 parsed proposals (GH1×11, GH2×16, GH3×7, GH4×8) + 12 unparsed (GH5×6, GH6×6) + 5 strays (D1, D2, D3, lifecycle-view-stop, tracked_run/object-ledger) = **59 items**.

**Merges applied:** GH1-05≡GH2-09 (same dispatcher owner) · GH1-09≡GH4-07 (same root Dockerfile) · GH1-08≡GH4-08 (compose overlap) · GH1-07 → riders on 130/132 · stray-lifecycle → rider on 124 ⇒ 59 items → **54 proposed gate ids**.

**Per-disposition tally (ruling requests):** LAND **46** · MERGE-WITH **5** (106→CFG-75; 130→DEPLOY-F1-API-LANE; 132→DEPLOY-F2-TRAINJOB-LANE; 135→IMAGE-TAG-COHERENCE; 136→CI-BUILD-BROADWAY-LATEST) · SPLIT-OWNER **1** (129) · REJECT-as-NOT-GATE **1** (121) · DEFER-to-lane **1** (140→K8S-CONFIGMAP-ENV). Sum 54 ✓.

**Per-band tally:** SURF 3 · CFG 10 · STATS 1 · TLINE 1 · ETL 3 · TRAIN 3 · INFRA 26 · CUST(new) 7 = 54 ✓.

**Two-owner arbitration tally (§1):** MERGE-rows 0 · PRECISE-SPLIT 7 (O1,O2,O3,O4,O6,O8,O9 + N1 counted below) · CO-TENANCY-LEGAL 12 (O5,O7,W1–W9) — precisely: 9 owner-collisions → 7 PRECISE-SPLIT + 2 CO-TENANCY-LEGAL (O5,O7); 9 two-writer artifacts → 9 CO-TENANCY-LEGAL; 1 near-identical → 1 PRECISE-SPLIT. Total 19 groups resolved ✓.

**Honestly flagged (could not fully classify):**
1. **E2E D4/D5** — only D1–D3 were recoverable from in-repo records + code verification; `/tmp/broadway-e2e/MANIFEST-run1.sha256` contains sha256 lines only (no findings notes). D4/D5 slots remain OPEN; seniors should re-source them from the session that ran the base-run.
2. **GATE-CUST-153 (tracked_run/object-ledger)** — zero in-repo citations exist (repo grep: only "chokepoint" appears, in gap-cd-mlflow.md); sourced solely from the human-endorsed mission text.
3. **GATE-INFRA-134 (PROPOSE-GH4-05 ABSENT-OBJECT)** — carried at title level; the gap-foreign sheet's full detail block was not re-derived this session.
4. Working tree carries foreign WIP (deleted src/broadway/trust/*, modified process.py etc.) — slate written strictly against HEAD 5016e93 registry content (gates.yaml meta.head matches).
