# DIGEST.md — rendered from agents/ledger/gates.yaml · NEVER HAND-EDIT ·

> 165 gates · rendered 2026-09-02 @ HEAD b9100f3 · load THIS into context;
> gates.yaml is the sole SSOT; the retired GATES.md/gates/*.md markdown world survives in agents/ledger/arbitration/2026-08-24/surface-and-analysis-preservation.md.

| band | phase | gates | findings |
|---|---|---|---|
| 01-ingest | ingest | 8 | 8 |
| 02-etl-lookup | etl-lookup | 9 | 5 |
| 03-features | features | 10 | 4 |
| 04-training | training-eval | 12 | 11 |
| 05-stats | stats | 11 | 7 |
| 06-timeline | timeline-lineage | 11 | 7 |
| 07-surfaces | surfaces | 50 | 1 |
| 08-config | config-schema | 15 | 10 |
| 80-hpo-optuna | hpo-optuna | 10 | 8 |
| 09-infra | infra-meta | 26 | 14 |
| 81-object-custody | object-custody | 3 | 0 |
| **total** | | **165** | **75** |

### 01-ingest — ingest

- **GATE-INGEST-02** `src/broadway/data/loader.py:124 load_with_audit()` ⚠FINDING
  [CFG-DATASET-CONTRACT, ARTIFACT-RAW-PARQUET] → [ARTIFACT-RAW-FRAME] · pins: 4
- **GATE-INGEST-03** `src/broadway/etl/module.py:74 run() CI-sample guard` ⚠FINDING
  [ARTIFACT-RAW-FRAME, CFG-ETL-STEP] → [ARTIFACT-RAW-FRAME] · pins: 1
- **GATE-INGEST-04** `project/etl/process.py: select_and_clean_columns() projection/missing/extra/dropna aspects (numeric anchors stripped pending DP-A07/R2 StageLedger merge)` ⚠FINDING
  [CFG-DATASET-CONTRACT, ARTIFACT-RAW-FRAME] → [ARTIFACT-SELECTED-FRAME] · pins: 3
- **GATE-INGEST-05** `src/broadway/cleaning/structural.py:49 parse_datetime()` ⚠FINDING
  [CFG-DATASET-CONTRACT, ARTIFACT-RAW-FRAME] → [ARTIFACT-CANONICAL-FRAME, ARTIFACT-CLEAN-EVIDENCE] · pins: 2
- **GATE-INGEST-06** `src/broadway/cleaning/structural.py:83 parse_numeric()` ⚠FINDING
  [CFG-DATASET-CONTRACT, ARTIFACT-RAW-FRAME] → [ARTIFACT-CANONICAL-FRAME, ARTIFACT-COERCION-AUDIT, ARTIFACT-CLEAN-EVIDENCE] · pins: 7
- **GATE-INGEST-07** `src/broadway/etl/module.py:129 run() evidence surfacing` ⚠FINDING
  [ARTIFACT-CLEAN-EVIDENCE, ARTIFACT-COERCION-AUDIT] → [ARTIFACT-CLEAN-EVIDENCE, ARTIFACT-COERCION-AUDIT] · pins: 2
- **GATE-INGEST-08** `src/broadway/data/cleaner.py:34 canonicalize()` ⚠FINDING
  [CFG-DATASET-CONTRACT, CFG-ETL-STEP, ARTIFACT-RAW-FRAME] → [ARTIFACT-CANONICAL-FRAME, ARTIFACT-CLEAN-EVIDENCE] · pins: 5
- **GATE-INGEST-09** `src/broadway/contracts/pandera.py:46 build_raw_schema()` ⚠FINDING
  [CFG-DATASET-CONTRACT, ARTIFACT-CANONICAL-FRAME] → [ARTIFACT-CANONICAL-FRAME] · pins: 3

### 02-etl-lookup — etl-lookup

- **GATE-ETL-10** `src/broadway/etl/module.py:61 run()`
  [PipelineConfig (cfg.dataset + cfg.etl + cfg.experiment), raw dataset at DatasetContract.path via load_with_audit] → [<processed_subdir>/<name>_canonical.parquet (ARTIFACT-CANONICAL-PARQUET), <name>_clean.json (ARTIFACT-STRUCTURAL-CLEAN), audit JSONs (GATE-ETL-13/14/15), lineage records node_id("etl")] · pins: 7
- **GATE-ETL-11** `src/broadway/data/loader.py:124 load_with_audit()` ⚠FINDING
  [DatasetContract.path (csv/parquet/xlsx via READERS dict keyed on suffix), dataset.lookup_tables CSVs read with keep_default_na=False + authored na_values] → [(df_merged, list[JoinAudit], list[LookupValueAudit]) — left-join frame with lookup columns renamed by _lookup suffix rule] · pins: 5
- **GATE-ETL-12** `src/broadway/data/loader.py:46 _assert_unique_merged_labels()`
  [DatasetContract, pre-merge df.columns, merged_names from merged_lookup_column_names(), left_key/right_key] → [SchemaError naming every duplicated label + provenance (raises before merge), or silently returns when produced label list is unique] · pins: 2
- **GATE-ETL-13** `src/broadway/data/join_audit.py:27 audit_join()`
  [left df, left_key column, LookupSpec (path/key), lookup_df] → [JoinAudit{rows_attempted, matched, unmatched, null_keys, unmatched_rate}] · pins: 5
- **GATE-ETL-14** `src/broadway/data/lookup_value_audit.py:33 audit_lookup_values()` ⚠FINDING
  [post-merge df, left_key, LookupSpec + value_policies sentinels, lookup_df, merged_names, matched count from JoinAudit] → [LookupValueAudit → LookupColumnValueAudit per non-key lookup column {null_count, sentinel_counts, affected_rows, affected_rate, affected_lookup_keys}; persisted as <name>_lookup_value_audit.json (ARTIFACT-LOOKUP-VALUE-AUDIT) + lineage node_id('lookup_value') parented under node_id('join') at src/broadway/etl/module.py:163-167] · pins: 5
- **GATE-ETL-15** `src/broadway/cleaning/structural.py:83 parse_numeric()` ⚠FINDING
  [pd.Series, declared target_dtype, optional caller-supplied coercions collector (CoercionRecord list threaded from etl.run through canonicalize/data.cleaner — anchors de-lined per packet F §4)] → [(repaired series, ParseFailure | None); collector append CoercionRecord{column, declared_dtype, arriving_dtype, rows_affected} at :83-90 at HEAD; persisted by etl.run as <name>_coercion_audit.json (ARTIFACT-COERCION-AUDIT) + lineage node_id('coercion') at src/broadway/etl/module.py:145-156] · pins: 6
- **GATE-ETL-16** `src/broadway/data/loader.py:112 canonical_path()` ⚠FINDING
  [EnvironmentConfig.data_dir + EnvironmentConfig.processed_subdir (CFG-ENV-PROCESSED-SUBDIR), dataset.name; sibling writes keyed off EtlStep.train_file/val_file/training_data_file (CFG-ETL-SPLIT-FILES) at src/broadway/etl/module.py:104-109] → [<data_dir>/<processed_subdir>/<dataset.name>_canonical.parquet (ARTIFACT-CANONICAL-PARQUET), written index=False at module.py:98; plus split pair train/val or flat training_data parquet; out_dir mkdir(parents=True, exist_ok=True)] · pins: 3
- **GATE-ETL-17** `project/etl/process.py: select_and_clean_columns() ledger-accounting grammar aspect (narrowed ledger-append-sites owner BLOCKED pending DP-A07/R2 StageLedger merge — packet F #1b)` ⚠FINDING
  [renamed post-filter df, DatasetContract (contract.columns order IS the projection order), optional StageLedger param (list[tuple[str,int]]; WIP-pending, see FINDING)] → [contract-projected, dropna'd, deduplicated df; ledger appended with ("dropna", rows_after) then ("duplicates", rows_after) — line anchors stripped pending DP-A07/R2] · pins: 4
- **GATE-ETL-116** `src/broadway/samples/generate.py:106 generate_sample()`
  [SampleSpec definition + source file (.parquet or .csv)] → [<name>@<version>.parquet + provenance json {artifact_sha256, definition_sha256, row_count}] · pins: 2

### 03-features — features

- **GATE-FEAT-20** `src/broadway/features/builders.py:46 BUILDERS (log1p at builders.py:53)` ⚠FINDING
  [ARTIFACT-SPLIT (etl train/val parquet frame), CFG-FEATURES (features.derived[].func/source), DerivedFeature(name,func,source)] → [derived columns appended onto a copy of df (builders.py:118 result=df.copy(), :145 result[feat.name]=...), declared dtypes via _BUILDER_DTYPES (builders.py:62-73)] · pins: 4
- **GATE-FEAT-21** `src/broadway/features/builders.py:122 build_derived()  # F6-GUARD label-collision guard`
  [df (input frame), features.derived[] in config order, target (dataset target name), registry=BUILDERS∪extra_builders (builders.py:115-117)] → [ValueError raise sites — derived name == target ⇒ ValueError (builders.py:123-127); name ∈ result.columns ⇒ ValueError with origin 'input data column' vs 'previously built derived feature' (builders.py:133-138); unknown func ⇒ ValueError (builders.py:140); missing source ⇒ ValueError (builders.py:144)] · pins: 7
- **GATE-FEAT-22** `src/broadway/features/builders.py:76 load_custom_builders()`
  [CFG-FEATURES (features.builder_module str|None)] → [extra_builders dict merged over BUILDERS inside build_derived (builders.py:115-117); raise sites: unimportable module ⇒ ValueError (builders.py:82), module without dict BUILDERS ⇒ ValueError (builders.py:85), generic-name collision ⇒ ValueError naming collisions (builders.py:88-90)] · pins: 4
- **GATE-FEAT-23** `src/broadway/features/pipeline.py:21 FeaturePipeline.fit()` ⚠FINDING
  [ARTIFACT-SPLIT train frame (loaded at src/broadway/features/module.py:25), cfg.dataset.target, cfg.features.encoding_smoothing, cfg.experiment.features.encodings (pipeline ctor arg, module.py:35)] → [fitted encoder state _target_encoders/_freq_encoders (pipeline.py:22-23 reset then append :27-31) persisted later into ARTIFACT-PIPELINE-PICKLE (module.py:52-54)] · pins: 3
- **GATE-FEAT-24** `src/broadway/features/pipeline.py:34 FeaturePipeline.transform()`
  [train/val frame, cfg.experiment.features (FeatureConfig), target, freq_fill (cfg.features.frequency_fill), builder_kwargs from cfg.features.builder_params (pipeline.py:37-41)] → [engineered frame: input copy + derived cols + <col>_freq_enc float64 + <col>_target_enc float64] · pins: 6
- **GATE-FEAT-25** `src/broadway/features/module.py:45 run()  # ARTIFACT-TRAIN-PARQUET write` ⚠FINDING
  [ARTIFACT-SPLIT train parquet (module.py:25), optional val parquet (:27), CFG-FEATURES/DATASET/EXPERIMENT/ETL precondition raise (module.py:32-33), encoding_smoothing/frequency_fill/max_drop_fraction] → [ARTIFACT-TRAIN-PARQUET out_dir/cfg.etl.train_features_file index=False (module.py:45), ARTIFACT-VAL-PARQUET val_features_file (:50), ARTIFACT-PIPELINE-PICKLE pickle.dump (:52-54), lineage record node_id("features", dataset) with TransformAudit rows/columns_before/after/added/removed (module.py:57-69)] · pins: 3
- **GATE-FEAT-26** `src/broadway/features/generic.py:14 build_generic_feature_specs()`
  [DatasetContract.columns (declared dtypes/null_count), CFG-FEATURES include/derived/encodings, builder_dtype→_BUILDER_DTYPES (builders.py:94-95)] → [ordered FeatureSpec registry — SSOT for both write-side (GATE-FEAT-25 :40-41) and read-side (GATE-FEAT-27 engineered_schema_for :49) schemas] · pins: 3
- **GATE-FEAT-27** `src/broadway/features/generic.py:122 validate_engineered_frame()  # the contract's 'validate_engineered_schema'` ⚠FINDING
  [ARTIFACT-TRAIN-PARQUET / ARTIFACT-VAL-PARQUET frames read back (training/module.py:42-44, evaluate/module.py:49,58), PipelineConfig (dataset+experiment)] → [SchemaError raise chain: ordered-schema violation incl. SchemaErrors→SchemaError normalization (generic.py:131-138); extra-column guard FIX_4 G2 declared=contract∪joined_lookup∪specs (generic.py:91-119, raise :110); explicit target-dtype hook (generic.py:59-88, raises :69 missing-target and :79 dtype-mismatch)] · pins: 4
- **GATE-FEAT-28** `src/broadway/utils.py:16 eligible_feature_columns()`
  [engineered frame (post GATE-FEAT-27 validation), PipelineConfig.experiment.data_source.schema_contract, PipelineConfig.dataset.target, PipelineConfig.experiment.preprocessing claims (utils.py:32)] → [model-input matrix X = declared surface ∩ frame − target, FRAME ORDER PRESERVED not sorted (utils.py:33,42 docstring :23); raise sites: missing experiment/dataset ⇒ ValueError (utils.py:26); eligible categorical neither preprocessing-claimed nor numeric ⇒ ValueError with exact pinned message (utils.py:37-41)] · pins: 12
- **GATE-FEAT-29** `src/broadway/schemas/__init__.py:17 _engineered_schema_columns()  # dispatch at schemas/__init__.py:41 schema_columns()`
  [cfg.experiment.data_source.schema_contract string, DatasetContract, FeatureConfig|None] → [frozenset declared surface feeding GATE-FEAT-28 (utils.py:27-31) and recipe.validate_preprocessing_columns binding check (recipe.py:129-133); raise sites: 'engineered' without features config ⇒ ValueError (schemas/__init__.py:25-30); unknown schema_contract ⇒ ValueError listing supported modules (schemas/__init__.py:53-57)] · pins: 6

### 04-training — training-eval

- **GATE-TRAIN-30** `src/broadway/training/module.py:89 run()` ⚠FINDING
  [CFG-DATASET-CONTRACT, CFG-STEP-TRAIN, experiment+etl config sections, analysis contract] → [ARTIFACT-TRAINING-RESULT, lineage node training:<dataset> (:153-158)] · pins: 4
- **GATE-TRAIN-31** `src/broadway/training/module.py:39 _load_features()` ⚠FINDING
  [ARTIFACT-CANONICAL-FRAME derivatives: etl train_features_file parquet, optional val_features_file parquet] → [train_df, val_df (validated engineered frames)] · pins: 7
- **GATE-TRAIN-33** `src/broadway/training/module.py:59 _resolve_params()` ⚠FINDING
  [CFG-STEP-TRAIN.model.params, optional HPO block (models/search_space/total_trials/top_k/storage_url), X/y train+val from GATE-TRAIN-31] → [final model param dict fed to GATE-TRAIN-34] · pins: 8
- **GATE-TRAIN-34** `src/broadway/training/trainer.py:17 build_model_pipeline()` ⚠FINDING
  [PipelineConfig with experiment block, model_type registry key, params dict (bare + pre__ prefixed)] → [fitted sklearn Pipeline + TrainingResult{train_time_seconds} (via train() :52-69)] · pins: 6
- **GATE-TRAIN-35** `src/broadway/evaluate/module.py:82 run()` ⚠FINDING
  [CFG-STEP-EVALUATE, val_features_file parquet (REQUIRED), train_features_file parquet, ARTIFACT-TRAINING-RESULT, champion via models:/<dataset>@champion] → [ARTIFACT-EVALUATION-RESULT, lineage node evaluation:<dataset> (:166-171), conditional champion promotion (:181-194)] · pins: 5
- **GATE-TRAIN-36** `src/broadway/evaluate/metrics.py:23 compute_metrics()` ⚠FINDING
  [y_true/y_pred arrays from holdout predict, CV folds via cross_validate (src/broadway/evaluate/validation.py:42)] → [rounded metric dicts persisted in ARTIFACT-EVALUATION-RESULT / ARTIFACT-MLFLOW-RUN] · pins: 9
- **GATE-TRAIN-37** `src/broadway/evaluate/promotion.py:6 should_promote()` ⚠FINDING
  [candidate_metrics[target_metric], champion_score (None when no champion), CFG-STEP-EVALUATE.promotion_threshold (configs/step/evaluate.yaml:2 → 0.05)] → [(promote: bool, reason: str) recorded in ARTIFACT-EVALUATION-RESULT; champion alias move on promote] · pins: 6
- **GATE-TRAIN-38** `src/broadway/stats/robust.py:34 estimation_table()` ⚠FINDING
  [any FITTED statsmodels regression results object, alpha (default 0.05)] → [coef/HC3_SE/CI_low/CI_high DataFrame] · pins: 3
- **GATE-TRAIN-39** `src/broadway/training/mlflow_utils.py:195 list_champions()` ⚠FINDING
  [tracking_uri (file store auto-flagged via MLFLOW_ALLOW_FILE_STORE :202-204), alias default 'champion', logged runs/artifacts from the training path] → [ChampionArtifact records bucketed bare_model / pipeline_signature / ambiguous (:21-23), manifest report + retirement verdict from scripts/check_champion_manifest.sh] · pins: 11
- **GATE-TRAIN-118** `src/broadway/baseline/module.py:75 run() (+ _git_commit() :20-27 subprocess-provenance leg)` ⚠FINDING
  [PipelineConfig baseline section + dataset frame] → [BaselineResult JSON saved :85 (save_result, out_path :84) + lineage node_id("baseline") with parents :87-91] · pins: none direct
- **GATE-TRAIN-119** `src/broadway/causal/module.py:18 run()`
  [cfg.causal design params + canonical frame] → [causal design artifact save_design(:35) + lineage record node_id("causal") with parents [baseline, analysis] (:37-42)] · pins: none direct
- **GATE-TRAIN-120** `src/broadway/training/mlflow_utils.py:47-51 _CONNECTION_REFUSED_MARKERS/_is_unreachable_http_store connectivity-marker net` ⚠FINDING
  [tracking URI] → [named RuntimeError with README hint (:63, :75-76)] · pins: 1

### 05-stats — stats

- **GATE-STATS-40** `src/broadway/stats/module.py:24 run()`
  [PipelineConfig (optional sections cfg.dataset, cfg.stats; required cfg.analysis), AnalysisMode contract] → [ValueError "stats step requires dataset and stats config" (:24-25), AnalysisMode.HYPOTHESIS enforcement via require_mode (:26), ValueError "hypothesis mode requires a 'hypothesis' block (group_column, group_values)" (:28-29)] · pins: 2
- **GATE-STATS-41** `src/broadway/stats/module.py:31 run() data leg` ⚠FINDING
  [canonical parquet at canonical_path(cfg.dataset, cfg.environment) when sample is None; else SampleSpec.path (+ sample.column_mapping logical→source remap of group_column :41-42)] → [FileNotFoundError "canonical dataset not found: … — run the etl step first" (:32-33), FileNotFoundError "sample dataset not found: …" (:37-38), ValueError "group column '<g>' not found in data" (:43-44), ARTIFACT-STATS-PLAN written via save_plan (:58-61 → src/broadway/stats/plan.py:31-32 write_text), lineage node_id("stats", analysis.name) parents [node_id("baseline",…), node_id("analysis",…)] (:66-73)] · pins: 6
- **GATE-STATS-42** `src/broadway/stats/groups.py:9 build_declared_groups()` ⚠FINDING
  [df, source_group_column, declared ordered group_values, target column name] → [(dict[str, np.ndarray] containing EVERY declared value in declared order, sorted absent list); a value with no matching rows OR all-NaN target maps to size-0 array (:28-31), absent = sorted size-0 names (:32)] · pins: 4
- **GATE-STATS-43** `src/broadway/stats/module.py:50 run() floor binding` ⚠FINDING
  [groups from GATE-STATS-42, cfg.stats.min_rows_for_sampling] → [small_group_threshold argument into run_anova; surfaces as AnalysisPlan.threshold_context.any_small_group + imbalance_ratio (src/broadway/stats/anova.py:60, 68-71) and warning "underpowered: small group(s)" when ANY group < floor] · pins: 2
- **GATE-STATS-44** `src/broadway/stats/guards.py:8 validate_groups()`
  [dict[str, np.ndarray] group arrays] → [list[str] of per-group zero-variance warnings (non-fatal); ValueErrors: "at least two groups required, got N" (:9-10), "group '<name>' is empty" (:15-16), "group '<name>' contains non-finite values" (:17-18), "group '<name>' has fewer than 2 observations" (:19-20), "all groups have zero variance — no variation to compare" (:26-27)] · pins: 6
- **GATE-STATS-45** `src/broadway/stats/anova.py:79 run_anova() family (run_welch :121, run_kruskal :164)` ⚠FINDING
  [groups, alpha (default 0.05 — callers: NONE bind it at stats step, walkthrough binds significance_alpha), small_group_threshold (default 30 — bound ONLY by module.py:50), sizes via _group_sizes (:18-19)] → [AnalysisPlan{statistics.statistic/p_value, effect_sizes eta_squared+omega_squared (anova/welch) or epsilon_squared (kruskal), threshold_context.imbalance_ratio/any_small_group, passed=bool(p<alpha), next_step="posthoc" if passed} (_build_plan :45-76)] · pins: 8
- **GATE-STATS-46** `src/broadway/stats/assumptions.py:19 check_normality() (companion run_levene :11)` ⚠FINDING
  [groups; shapiro_max_n default 5000 (:20); run_levene(groups) plain] → [per-group {skew, kurtosis, shapiro_p} (:31-35); levene {statistic, p_value} (:15-16); ValueError "normality checks require non-constant groups" (:23-24); ValueError "Levene's test requires non-zero variance in every group" (:13-14)] · pins: 4
- **GATE-STATS-47** `src/broadway/stats/post_hoc.py:11 games_howell()`
  [long df, dv, between, small_n default 30 (:11); SOLE production caller timeline/runners.py:396 passes dv/between only] → [pingouin pairwise_gameshowell frame augmented with cohens_d / hedges_g / effect_size_note columns (:29-31)] · pins: 3
- **GATE-STATS-48** `src/broadway/stats/robust.py:47 estimation_table() (regression/diagnostics/time_series/baseline cluster)` ⚠FINDING
  [fitted statsmodels results object exposing get_robustcov_results, alpha default 0.05 (:34); siblings: outlier_mask(threshold) :17, winsorize(cap_quantile) :25, modified_zscore MAD-guard :12-13, fit_ols/formula + fit_robust(cov_type="HC3") (regression.py:11-16), bp_test/jb_test/durbin_watson wrappers (diagnostics.py:20-31), durbin_watson_test/plot_acf(lags) (time_series.py:13-21), train_lgbm(**params)/evaluate(tail_quantile) (baseline.py:11-23)] → [TypeError "estimation_table requires a fitted statsmodels regression results object exposing get_robustcov_results…" (:48-53) — the ONLY raising guard in this cluster; coef/HC3_SE/CI_low/CI_high table derived by INTERNAL get_robustcov_results("HC3") re-fit (:54-56) so labels are truthful regardless of how the input was fitted (T-BUG-1 d056164, DECISIONS.md context :72); DiagnosticResult model (diagnostic_models.py:6-11); PNG artifacts via savefig (diagnostics.py:49, :84; time_series.py:21)] · pins: 10
- **GATE-STATS-49** `src/broadway/stats/describe.py:121 run()`
  [PipelineConfig (cfg.dataset/cfg.stats/cfg.analysis), SampleSpec REQUIRED (no canonical leg, unlike GATE-STATS-41), sample.column_mapping logical→source resolution (:132), cfg.dataset.target] → [ValueError "stats describe requires dataset and stats config" (:122-123), require_mode HYPOTHESIS (:124), hypothesis-block ValueError (:125-126), FileNotFoundError "sample dataset not found" (:127-129), ValueError "group column '<g>' not found in sample data" (:134-135), ARTIFACT-DESCRIBE-SUMMARY describe.json via model_dump_json (:137-140), lineage node_id("describe", analysis.name) parents [node_id("etl",…), node_id("analysis",…)] (:143-147), group figures PNG (:98-118, savefig :117)] · pins: 7
- **GATE-STATS-113** `src/broadway/stats/guards.py:8 validate_groups() + sample-size floor config` ⚠FINDING
  [sample parquet + declared group_values incl Staten Island] → [loud unreachable-group verdict instead of vacuous comparison] · pins: 1

### 06-timeline — timeline-lineage

- **GATE-TLINE-50** `src/broadway/timeline/models.py:31 AnalysisStep()`
  [step_id/order/question/kind from configs/flow/hypothesis_walkthrough.yaml, runner-computed result_summary dicts, StepStatus enum (models.py:9), FigureRef (models.py:24), runners.now_iso() timestamp] → [one validated AnalysisStep record per analysis step; serialized by timeline/module.save_step to artifacts/timeline/<analysis>/steps/<step_id>.json; consumed by suggest.py, reports/results.py:244 write_results(), reports/timeline.py:16 render_timeline(), reports/index.py:61 render_dashboard()] · pins: 5
- **GATE-TLINE-51** `src/broadway/timeline/models.py:50 AnalysisDecision()`
  [kind restricted to Literal["omnibus","posthoc"], status pinned to Literal["resolved"], method string from configs/step/walkthrough.yaml decisions.<kind>.methods, reason list, parents list from walkthrough.yaml, decided_at ISO stamp] → [AnalysisDecision record-of-record; persisted as artifacts/timeline/<analysis>/decisions/<id>.json via timeline/module.save_decision; read back by walkthrough._resolved_decision (walkthrough.py:64-68), _warn_stale_decisions (walkthrough.py:192-201), suggest.suggest_next (suggest.py:312 decided_ids), all three report renderers] · pins: 3
- **GATE-TLINE-52** `src/broadway/timeline/decide.py:21 record()` ⚠FINDING
  [analysis.name from AnalysisContract (hypothesis mode enforced upstream at cli.py:182 require_mode), kind, method, reason from `ds-pipeline decide` args, allowed-method vocabulary from configs/step/walkthrough.yaml:7-13 via sequence.load_walkthrough_config()] → [validated AnalysisDecision ready for module.save_decision; ValueError on unknown kind (decide.py:28-29) or out-of-vocabulary method (decide.py:30-34)] · pins: 4
- **GATE-TLINE-53** `src/broadway/timeline/module.py:19 save_step()` ⚠FINDING
  [validated AnalysisStep / AnalysisDecision objects, BROADWAY_TIMELINE_DIR env override (module.py:8, default artifacts/timeline)] → [artifacts/timeline/<analysis>/steps/<step_id>.json and .../decisions/<decision.id>.json (model_dump_json indent=2); read back by load_step/load_steps/load_decision/load_decisions which feed every gate check, suggestion pass, and renderer] · pins: 3
- **GATE-TLINE-54** `src/broadway/timeline/walkthrough.py:291 run()`
  [PipelineConfig (stats step, hypothesis-mode enforced walkthrough.py:292-296), WalkthroughSequence + WalkthroughConfig (sequence.py loads), frame/groups from runners.load_frame_and_groups (runners.py:86), persisted steps/decisions, --force flag] → [per-order executor dispatch through _STEP_RUNNERS (walkthrough.py:281-288); step JSONs saved (walkthrough.py:384); DECISION REQUIRED console panels (walkthrough.py:138-189); reports regenerated via _write_timeline (walkthrough.py:71-91 -> reports/timeline.md, reports/index.md, reports/results/*); stale-decision warnings (walkthrough.py:192-201)] · pins: 8
- **GATE-TLINE-55** `src/broadway/timeline/runners.py:302 run_omnibus()` ⚠FINDING
  [groups dict[str,np.ndarray] built by build_declared_groups (runners.py:107), decision.method from the resolved AnalysisDecision, alpha from walkthrough.yaml significance_alpha, posthoc: full df + source_group_column + target] → [omnibus/posthoc/conclusion AnalysisStep records; effect-size ramification text; artifacts/timeline/<analysis>/evidence/omnibus.json (runners.py:324) and posthoc.json (runners.py:413); conclusion consumes prior steps' result_summary (runners.py:462-512) rather than recomputing] · pins: 7
- **GATE-TLINE-56** `src/broadway/timeline/runners.py:114 run_describe()` ⚠FINDING
  [runner-local evidence models from timeline/evidence.py (NormalityEvidence evidence.py:14, VarianceEvidence evidence.py:22, PosthocEvidence evidence.py:40, ConclusionEvidence evidence.py:48), GroupSummary from stats.describe, threshold flags] → [six evidence JSONs under artifacts/timeline/<analysis>/evidence/: describe.json (runners.py:138), normality.json (runners.py:219), variance.json (runners.py:274), omnibus.json (runners.py:324), posthoc.json (runners.py:413), conclusion.json (runners.py:499); plus figures under reports/figures/ referenced by FigureRef] · pins: 6
- **GATE-TLINE-57** `src/broadway/timeline/sequence.py:47 load_walkthrough_sequence()`
  [configs/flow/hypothesis_walkthrough.yaml (step ids/orders/questions/kind/action), configs/step/walkthrough.yaml (skew/kurtosis/shapiro/imbalance/significance thresholds, max_qq_groups, decisions.{omnibus,posthoc}.{methods,parents})] → [WalkthroughSequence (sequence.py:22) and WalkthroughConfig (sequence.py:35) pydantic models; the single source of gate order, gate questions, threshold flags, and the decision allowlists/parent sets used by GATE-TLINE-52/-54/-55] · pins: 3
- **GATE-TLINE-58** `src/broadway/lineage/records.py:19 write_record()` ⚠FINDING
  [node_id strings from lineage/ids.py:1 node_id() (f"{kind}:{name}", no character validation), TransformAudit (lineage/models.py:73) from etl/canonicalize accounting, sample_name/sample_role, parents list, BROADWAY_LINEAGE_DIR env (records.py:8, default artifacts/lineage)] → [artifacts/lineage/records/<sanitized_node_id>.json LineageRecords; consumed only by graph.build_graph (graph.py:105-123) -> LineageGraph -> reports/lineage/graph.json + graph.md (lineage/module.py:27/:38) and mermaid rendering (mermaid.py:6); enforce_drop_fraction (records.py:43) raises before any record is written when unexplained row loss exceeds max_drop_fraction] · pins: 4
- **GATE-TLINE-59** `src/broadway/cli.py:105 main()` ⚠FINDING
  [argv parsed by _build_parser (cli.py:26-102): walkthrough --analysis/--dataset/--sample/--force (cli.py:87-92), decide --analysis/--method/--reason/--kind choices={omnibus,posthoc} (cli.py:94-100), lineage --analysis/--dataset (cli.py:40-42), report (cli.py:44-46), stats run|describe --sample (cli.py:80-84); console-script entry ds-pipeline = broadway.cli:main (pyproject.toml:53-54)] → [dispatch into timeline.walkthrough.run (cli.py:166-173), decide_module.record + timeline_module.save_decision (cli.py:174-189), lineage.module.run (cli.py:118-121), report path re-rendering results from persisted steps/decisions with 'run the walkthrough first' short-circuit (cli.py:122-134)] · pins: 5
- **GATE-TLINE-114** `src/broadway/lineage/state.py:5 LINEAGE_STEPS + src/broadway/baseline/module.py:87-91 parents` ⚠FINDING
  [flow step sequence] → [loud graph incompleteness when a declared parent kind never ran] · pins: none direct

### 07-surfaces — surfaces

- **GATE-SURF-60** `src/broadway/timeline/walkthrough.py:87 _write_timeline()  # renders via src/broadway/reports/index.py:61 render_dashboard()`
  [persisted AnalysisStep list (timeline/module.load_steps), persisted AnalysisDecision list, WalkthroughSequence (configs/flow walkthrough sequence), Suggestion from timeline/suggest.suggest_next] → [reports/index.md] · pins: 3
- **GATE-SURF-61** `src/broadway/timeline/walkthrough.py:85 _write_timeline()  # renders via src/broadway/reports/timeline.py:16 render_timeline()`
  [persisted AnalysisStep list, AnalysisDecision list, WalkthroughSequence, Suggestion appended as '## Suggested next action' (walkthrough.py:78-83)] → [reports/timeline.md] · pins: 3
- **GATE-SURF-62** `src/broadway/reports/results.py:244 write_results()   # sole writer of reports/results/*.md; index via _render_index results.py:146; save loop results.py:258-259`
  [persisted AnalysisStep list, AnalysisDecision list, WalkthroughSequence (labels slugified via slugify results.py:54)] → [reports/results/index.md, reports/results/describe-groups.md, reports/results/normality-diagnostics.md, reports/results/variance-homogeneity.md, reports/results/post-hoc-comparisons.md, reports/results/principal-analysis.md, reports/results/conclusion.md] · pins: 4
- **GATE-SURF-63** `src/broadway/reports/audit.py:684 run()               # writes all five audit pages at audit.py:694-706`
  [artifacts <dataset>_clean.json (StructuralCleanResult), <dataset>_join_audit.json, <dataset>_lookup_value_audit.json, artifacts/discover/profile.json (DatasetProfile), artifacts/discover/qq_overview.json (QqOverview)] → [reports/audit/index.md, reports/audit/profile.md, reports/audit/transform.md, reports/audit/join.md, reports/audit/lookup_values.md] · pins: 4
- **GATE-SURF-64** `src/broadway/lineage/module.py:15 run()               # graph.json at :27; graph.md at :38-40`
  [configs/ tree, artifacts/lineage records + decisions (lineage/records.py LINEAGE_DIR), LineageGraph from lineage/graph.build_graph + scope_graph, Mermaid text from lineage/mermaid.to_mermaid] → [reports/lineage/graph.json, reports/lineage/graph.md] · pins: 3
- **GATE-SURF-65** `src/broadway/stats/describe.py:117 plot_describe_figures() [savefig describe.png] + src/broadway/discover/qq.py:478 _plot_qq_joint() [savefig normality_qq.png]  # invoked solely by timeline/runners.py:143-146 and :210 inside run_describe()/run_normality()`
  [raw canonical/sample frame groups loaded by runners.load_frame_and_groups (runners.py:86), GroupSummary from stats.describe, viz config configs/step/viz.yaml (describe_figure :16, normality_figure :17)] → [reports/figures/describe.png, reports/figures/normality_qq.png] · pins: 4
- **GATE-SURF-66** `src/broadway/discover/module.py:74 _write_qq_overview() -> src/broadway/discover/qq.py:566 plot_numeric_qq()  # savefig sites qq.py:297/:356/:514/:562; called from module.py:78 (run() :105 and profile() :127)`
  [raw CSV/parquet frame, configs/step/viz.yaml knobs (qq_figure :14, qq_log_figure :15, dist_figure :15, diagnostics.figure :31, dpi, chunk sizes, zones/markers toggles), contract exclude_from_profiling list] → [reports/figures/numeric_qq_{n}.png, reports/figures/numeric_dist_{n}.png, reports/figures/numeric_qq_log_{n}.png, reports/figures/numeric_diagnostics.png, artifacts/discover/qq_overview.json (figure-name registry consumed by reports/audit)] · pins: 4
- **GATE-SURF-67** `tests/test_surface_integrity.py:41 test_report_markdown_links_resolve()`
  [git ls-files reports/ '*.md' (tracked surface inventory, :26-38), markdown link regex :23, size caps HTML_CAP_BYTES=5MiB :20 / PNG_CAP_BYTES=2MiB :21] → [] · pins: 3
- **GATE-SURF-68** `.gitignore:17-19  # project/experiments/results tracking convention (negation triad), pinned-sample negations :21-22`
  [experiment script outputs written ad hoc under project/experiments/, project/experiments/mlflow/_common.py RESULTS convention] → [project/experiments/results/**/*.csv (tracked), project/experiments/results/univariate/fare_amount_trip_distance/ratecode1_sample.parquet|.json (pinned, tracked)] · pins: 1
- **GATE-SURF-69** `src/broadway/reports/__init__.py  # renderer-purity contract over the package (markdown.py, results.py, timeline.py, index.py, audit.py, registry.py)`
  [persisted typed evidence JSON artifacts, persisted AnalysisStep/AnalysisDecision models, QqOverview/DatasetProfile/JoinAuditReport/LookupValueAuditReport/StructuralCleanResult models] → [(cross-cutting) all reports/**/*.md listed in GATE-SURF-60..64] · pins: 3
- **GATE-SURF-100** `src/broadway/inference/api.py:1 FastAPI app stub (module docstring only — NO app symbol at HEAD; gate demands its creation) + k8s/api-deployment.yaml:18 uvicorn command` ⚠FINDING
  [uvicorn ASGI target import (inference.api:app), models:/euromonitor@champion via MLflow registry] → [HTTP GET /health POST /predict GET /metrics bound 0.0.0.0:8000 — or loud CrashLoopBackOff with named cause on every replica] · pins: none direct
- **GATE-SURF-101** `src/broadway/discover/columns.py:10 run()`
  [raw CSV path (ds-pipeline columns subparser argv)] → [stdout per-column dtype report (read-only probe, zero artifact writes)] · pins: none direct
- **GATE-SURF-102** `src/broadway/reports/experiments_dashboard.py:56 FastAPI app custody (generic dashboard series endpoints)`
  [project-provided experiment results CSVs via BROADWAY_EXPERIMENTS_ROOT] → [dashboard series endpoints served by the FastAPI app object] · pins: 3
- **GATE-SURF-103** `project/experiments/euromonitor/01_eda.py:38 plot_barcode_coverage()`
  [project/data/euromonitor/dataset.csv] → [project/data/euromonitor/eda.parquet (evidence)] · pins: 1
- **GATE-SURF-104** `project/experiments/euromonitor/_common.py:49 load_dataset()`
  [project/data/euromonitor/dataset.csv, project/data/euromonitor/dataset_deduped.csv, project/config/experiments/euromonitor.yaml] → [project/experiments/results/euromonitor/**/* (shared RESULTS convention)] · pins: none direct
- **GATE-SURF-105** `project/experiments/euromonitor/_text.py:197 extract_volume_ml()`
  [] → [(shared) canonical volume ml, bucket_ml, validate_measurement, flavor/pack/disposition heuristics] · pins: 9
- **GATE-SURF-106** `project/experiments/euromonitor/_blocking.py:33 build_pairs()`
  [project/data/euromonitor/dataset_deduped.csv] → [(shared) true pairs + negative pairs + blocking recall/candidates] · pins: none direct
- **GATE-SURF-107** `project/experiments/euromonitor/_matching.py:16 build_vectorizer()`
  [] → [(shared) TF-IDF title vectors + cosine score_pairs] · pins: 3
- **GATE-SURF-108** `project/experiments/euromonitor/_hard_negatives.py:130 mine_hard_negatives()`
  [] → [(shared) hard-negative pairs + triplets + conflicting-barcode exclusion set] · pins: none direct
- **GATE-SURF-109** `project/experiments/euromonitor/01b_data_views.py:168 main()`
  [project/data/euromonitor/dataset.csv, project/experiments/results/euromonitor/01_dtypes.csv] → [project/experiments/results/euromonitor/01b_column_scatter.png, project/experiments/results/euromonitor/01b_product_space.png, project/experiments/results/euromonitor/01b_price_strip.png, project/experiments/results/euromonitor/01b_product_space.csv, project/experiments/results/euromonitor/01b_product_space_plot_data.csv] · pins: none direct
- **GATE-SURF-110** `project/experiments/euromonitor/01c_sparsity_noise.py:38 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/01c_sparsity_noise.csv, project/experiments/results/euromonitor/01c_price_outliers.png, project/experiments/results/euromonitor/01c_brand_long_tail.png, project/experiments/results/euromonitor/01c_cardinality.png] · pins: none direct
- **GATE-SURF-111** `project/experiments/euromonitor/01d_description_missingness.py:34 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/01d_description_missing.csv, project/experiments/results/euromonitor/01d_missing_by_category.png, project/experiments/results/euromonitor/01d_missing_by_retailer.png] · pins: none direct
- **GATE-SURF-112** `project/experiments/euromonitor/01e_barcode_analysis.py:23 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/01e_barcode_analysis.png] · pins: none direct
- **GATE-SURF-113** `project/experiments/euromonitor/01f_barcode_bias.py:35 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/01f_barcode_bias.csv] · pins: none direct
- **GATE-SURF-114** `project/experiments/euromonitor/01g_ground_truth_funnel.py:21 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/01g_ground_truth_funnel.png, project/experiments/results/euromonitor/01g_usable_by_country.png] · pins: none direct
- **GATE-SURF-115** `project/experiments/euromonitor/01h_cross_country_probe.py:31 main()`
  [project/data/euromonitor/dataset_deduped.csv, project/config/experiments/nlp.yaml] → [project/experiments/results/euromonitor/01h_cross_country_probe.png, project/experiments/results/euromonitor/01h_cross_country_spotcheck.csv] · pins: none direct
- **GATE-SURF-116** `project/experiments/euromonitor/02_volume_normalize.py:171 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/02_volume_normalize.csv, project/experiments/results/euromonitor/02_volume_agreement_before_after.csv, project/experiments/results/euromonitor/02_volume_disagreement.png, project/experiments/results/euromonitor/02_flavor_vocab.png, project/experiments/results/euromonitor/02_volume_agreement_before_after.png] · pins: none direct
- **GATE-SURF-117** `project/experiments/euromonitor/02b_volume_disagreement_split.py:55 main()`
  [project/experiments/results/euromonitor/02_volume_normalize.csv] → [project/experiments/results/euromonitor/02b_volume_disagreement_split.csv, project/experiments/results/euromonitor/02b_disagreement_summary.csv] · pins: none direct
- **GATE-SURF-118** `project/experiments/euromonitor/02c_case_analysis.py:61 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/02c_uppercase_letters.csv, project/experiments/results/euromonitor/02c_case_match_summary.csv, project/experiments/results/euromonitor/02c_case_normalization.png] · pins: none direct
- **GATE-SURF-119** `project/experiments/euromonitor/02d_measurement_validation.py:64 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/02d_measurement_status.csv, project/experiments/results/euromonitor/02d_sanity_checks.csv, project/experiments/results/euromonitor/02d_flagged_examples.csv, project/experiments/results/euromonitor/02d_measurement_status.png] · pins: 2
- **GATE-SURF-120** `project/experiments/euromonitor/03_pack_reconcile.py:32 main()`
  [project/experiments/results/euromonitor/02b_volume_disagreement_split.csv] → [project/experiments/results/euromonitor/03_pack_reconcile.csv] · pins: none direct
- **GATE-SURF-121** `project/experiments/euromonitor/03b_disposition.py:86 main()`
  [project/experiments/results/euromonitor/03_pack_reconcile.csv] → [project/experiments/results/euromonitor/03b_disposition.csv] · pins: none direct
- **GATE-SURF-122** `project/experiments/euromonitor/04_tfidf_matching.py:53 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/04_tfidf_stats.csv, project/experiments/results/euromonitor/04_tfidf_threshold.csv, project/experiments/results/euromonitor/04_tfidf_scores.png, project/experiments/results/euromonitor/04_tfidf_threshold.png] · pins: 2
- **GATE-SURF-123** `project/experiments/euromonitor/04b_category_utility.py:65 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/04b_macro_mapping.csv, project/experiments/results/euromonitor/04b_category_audit.csv, project/experiments/results/euromonitor/04b_category_audit.png] · pins: none direct
- **GATE-SURF-124** `project/experiments/euromonitor/04c_blocking_ab.py:44 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/04c_blocking_ab.csv, project/experiments/results/euromonitor/04c_blocking_ab.png] · pins: none direct
- **GATE-SURF-125** `project/experiments/euromonitor/05_blocking_feature_selection.py:59 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/05_blocking_feature_audit.csv, project/experiments/results/euromonitor/05_blocking_feature_audit.png] · pins: none direct
- **GATE-SURF-126** `project/experiments/euromonitor/05b_exact_duplicates.py:38 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/05b_exact_duplicates.csv, project/experiments/results/euromonitor/05b_duplicate_examples.csv, project/experiments/results/euromonitor/05b_exact_duplicates.png] · pins: none direct
- **GATE-SURF-127** `project/experiments/euromonitor/06_dedupe.py:39 main()`
  [project/data/euromonitor/dataset.csv] → [project/data/euromonitor/dataset_deduped.csv, project/data/euromonitor/sku_to_rep.csv, project/experiments/results/euromonitor/06_dedupe_summary.csv, project/experiments/results/euromonitor/06_ambiguous_offer_groups.csv] · pins: none direct
- **GATE-SURF-128** `project/experiments/euromonitor/06b_mislabeled_barcode_report.py:25 main()`
  [project/data/euromonitor/dataset.csv] → [project/experiments/results/euromonitor/06b_conflicting_barcode_groups.csv, project/experiments/results/euromonitor/06b_conflicting_summary.csv] · pins: none direct
- **GATE-SURF-129** `project/experiments/euromonitor/06c_validation_sets.py:33 main()`
  [project/data/euromonitor/dataset_deduped.csv] → [project/experiments/results/euromonitor/06c_validation_summary.csv, project/experiments/results/euromonitor/06c_hard_validation_pairs.csv] · pins: none direct
- **GATE-SURF-130** `project/experiments/euromonitor/07_nlp_hpo.py:50 main()`
  [project/data/euromonitor/dataset_deduped.csv, project/config/experiments/nlp.yaml] → [project/experiments/results/euromonitor/07_nlp_hpo_benchmark.csv, project/experiments/results/euromonitor/07_nlp_hpo_timing.csv, project/experiments/results/euromonitor/07_nlp_hpo_pareto.png] · pins: none direct
- **GATE-SURF-131** `project/experiments/euromonitor/07b_finetune.py:95 main()`
  [project/data/euromonitor/dataset_deduped.csv] → [project/experiments/results/euromonitor/07b_four_pop_scores.csv] · pins: none direct
- **GATE-SURF-132** `project/experiments/euromonitor/07c_field_ablation.py:30 main()`
  [project/data/euromonitor/dataset_deduped.csv] → [project/experiments/results/euromonitor/07c_field_ablation.csv] · pins: none direct
- **GATE-SURF-133** `project/experiments/euromonitor/07d_data_scaling.py:35 main()`
  [project/data/euromonitor/dataset_deduped.csv] → [project/experiments/results/euromonitor/07d_data_scaling.csv] · pins: none direct
- **GATE-SURF-134** `project/experiments/euromonitor/07_report_plots.py:34 main()`
  [project/data/euromonitor/dataset_deduped.csv, project/experiments/results/euromonitor/07c_field_ablation.csv, project/experiments/results/euromonitor/07d_data_scaling.csv, project/experiments/results/euromonitor/07b_four_pop_scores.csv] → [project/experiments/results/euromonitor/07_report_score_dist.png, project/experiments/results/euromonitor/07_report_pr_curve.png, project/experiments/results/euromonitor/07_report_threshold_sweep.png, project/experiments/results/euromonitor/07_report_error_breakdown.png, project/experiments/results/euromonitor/07_report_fn_analysis.csv, project/experiments/results/euromonitor/07_report_fn_breakdown.png, project/experiments/results/euromonitor/07_report_field_ablation.png, project/experiments/results/euromonitor/07_report_data_scaling.png, project/experiments/results/euromonitor/07_report_four_pop_dist.png] · pins: none direct
- **GATE-SURF-135** `project/experiments/euromonitor/make_notebook.py:11 md()`
  [project/data/euromonitor/sku_to_rep.csv, project/data/euromonitor/dataset_deduped.csv, project/experiments/results/euromonitor/07_nlp_hpo_benchmark.csv] → [project/experiments/euromonitor/entity_resolution.ipynb] · pins: none direct
- **GATE-SURF-136** `project/experiments/euromonitor/entity_resolution.ipynb:1 Euromonitor`
  [project/experiments/euromonitor/make_notebook.py] → [project/experiments/results/euromonitor/sku_to_item.csv] · pins: none direct
- **GATE-SURF-137** `project/experiments/euromonitor/07e_cross_encoder_rerank.py:229 main()`
  [project/data/euromonitor/dataset_deduped.csv, project/data/euromonitor/embeddings_cache (bi-encoder .npz, all-MiniLM-L6-v2)] → [project/experiments/results/euromonitor/07e_cross_encoder_rerank.csv, project/experiments/results/euromonitor/07e_cross_encoder_pairs.csv] · pins: none direct
- **GATE-SURF-138** `project/experiments/euromonitor/_link.py:23 resolve_items()`
  [] → [(shared) rep-level ITEM_ID + linking stats (n_items, edges, soft admitted/blocked, cand_pairs)] · pins: none direct
- **GATE-SURF-139** `project/experiments/euromonitor/08_pipeline.py:33 main()`
  [project/data/euromonitor/dataset.csv, project/data/euromonitor/sku_to_rep.csv] → [project/experiments/results/euromonitor/sku_to_item.csv] · pins: none direct

### 08-config — config-schema

- **GATE-CFG-70** `src/broadway/config/loader.py:92 config_path() + :102 _load_yaml()` ⚠FINDING
  [config-relative path via CONFIGS_DIR plus optional BROADWAY_CONFIG_OVERLAY_DIR, yaml.safe_load stream] → [raw dict[Any, Any] section payload, or raised gate failure] · pins: 6
- **GATE-CFG-71** `src/broadway/config/resolver.py:9 _resolve_string()` ⚠FINDING
  [every string scalar in the merged config dict] → [strings with ~/$VAR expanded, or the ORIGINAL string when the variable is unset] · pins: none direct
- **GATE-CFG-72** `src/broadway/config/loader.py:104 _merge_section()` ⚠FINDING
  [environment (required name, loader.py:193), dataset/experiment/analysis (optional CLI names, loader.py:194-196), step/<name>.yaml (required, loader.py:197)] → [single merged dict keyed by section] · pins: 3
- **GATE-CFG-73** `src/broadway/config/schema.py:62 EnvironmentConfig`
  [merged environment section (post GATE-CFG-71)] → [typed EnvironmentConfig instance embedded in PipelineConfig (loader.py:135)] · pins: 4
- **GATE-CFG-74** `src/broadway/config/schema.py:51 DatasetContract` ⚠FINDING
  [project/config/dataset/*.yaml (column pins: dtype/null_count/role per column, e.g. project/config/dataset/euromonitor.yaml:1-29; lookup_tables with value_policies/sentinel_values euromonitor.yaml:31-59)] → [DatasetContract with dict[str, ColumnSchema], consumed by pandera_dtype/build_raw_schema and every loader] · pins: 5
- **GATE-CFG-75** `src/broadway/contracts/pandera.py:46 build_raw_schema()` ⚠FINDING
  [DatasetContract.columns (post GATE-CFG-74)] → [pa.DataFrameSchema, one pa.Column per contract entry, coerce=False strict dtypes, nullable=True (pandera.py:55-59)] · pins: 5
- **GATE-CFG-76** `src/broadway/config/schema.py:187 ExperimentConfig` ⚠FINDING
  [configs/experiment/*.yaml (data_source/features/model/split/hpo/preprocessing), allowed_params registry (training/models/registry.py)] → [validated ExperimentConfig; HPO search spaces registry-checked; flow-mode map checked] · pins: 10
- **GATE-CFG-77** `src/broadway/features/recipe.py:102 validate_preprocessing_columns()`
  [PipelineConfig with experiment.preprocessing + data_source.schema_contract + dataset] → [config-load pass, or ValueError naming offending step/columns/schema_contract] · pins: 12
- **GATE-CFG-78** `src/broadway/config/loader.py:155 resolve_full_steps()`
  [PipelineConfig(full+analysis), configs/flow/<name>.yaml (FlowConfig steps list, schema.py:319-320)] → [ordered list[str] of concrete step names for the analysis mode] · pins: 5
- **GATE-CFG-79** `src/broadway/samples/loader.py:34 _build_schema()` ⚠FINDING
  [SampleSpec.schema block from project/config/sample/*.yaml (e.g. project/config/sample/demo.yaml:29-36 dtype/nullable/checks), provenance JSON sidecar, parquet artifact] → [pa.DataFrameSchema with op-derived pa.Checks (_CHECK_BUILDERS map loader.py:24-31); validated Sample(df, spec, provenance)] · pins: 9
- **GATE-CFG-103** `src/broadway/config/loader.py:50 STEP_MODULES` ⚠FINDING
  [step name argv] → [module binding or loud unknown-step error] · pins: 1
- **GATE-CFG-105** `src/broadway/onboard/module.py:215 init() (_write_configs :178-198)`
  [stdin prompts + 13 argv flags] → [configs dataset/analysis/experiment YAMLs + profile JSON + 1 lineage record (write call :302)] · pins: none direct
- **GATE-CFG-107** `src/broadway/data/loader.py:134 lookup pre-read existence admission (declared-lookup bootstrap check)` ⚠FINDING
  [DatasetContract.lookup_tables paths — project/config/dataset/euromonitor.yaml:31-34 declares out-of-repo symlink data/raw/(out-of-repo lookup symlink)] → [dangling-symlink-aware loud pre-merge error NAMING the bootstrap step] · pins: none direct
- **GATE-CFG-108** `src/broadway/onboard/infer.py:10 _IDENTIFIER_THRESHOLD`
  [BROADWAY_IDENTIFIER_THRESHOLD env var] → [typed float threshold or named parse error] · pins: none direct
- **GATE-CFG-112** `src/broadway/lineage/models.py:37 SampleSpec.column_mapping (+ consumers stats/module.py:42, stats/describe.py:132, timeline/runners.py:100)` ⚠FINDING
  [column_mapping block + analysis group_column (project/config/analysis/euromonitor.yaml group_column)] → [validated mapping direction (logical→source); load-time existence validation of mapped values against the sample artifact] · pins: 3

### 80-hpo-optuna — hpo-optuna

- **GATE-HPO-80** `src/broadway/training/hpo.py:120 run_model_study()` ⚠FINDING
  [ModelHPOSpec search space via _trial_objective (hpo.py:99), HPOConfig.direction + storage_url (CFG-HPO-SPEC / CFG-HPO-EXPERIMENT), random_state seed int] → [ARTIFACT-OPTUNA-STUDY (optuna.Study — in-memory object or RDB-backed shared study)] · pins: 4
- **GATE-HPO-81** `src/broadway/training/hpo.py:30 make_objective()` ⚠FINDING
  [PipelineConfig experiment block, ModelHPOSpec.name registry key, CFG-HPO-SPEC.target_metric, X/y train+val frames] → [Objective(params[, trial]) -> float closure; per-trial 'broadway_metrics' user attr] · pins: 5
- **GATE-HPO-82** `src/broadway/training/hpo.py:159+:177 study.optimize call sites (run_model_study leg + _optimize_study bandit continuation)` ⚠FINDING
  [_trial_objective wrapper (hpo.py:99), n_trials budget (initial_trials_per_model or bandit allocation), callbacks list from GATE-HPO-83] → [COMPLETE/PRUNED/FAIL trials appended to ARTIFACT-OPTUNA-STUDY] · pins: 5
- **GATE-HPO-83** `src/broadway/training/hpo.py:66 _mlflow_callback()` ⚠FINDING
  [FrozenTrial states/values/user attrs from the optimize callbacks hook, ambient MLFLOW_TRACKING_URI, caller mlflow_tags] → [one nested ARTIFACT-MLFLOW-RUN per COMPLETE trial (params + numeric metrics + tags)] · pins: 2
- **GATE-HPO-84** `src/broadway/training/hpo.py:350 best_model min()/max() selection in run_hpo_bandit()` ⚠FINDING
  [leaderboard from _leaderboard (:285-293), studies dict, cfg.experiment.model.type (trainer side)] → [result dict {'models': {name: {best_params, best_value, n_trials}}, 'best_model', 'best_params', 'best_value'}] · pins: 2
- **GATE-HPO-85** `src/broadway/training/hpo.py:180 bandit_allocate()`
  [leaderboard {model: best objective} (lower=better for minimize, higher=better for maximize), remaining trial budget (:330), top_k (CFG-HPO-SPEC), direction str (:184, default minimize)] → [{model: n_trials} allocation dict fed to _bandit_round (:333)] · pins: 10
- **GATE-HPO-86** `src/broadway/training/optuna.py:48 run_study_rdb()` ⚠FINDING
  [storage_url string (sqlite:/// locally; postgresql://…:5432/optuna in k8s via compose_db_url), study_name, direction, optional random_state] → [durable Optuna schema in the RDB (ARTIFACT-OPTUNA-STUDY at rest); stale RUNNING trials flipped FAIL on resume, ARTIFACT-OPTUNA-SNAPSHOT dumps under gitignored data/optuna-backup/] · pins: 5
- **GATE-HPO-88** `src/broadway/training/hpo.py:204 _initial_round()` ⚠FINDING
  [random_state base seed, hpo.models list ORDER, CFG-HPO-SPEC / CFG-HPO-EXPERIMENT seed literals upstream] → [deterministic per-model sampler seeds; reproducible trajectories across runs/processes] · pins: 3
- **GATE-HPO-89** `src/broadway/config/schema.py:125 HPOConfig` ⚠FINDING
  [configs/experiments/mlflow.yaml hpo: block (:14-44) = CFG-HPO-SPEC, ExperimentConfig.hpo (experiment-embedded twin, e.g. configs/experiment/hyperopt.yaml:23-36) = CFG-HPO-EXPERIMENT, k8s configmap inline config.yaml = CFG-K8S-OPTUNA-INFRA (infra keys only)] → [typed HPOConfig{engine, direction, total_trials, initial_trials_per_model, top_k, target_metric, models[].search_space, storage_url}] · pins: 3
- **GATE-HPO-154** `src/broadway/training/nlp.py:97 precision_at_recall_breakdown() + :122 calibrate_isotonic() + :147 calibrate_isotonic_heldout() + :200 precision_ci() + :227 split_pos_by_country() + :248 log_nlp_eval() + :411 make_objective() + :464 run_nlp_hpo() + :514 run_nlp()`
  [src/broadway/config/schema.py:135 NLPConfig (typed data-agnostic config: model_zoo + hpo + knobs), project/config/experiments/nlp.yaml model_zoo + hpo: block (bi-encoder zoo SSOT, direction: maximize), sentence-transformers bi-encoders resolved via model_zoo (zero-shot), project/experiments/euromonitor/07_nlp_hpo.py pair construction (same 10,891 pos / 10,000 neg population as step 04)] → [per-model entity-resolution metrics (auc, average_precision, recall_at_5pct_fpr, precision_at_90pct_recall, f1_at_5pct_fpr, tp_at_90pct_recall, fp_at_90pct_recall, threshold_at_90pct_recall, tp_at_5pct_fpr, fp_at_5pct_fpr, threshold_at_5pct_fpr, pos_median, neg_p90, encode_s) via trial broadway_metrics user attr, bandit result {models, best_model, best_params, best_value, metrics} (direction: maximize, reused run_hpo_bandit)] · pins: 5

### 09-infra — infra-meta

- **GATE-INFRA-90** `scripts/run_local_ci.sh:16 case-dispatch STATIC/TIER/CLEAN_LINT (+ run() aggregator :26-31, banner law :23/:100)` ⚠FINDING
  [argv[1] ∈ {unset, --static, --tier=fast, --tier=full, --clean-lint} (else usage exit 2), sub-gate verdicts via run() <name> <cmd...>] → [banner FAST-GREEN | LOCAL-CI GREEN | LOCAL-CI RED, exit 0 green / 1 red / 2 usage] · pins: none direct
- **GATE-INFRA-91** `scripts/run_local_ci.sh:30 gate_parity() (F1b pin-guard, wired at :43)` ⚠FINDING
  [refs/remotes/origin/sklearn:scripts/check_branch_parity.sh via `git show` (:33)] → [parity sub-verdict to run() aggregator, FAIL parity (F1b): origin/sklearn unavailable | legacy pre-D16 checker on track ref] · pins: 3
- **GATE-INFRA-92** `scripts/run_local_ci.sh:102 gate battery (ruff, mypy, Vulture, configs, shell-scripts, data-refs, graphify, pytest+cov floor=95, project-tests)` ⚠FINDING
  [src/** tests/test_project_paths.py project/experiments/** project/experiments.py project/dashboard.py project/paths.py project/working.py project/data.py scripts/check_project_paths.py scripts/ (ruff), project/config/ (project-owned layout, experiment, and config-overlay SSOT), src/broadway/** (mypy), configs/experiment/*.yaml via load_config(dataset='test') (configs), k8s/optuna/*.sh + scripts/*.sh via sh -n + shellcheck (shell-scripts), Dockerfile COPY/ADD sources + ci.yml -f dockerfile + k8s/*.yaml + config parquet/path/file refs via scripts/check_data_refs.py against project/config/layout.yaml build:* (data-refs), graphify-out/graph.json callables vs agents/ledger/gates.yaml owner fields via scripts/check_graphify_surfaces.py (graphify), src/broadway project scripts via Vulture --min-confidence 95 (vulture), tests/** (pytest -n 4 --dist worksteal, --cov-fail-under=95), project/tests/** (project-tests: full tier only, -q --dist worksteal, NO coverage flags)] → [PASS/FAIL ruff|mypy|configs|project-paths|shell-scripts|data-refs|graphify|vulture|pytest|project-tests banners + 40-line tails, cov floor breach ⇒ FAIL pytest] · pins: 4
- **GATE-INFRA-93** `scripts/check_branch_parity.sh:71 check() (SHARED lockstep, list at :43-69) + sync_to_main() :89` ⚠FINDING
  [24-entry SHARED surface: src/ tests/ demo/ configs/dataset/test.yaml configs/experiment/{baseline,engineered,hyperopt}.yaml configs/analysis/{test,test_hypothesis,test_causal}.yaml configs/step/{causal,etl}.yaml configs/environment/ configs/flow/ k8s/ docker/ .github/workflows/ pyproject.toml Dockerfile docker-compose.yml .gitignore .dockerignore README.md scripts/, origin/main vs origin/taxi tips] → [PARITY OK | DRIFT: <path> differs … PARITY FAILED — run $0 --sync, sync mode: taxi→main checkout + deletion mirror (:89-103)] · pins: 3
- **GATE-INFRA-94** `scripts/check_branch_parity.sh:111 inline era declaration PARITY_ERA/PARITY_TRACK_BRANCH/PARITY_ALLOWLIST/PARITY_MAIN_ANCHOR (:111-114) + anchor guards :121-132 + dev-era dispatch :192-226` ⚠FINDING
  [inline constants (no env dialect, D21), GITHUB_REF_NAME else `git rev-parse --abbrev-ref HEAD` (:187), origin refs] → [PARITY OK (era=… branch=…) | FATAL anchor shape/resolution errors | TAXI DRIFT | FORK | REFUSED (--sync off-era)] · pins: 2
- **GATE-INFRA-95** `scripts/check_branch_parity.sh:134 custody() — layer 1 anchor-drift diff :146, freeze-intact shortcut :162, layer 2 blob-provenance comm -23 :166` ⚠FINDING
  [PARITY_MAIN_ANCHOR=18607091ddbb2602ad4475341ad377bafee5ec4b (:114), origin/main SHARED subtree, origin/sklearn object universe (rev-list --objects :178), PARITY_ALLOWLIST=() prefix skips (:168-177)] → [ROGUE MAIN WRITE: frozen main changed since anchor … (adds/deletes/mods), ROGUE MAIN WRITE: novel blob(s) absent from track universe (head -10), silent return 0 via shortcut when main==anchor] · pins: none direct
- **GATE-INFRA-96** `scripts/ship.sh:24 ship gate (`if ! bash scripts/run_local_ci.sh` → refuse :24-27; single push later) + WAVE-A teeth insertions (:15-19 tier_gate sourcing, :31-40 L1 hook guard, :42-53 TIER-GATE batch) mirrored by .git/hooks/pre-push:5` ⚠FINDING
  [argv remote/refspec (default origin sklearn:sklearn :14), full-tier verdict of scripts/run_local_ci.sh] → [SHIP OK | SHIP REFUSED: LOCAL-CI RED … exit 1, single git push invocation → one hook gate (:25)] · pins: none direct
- **GATE-INFRA-97** `scripts/check_e2e_determinism.sh:106 compare_trees() (json_diff whitelist :35-46, compare_file :89, --run chain run_e2e :163)` ⚠FINDING
  [two artifact trees (positional) or --run, JSON leaves after canonical sorted-keys re-dump, whitelist EXACT={trace.created_at, artifact_path, train_time_seconds, promote, reason, warnings} :45 + PATTERN comparison.metrics.<m>.{champion,delta,delta_pct} :46] → [DETERMINISM OK + exit 0 | one `<file>: <field-path>` line per offender + exit 1 | `<file>: missing counterpart` | usage exit 2] · pins: 6
- **GATE-INFRA-98** `scripts/tier_classifier.py:104 classify() (triggers :51-64, parse_diff_payload :138, CLI main :162)` ⚠FINDING
  [git-diff payload on stdin (files + added lines), governance basenames {CONTRACT_TEMPLATE,WORKER_CONTRACT,MAIN_AGENT_CONTRACT,DECISIONS,FIXES}.md + agents/ledger/STATE.md (:34-37), behavior prefixes src/ tests/ project/ scripts/ .github/ k8s/ + pyproject.toml uv.lock *.sh docker* + configs/*.yaml (:40-43)] → [{"tier": FULL|CHECKLIST, "reasons": [...]} JSON] · pins: 7
- **GATE-INFRA-99** `.github/workflows/ci.yml:47 platform job step "Platform gates (SSOT)" (delegation law :42-46; docker-only checks :51-66; build-and-boot :151; CD job :310; concurrency :11-13)` ⚠FINDING
  [push/PR to all branches (branches: ['**'], :3-7), fetch-depth 0 for parity tips (:28), uv sync --all-extras --frozen (:40), bash scripts/run_local_ci.sh (no args ⇒ full tier), k8s/optuna/ + project/k8s/optuna/configmap.yaml (Kubernetes manifest inputs)] → [platform job verdict (parity+ruff+mypy+configs+shell-scripts+pytest+cov≥95+project-tests via SSOT script), docker-only verdicts: sh -n + shellcheck k8s/optuna/*.sh AND scripts/*.sh (:57-66), kubeconform -strict k8s/optuna/ plus project/k8s/optuna/configmap.yaml minus kind-config.yaml (:79), orchestrator dry-run render+kubeconform (:104), sha-tagged images built/boot-tested; CD publishes bit-for-bit verified tarball to GHCR on main/taxi pushes only (:290-376)] · pins: none direct
- **GATE-INFRA-123** `project/dashboard.py:15 main() project composition entry`
  [host/port argv] → [dashboard server (uvicorn.run(app, host="127.0.0.1", port=8000) :19)] · pins: 1
- **GATE-INFRA-127** `./Dockerfile:19 CMD bare deployable-image default entrypoint`
  [deployable image build] → [CMD ["ds-pipeline"] default entrypoint (:19)] · pins: none direct
- **GATE-INFRA-128** `docker-compose.yml:3 mlflow/postgres service command custody` ⚠FINDING
  [compose stack definition] → [mlflow server command (:8 block) over built contexts ./docker/mlflow (:3) + ./docker/postgres (:16) with named volumes (:26)] · pins: none direct
- **GATE-INFRA-131** `k8s/api-deployment.yaml:40 HorizontalPodAutoscaler bounds`
  [api workload metrics] → [autoscaling bounds minReplicas: 2 / maxReplicas: 10 / averageUtilization: 70 (:48-56)] · pins: none direct
- **GATE-INFRA-133** `k8s/postgres-deployment.yaml:2 StatefulSet durable-state workload`
  [postgres manifest apply] → [durable-state workload custody (kind: StatefulSet :2)] · pins: none direct
- **GATE-INFRA-137** `docker-compose.yml:26 compose-stack build/volume FILE law`
  [compose stack definition] → [FILE law: build contexts resolve, volumes declared (:3/:16/:26)] · pins: none direct
- **GATE-INFRA-138** `tests/conftest.py:8 _SNAPSHOT_DIRS snapshot hygiene`
  [pytest session lifecycle] → [working-tree snapshot/restore of artifacts/ + reports/ around test runs] · pins: none direct
- **GATE-INFRA-139** `tests/test_uv_probe_guard.py:27 package-discovery probe guard suite`
  [uv editable-install discovery behavior] → [package-discovery guard contract (excludes the harness checkout, walks the repo root)] · pins: 2
- **GATE-INFRA-141** `scripts/ship.sh push-path creator site for annotated tags (law: manual git tag -a mints ONLY via a named ledger-row procedure; NO scripted minter exists at HEAD)` ⚠FINDING
  [release/tag minting intent] → [published annotated tag with a ledger row naming its procedure] · pins: none direct
- **GATE-INFRA-142** `scripts/check_branch_parity.sh:89 sync_to_main() ref-management site (stale-ref retirement law; objects: LOCAL refs refs/heads/pr-1 + pr-2, closed-unmerged snapshots)` ⚠FINDING
  [named local refs (pr-1/pr-2 and future strays)] → [retire-or-own verdict per named ref; detect-and-retire loop execution record] · pins: none direct
- **GATE-INFRA-143** `.github/workflows/ci.yml:145 actions/cache@v4 cache retention/dedupe law`
  [CI cache key family (broadway-base hashFiles key)] → [one save-site per key family; size-budget watch; purge as invoked workflow step] · pins: none direct
- **GATE-INFRA-144** `.github/workflows/ci.yml:355 Push verified images to GHCR (CD single writer)` ⚠FINDING
  [verified image builds from the CD job] → [registry writes: broadway-optuna-worker + mlflow-server ONLY, $sha always; :latest main-only; :taxi taxi-only] · pins: none direct
- **GATE-INFRA-146** `scripts/uv.sh:1 host-local cache selector — the sole UV_CACHE_DIR runtime owner`
  [HOME/XDG_CACHE_HOME/TMPDIR/UV_CACHE_DIR environment] → [writable host-local uv cache or loud nonzero failure] · pins: 3
- **GATE-INFRA-147** `scripts/deadcode_census.py:1 module — teeth ⑥ DEADCODE-CENSUS advisory engine`
  [tracked *.py corpus via git ls-files (src/project/tests/scripts), pyproject [project.scripts] entrypoint table] → [data/processed/deadcode_census.md suspicion report (gitignored sink); stdout default] · pins: none direct
- **GATE-INFRA-148** `agents/tools/state_records.py sync()`
  [STATE CURRENT record and immutable EVENTS section, GitHub Project] → [pending CURRENT intent followed by one synced mirror item] · pins: 1
- **GATE-INFRA-149** `scripts/check_graphify_surfaces.py:1 main() graphify surface-reconciliation gate (callables vs gates.yaml owners)`
  [graphify-out/graph.json (deterministic --code-only AST graph over src/ + project/), agents/ledger/gates.yaml owner fields] → [PASS/FAIL graphify banner + KNOWN_UNMAPPED baseline report] · pins: 3

### 81-object-custody — object-custody

- **GATE-CUST-147** `src/broadway/training/mlflow_utils.py:162 promote_candidate()`
  [model_uri + alias="champion"] → [registered version + alias move; elsewhere loud refusal] · pins: 1
- **GATE-CUST-148** `src/broadway/training/mlflow_utils.py:163 register_model call sites`
  [candidate model_uri + dataset name] → [registered model name pinned to cfg.dataset.name (= configmap dataset.name)] · pins: none direct
- **GATE-CUST-150** `src/broadway/training/mlflow_utils.py:59 set_experiment auto-create chokepoint (inside setup_mlflow :54-63)`
  [experiment_name argument at setup time] → [experiment created/reopened ONLY within lawful namespaces: ratecode1_model_battle family ∪ dataset-name experiments; orphans refused loudly] · pins: none direct
