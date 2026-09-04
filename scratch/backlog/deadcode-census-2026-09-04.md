# Dead-Code Census — ADVISORY (D34 tripwire 1)

> Ratified law (verbatim): census is ADVISORY — output files findings to the backlog; NEVER a red gate; measures suspicion, not guilt.

- Generated (UTC): 2026-09-04T12:40:48+00:00
- HEAD: `9d14ecf1a364`
- Disposition: file to backlog; NEVER a gate; every item below is
  suspicion-not-guilt until a human rules.

## Methodology & limitations

- Corpus: 298 tracked `*.py` files under src/, project/, experiments/, tests/, scripts/
  (exact-token AST graph: Name ids, Attribute attrs, string literals;
  substring matching NEVER used).
- Candidate surface: 563 module-level defs/classes in
  `src/broadway/` (2 `__init__.py`-resident defs excluded
  by policy).
- Class/method/nested-def level death is OUT OF SCOPE (module-level only).
- Same-name defs across modules blur attribution; such suspects carry a
  collision note instead of the uniqueness point.
- Unparseable or unreadable files are logged to stderr and dropped from
  BOTH sides of the graph (disclosed bias toward under-reporting).
- Suspicion bands: HIGH ≥5, ELEVATED 4, MODERATE 3, LOW ≤2 (max 5:
  zero-refs 2 + no-string-hit 1 + test-blind module 1 + unique name 1).

## A1. Zero-reference module-level defs/classes (strongest suspicion)

_none found_

## A2. Same-module-only defs (never referenced from any other tracked file)

- **class `HypothesisConfig`** — `src/broadway/analysis/contracts.py:14` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_build_trace`** — `src/broadway/baseline/module.py:30` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_welch_df`** — `src/broadway/causal/analysis.py:12` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_mean_diff_ci`** — `src/broadway/causal/analysis.py:22` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_allocate`** — `src/broadway/causal/assignment.py:18` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_fractional_refusal`** — `src/broadway/cleaning/structural.py:67` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_add_step_args`** — `src/broadway/cli.py:19` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_build_parser`** — `src/broadway/cli.py:26` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_deep_merge`** — `src/broadway/config/loader.py:82` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_merge_section`** — `src/broadway/config/loader.py:118` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_missing_step_sections`** — `src/broadway/config/loader.py:127` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_build_config`** — `src/broadway/config/loader.py:138` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_resolve_string`** — `src/broadway/config/resolver.py:9` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **class `DiagnosticsThresholds`** — `src/broadway/config/viz.py:40` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_random_split`** — `src/broadway/data/splitter.py:11` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_time_split`** — `src/broadway/data/splitter.py:16` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_stratified_split`** — `src/broadway/data/splitter.py:22` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_assign_role`** — `src/broadway/discover/module.py:27` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_build_contract`** — `src/broadway/discover/module.py:41` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_log_identifier_recommendations`** — `src/broadway/discover/module.py:68` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_write_qq_overview`** — `src/broadway/discover/module.py:74` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_stringify`** — `src/broadway/discover/profile.py:29` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_numeric_cols`** — `src/broadway/discover/qq.py:103` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_resolve_min_unique`** — `src/broadway/discover/qq.py:109` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_grid_dims`** — `src/broadway/discover/qq.py:130` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `draw_qq_zones`** — `src/broadway/discover/qq.py:136` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `attach_qq_legend`** — `src/broadway/discover/qq.py:220` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_new_figure`** — `src/broadway/discover/qq.py:229` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_qq_log_pairs`** — `src/broadway/discover/qq.py:361` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_explained_rows`** — `src/broadway/etl/module.py:32` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **class `Predictable`** — `src/broadway/evaluate/explain.py:19` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_load_train_features`** — `src/broadway/evaluate/module.py:55` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_load_training_result`** — `src/broadway/evaluate/module.py:64` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_load_candidate`** — `src/broadway/evaluate/module.py:72` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_in_hour_window`** — `src/broadway/features/builders.py:32` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_rate_per_hour`** — `src/broadway/features/builders.py:36` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `engineered_schema_for`** — `src/broadway/features/generic.py:36` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `validate_target_dtype`** — `src/broadway/features/generic.py:59` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_assert_no_extra_columns`** — `src/broadway/features/generic.py:91` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_validate_step_params`** — `src/broadway/features/recipe.py:33` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_coerce_bool_param`** — `src/broadway/features/recipe.py:49` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_key_series`** — `src/broadway/features/transformers.py:18` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_read_yaml_dir`** — `src/broadway/lineage/graph.py:39` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_dedupe_edges`** — `src/broadway/lineage/graph.py:52` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_forward_closure`** — `src/broadway/lineage/graph.py:162` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_datetime_candidate`** — `src/broadway/onboard/infer.py:13` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_is_categorical`** — `src/broadway/onboard/infer.py:28` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_prompt`** — `src/broadway/onboard/module.py:41` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_prompt_required`** — `src/broadway/onboard/module.py:47` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_split_list`** — `src/broadway/onboard/module.py:54` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `build_dataset_contract`** — `src/broadway/onboard/module.py:60` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `build_analysis_contract`** — `src/broadway/onboard/module.py:97` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_scaffold_random_state`** — `src/broadway/onboard/module.py:119` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `build_experiment_config`** — `src/broadway/onboard/module.py:127` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_print_summary`** — `src/broadway/onboard/module.py:178` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_write_configs`** — `src/broadway/onboard/module.py:187` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_write_profile`** — `src/broadway/onboard/module.py:209` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_identifier_threshold`** — `src/broadway/reports/audit.py:29` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_sig3`** — `src/broadway/reports/audit.py:37` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_incomplete_answer`** — `src/broadway/reports/audit.py:41` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_page`** — `src/broadway/reports/audit.py:45` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_figure_block`** — `src/broadway/reports/audit.py:80` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_profile_evidence`** — `src/broadway/reports/audit.py:99` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_dataset_status`** — `src/broadway/reports/audit.py:472` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_lookup_affected_total`** — `src/broadway/reports/audit.py:491` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_considerations`** — `src/broadway/reports/audit.py:545` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_load`** — `src/broadway/reports/audit.py:677` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_series_dir`** — `src/broadway/reports/experiments_dashboard.py:71` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `script_profile`** — `src/broadway/reports/experiments_dashboard.py:108` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_result_files`** — `src/broadway/reports/experiments_dashboard.py:169` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_has_results`** — `src/broadway/reports/experiments_dashboard.py:177` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_reorder_moves`** — `src/broadway/reports/experiments_dashboard.py:420` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_series_selector`** — `src/broadway/reports/experiments_dashboard.py:452` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_dashboard_page`** — `src/broadway/reports/experiments_dashboard.py:464` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_experiment_page`** — `src/broadway/reports/experiments_dashboard.py:500` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_series_cards`** — `src/broadway/reports/experiments_dashboard.py:566` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_new_step_page`** — `src/broadway/reports/experiments_dashboard.py:719` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_script_for`** — `src/broadway/reports/experiments_dashboard.py:749` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_done`** — `src/broadway/reports/index.py:12` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_resolved`** — `src/broadway/reports/index.py:55` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_is_pvalue_key`** — `src/broadway/reports/results.py:18` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `humanize_value`** — `src/broadway/reports/results.py:27` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_flatten`** — `src/broadway/reports/results.py:41` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_decision_for`** — `src/broadway/reports/results.py:117` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_index`** — `src/broadway/reports/results.py:146` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_render_step_page`** — `src/broadway/reports/results.py:165` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_read_source`** — `src/broadway/samples/generate.py:50` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_apply_filters`** — `src/broadway/samples/generate.py:60` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_apply_derived`** — `src/broadway/samples/generate.py:68` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_apply_exclude_any`** — `src/broadway/samples/generate.py:87` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_build_schema`** — `src/broadway/samples/loader.py:34` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_read_lookup_columns`** — `src/broadway/schemas/joined.py:15` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_group_sizes`** — `src/broadway/stats/anova.py:18` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_welch_anova`** — `src/broadway/stats/anova.py:22` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_build_plan`** — `src/broadway/stats/anova.py:45` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_draw_group_sizes`** — `src/broadway/stats/describe.py:87` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_plot_residuals_vs_fitted`** — `src/broadway/stats/diagnostics.py:34` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_bp_statistics`** — `src/broadway/stats/diagnostics.py:67` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_influence_statistics`** — `src/broadway/stats/diagnostics.py:89` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_plot_cooks_distance`** — `src/broadway/stats/diagnostics.py:95` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `plot_cooks_distance`** — `src/broadway/stats/diagnostics.py:112` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_plot_residuals_qq`** — `src/broadway/stats/diagnostics.py:145` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_plot_residuals_histogram`** — `src/broadway/stats/diagnostics.py:152` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_jb_statistics`** — `src/broadway/stats/diagnostics.py:176` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_question_for`** — `src/broadway/timeline/decide.py:14` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_attrition`** — `src/broadway/timeline/runners.py:64` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **class `DecisionSpec`** — `src/broadway/timeline/sequence.py:28` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_alternative`** — `src/broadway/timeline/suggest.py:17` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_describe`** — `src/broadway/timeline/suggest.py:21` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_normality`** — `src/broadway/timeline/suggest.py:58` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_variance`** — `src/broadway/timeline/suggest.py:100` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_omnibus`** — `src/broadway/timeline/suggest.py:143` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_decide_omnibus`** — `src/broadway/timeline/suggest.py:196` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_decide_posthoc`** — `src/broadway/timeline/suggest.py:220` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_posthoc`** — `src/broadway/timeline/suggest.py:244` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_suggest_not_started`** — `src/broadway/timeline/suggest.py:260` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_resolved_decision`** — `src/broadway/timeline/walkthrough.py:64` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_write_timeline`** — `src/broadway/timeline/walkthrough.py:71` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_handle_failure`** — `src/broadway/timeline/walkthrough.py:94` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_print_suggestion`** — `src/broadway/timeline/walkthrough.py:126` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_print_posthoc_decision_required`** — `src/broadway/timeline/walkthrough.py:175` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_warn_stale_decisions`** — `src/broadway/timeline/walkthrough.py:192` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_run_describe`** — `src/broadway/timeline/walkthrough.py:204` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_run_normality`** — `src/broadway/timeline/walkthrough.py:214` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_run_variance`** — `src/broadway/timeline/walkthrough.py:223` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_run_posthoc`** — `src/broadway/timeline/walkthrough.py:245` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_run_conclusion`** — `src/broadway/timeline/walkthrough.py:266` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_mlflow_callback`** — `src/broadway/training/hpo.py:66` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_trial_objective`** — `src/broadway/training/hpo.py:99` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_optimize_study`** — `src/broadway/training/hpo.py:163` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_initial_round`** — `src/broadway/training/hpo.py:208` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_bandit_round`** — `src/broadway/training/hpo.py:248` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_leaderboard`** — `src/broadway/training/hpo.py:285` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_is_unreachable_http_store`** — `src/broadway/training/mlflow_utils.py:66` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_unreachable_server_hint`** — `src/broadway/training/mlflow_utils.py:75` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `classify_champion`** — `src/broadway/training/mlflow_utils.py:167` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_xy`** — `src/broadway/training/module.py:54` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_resolve_params`** — `src/broadway/training/module.py:59` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_extract_broadway_metrics`** — `src/broadway/training/nlp.py:485` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)
- **def `_create_study_with_retry`** — `src/broadway/training/optuna.py:24` — suspicion **4/5 (ELEVATED)**
  - rationale: zero cross-corpus references outside its own body; name absent from every string literal (no dynamic-dispatch escape); defining module IS imported by tests/ (blind-spot point withheld); name unique across all tracked top-level defs (resolution unambiguous)

## B. Console-script entrypoints (EXEMPT-BY-DESIGN)

Packaging console-scripts invoke these targets at install time;
absence of in-tree python imports is EXPECTED, not evidence of death.

- `ds-pipeline` → `broadway.cli:main` — target-found; in-tree references: 1 (tests/test_cli_dispatch.py)

## C. configs/** leaf keys with no loader-side token match (HEURISTIC)

Best-effort key-name match over identifier and string-literal tokens;
a hit anywhere (schema field, `cfg['key']`, docs string) suppresses the
row, so false NEGATIVES are likely and every row needs human review.

- `configs/nlp.yaml` → `model_zoo.minilm_l6` — token `minilm_l6` unmatched — proof: `git grep -n -w 'minilm_l6' -- '*.py'` (HEURISTIC)
- `configs/nlp.yaml` → `model_zoo.bge_small` — token `bge_small` unmatched — proof: `git grep -n -w 'bge_small' -- '*.py'` (HEURISTIC)

## Known-intentional exclusions (derived categories, not a symbol allowlist)

- `__init__.py`-resident defs: excluded by policy (package re-export
  surface); count excluded: see methodology.
- Entrypoint targets: section B, exempt-by-design.
- `__main__`-guard-sustained defs (in-file runnable demos; the qq-demo `__main__` ratification packet class): none found
- dynamic-string-only referenced defs (exact string-literal mention is their only cross-file evidence — weak, review manually): none found
- `__init__` re-export-sustained defs (referenced ONLY from packages'  `__init__.py`): none found

_End of census. Advisory only — route findings through the ledger._
