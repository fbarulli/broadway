# Surface & Analysis Preservation — non-encodable residue of `agents/ledger/gates/*.md` (REG-PRESERVE)

Date: 2026-08-24 · Repo `/home/opc/ONE/broad-way` · Branch `sklearn` @ `5016e93`

Custody: this file is the SINGLE artifact created by the REG-PRESERVE pass. No other file was
created, modified, or deleted; zero git operations were performed. The nine gate fragments under
`agents/ledger/gates/` were read in full; everything below is the residue that is NOT machine-encoded
in `agents/ledger/gates.yaml` (i.e., everything outside the per-gate 11-key schema
`id / phase / order / owner / inputs / outputs / transforms / touched_by / validated_by / if_changed / findings`).

## Table of contents

- [§1 Surfaces band (07-surfaces.md) — verbatim residue](#1-surfaces-band-07-surfacesmd--verbatim-residue)
  - [1.1 Band charter & single-writer verification method](#11-band-charter--single-writer-verification-method)
  - [1.2 SURFACE OWNERSHIP MAP (all 23 rows)](#12-surface-ownership-map-all-23-rows)
  - [1.3 REFUSAL CONDITION CHECK (incl. F-SURF-3)](#13-refusal-condition-check-incl-f-surf-3)
  - [1.4 COMPUTING-RENDERER FINDINGS F-SURF-1/1b/2 (+mitigating context)/4/5](#14-computing-renderer-findings-f-surf-11b2-mitigating-context45)
- [§2 Band-header & band-level analyses attached to no gate](#2-band-header--band-level-analyses-attached-to-no-gate)
- [§3 IDS DECLARED footer vocabulary vs gates.yaml top-level vocabulary](#3-ids-declared-footer-vocabulary-vs-gatesyaml-top-level-vocabulary)
- [§4 UNVERIFIED / Verification-status sections, consolidated per band](#4-unverified--verification-status-sections-consolidated-per-band)
- [§5 NOTE / OBSERVATION / trailing-comment residue dropped by YAML encoding](#5-note--observation--trailing-comment-residue-dropped-by-yaml-encoding)
- [§6 PROVENANCE](#6-provenance)

---

## §1 Surfaces band (07-surfaces.md) — verbatim residue

Source: `agents/ledger/gates/07-surfaces.md`. The ten gate blocks GATE-SURF-60..69 themselves are
machine-encoded in `gates.yaml` (band 07, flagged: 0 — no finding strings were ever extracted from
this band because its analyses live OUTSIDE the gate blocks, in the prose preserved verbatim below).

### 1.1 Band charter & single-writer verification method

Verbatim (file head, lines 1–5):

```markdown
# surfaces — renderers, reports, ownership (GATE-SURF)

Repo `/home/opc/ONE/broad-way`, branch `sklearn`, HEAD `5016e93`. All `path:line`
citations are CURRENT working-tree (uncommitted WIP included). Gate band 60–69,
phase `surfaces`: every human report/figure surface + its single writer-of-record.
```

The verification method is the three-column audit embodied in the map below: each rendered artifact
is assigned ONE writer-of-record (`module:line symbol`), second-writer candidates are named, and a
verdict is recorded; any second LIVE writer on an existing path triggers the refusal condition of §1.3.

### 1.2 SURFACE OWNERSHIP MAP (all 23 rows)

Verbatim (lines 109–135):

```markdown
## SURFACE OWNERSHIP MAP

| # | rendered artifact | single writer-of-record (module:line symbol) | second-writer candidates | verdict |
| --- | --- | --- | --- | --- |
| 1 | reports/index.md | src/broadway/timeline/walkthrough.py:87 `_write_timeline()` | none | OK |
| 2 | reports/timeline.md | src/broadway/timeline/walkthrough.py:85 `_write_timeline()` | none | OK |
| 3 | reports/results/index.md | src/broadway/reports/results.py:259 `write_results()` (page key `index.md`, :235) | **src/broadway/reports/index.py:16 `render_index()`** — dead, unwired renderer claiming the same conceptual surface (FINDING F-SURF-3) | OK today / LOUD FLAG |
| 4 | reports/results/describe-groups.md | src/broadway/reports/results.py:259 `write_results()` | none | OK |
| 5 | reports/results/normality-diagnostics.md | src/broadway/reports/results.py:259 `write_results()` | none | OK |
| 6 | reports/results/variance-homogeneity.md | src/broadway/reports/results.py:259 `write_results()` | none | OK |
| 7 | reports/results/post-hoc-comparisons.md | src/broadway/reports/results.py:259 `write_results()` | none | OK |
| 8 | reports/results/principal-analysis.md | src/broadway/reports/results.py:259 `write_results()` | none | OK |
| 9 | reports/results/conclusion.md | src/broadway/reports/results.py:259 `write_results()` | none | OK |
| 10 | reports/audit/index.md | src/broadway/reports/audit.py:703 `run()` | none | OK |
| 11 | reports/audit/profile.md | src/broadway/reports/audit.py:695 `run()` | none | OK |
| 12 | reports/audit/transform.md | src/broadway/reports/audit.py:698 `run()` | none | OK |
| 13 | reports/audit/join.md | src/broadway/reports/audit.py:699 `run()` | none | OK |
| 14 | reports/audit/lookup_values.md | src/broadway/reports/audit.py:700 `run()` | none | OK |
| 15 | reports/lineage/graph.md | src/broadway/lineage/module.py:38 `run()` | none | OK |
| 16 | reports/lineage/graph.json | src/broadway/lineage/module.py:27 `run()` | none | OK |
| 17 | reports/figures/describe.png | src/broadway/stats/describe.py:117 `plot_describe_figures()` (sole caller timeline/runners.py:143) | recomputes grouping instead of reading persisted summary (FINDING F-SURF-2) | OK / purity flag |
| 18 | reports/figures/normality_qq.png | src/broadway/discover/qq.py:478 `_plot_qq_joint()` (sole caller timeline/runners.py:210) | computes standardization+probplot fit inside plot module (FINDING F-SURF-1); shares savefig helper `_plot_raw_log_pairs` (qq.py:356) with row 21 | OK / purity flag |
| 19 | reports/figures/numeric_qq_{n}.png | src/broadway/discover/qq.py:297 `plot_numeric_qq()` (via discover/module.py:78) | none | OK |
| 20 | reports/figures/numeric_dist_{n}.png | src/broadway/discover/qq.py:514 `plot_numeric_qq()` | none | OK |
| 21 | reports/figures/numeric_qq_log_{n}.png | src/broadway/discover/qq.py:356 `_plot_raw_log_pairs()` under `plot_numeric_qq()` | same helper also renders normality_qq.png when show_log=True (qq.py:409) — different out_path, shared code path | OK / watch |
| 22 | reports/figures/numeric_diagnostics.png | src/broadway/discover/qq.py:562 `plot_numeric_qq()` | none | OK |
| 23 | experiments/results/**/*.csv | ad-hoc experiment scripts (convention anchored at .gitignore:19; e.g. experiments/fare_prediction/02_filtered_profile.py:46, experiments/mlflow/_common.py:51) | no central writer — convention enforced only by .gitignore, untested | OK / UNVERIFIED enforcement |
```

### 1.3 REFUSAL CONDITION CHECK (incl. F-SURF-3)

Verbatim (lines 137–148). F-SURF-3 has no standalone block — it lives here:

```markdown
REFUSAL CONDITION CHECK: no concrete artifact path has two LIVE writers today.
Nearest misses, flagged loudly:
- F-SURF-3 (two renderers, one concept): `reports/index.py:16 render_index()` vs
  `reports/results.py:146 _render_index()` both claim the "results index" surface.
  The former is production-dead (imported only by tests/test_reports.py:11) AND
  mismatched with reality: it probes `results/{describe,normality,levene,…}.md`
  names from configs/flow/stats_sequence.yaml (reports/index.py:12-13), while the
  real pages are slugified labels (`describe-groups.md`, …). If anyone ever wires
  it, reports/results/index.md gets TWO writers → refusal condition. Delete or
  align before reuse.
- Shared savefig helper `_plot_raw_log_pairs` serves two distinct surfaces (rows
  18/21) through different call chains — one refactor away from a collision.
```

### 1.4 COMPUTING-RENDERER FINDINGS F-SURF-1/1b/2 (+mitigating context)/4/5

Verbatim (lines 150–177):

```markdown
### COMPUTING-RENDERER FINDINGS (contract violation scan)

- F-SURF-1 (plot module computes): discover/qq.py:436-437 `_plot_qq_joint()` does
  per-group z-standardization and `scipy.stats.probplot` fitting from RAW arrays
  instead of reading persisted normality evidence.
- F-SURF-1b (render-time stat not persisted): timeline/runners.py:204-208
  `run_normality()` computes pooled `scipy.stats.skew` to pick show_log for
  normality_qq.png; that choice is NOT stored — NormalityEvidence
  (timeline/evidence.py:14-19) carries only groups/figure/standardization, so the
  figure's raw-vs-log layout is unverifiable from artifacts.
- F-SURF-2 (duplicate computation): stats/describe.py:107 `plot_describe_figures()`
  re-derives groups via `build_declared_groups` although the caller already
  computed and persisted GroupSummary (runners.py:130-146).
- Mitigating context: pure report renderers (reports/results.py, reports/timeline.py,
  reports/audit.py docstring contract) format persisted evidence only; the
  computing happens in step-runner/plot modules upstream, not in the markdown
  renderers themselves.
- F-SURF-4 (.svg coverage gap): test_surface_integrity enforces caps for .html
  (:59) and .png (:68) only; NO .svg/.jpg cap exists, while link-resolution
  covers all extensions. An .svg dropped into reports/figures would be git-tracked
  (*.png rule :23 does not match it; reports/ has no blanket ignore) with NO size
  gate. Answer to scope question: does the suite cover .svg? — NO.
- F-SURF-5 (toggles): configs/step/viz.yaml drives every figure FILENAME
  (`qq_figure`, `qq_log_figure`, `dist_figure`, `describe_figure`,
  `normality_figure`, `diagnostics.figure`) and overlay switches
  (`qq_zones.enabled`, `qq_markers.enabled`, `diagnostics.annotate`), but no knob
  suppresses emitting a figure outright; describe/normality/discover figures
  regenerate unconditionally whenever their step runs.
```

The band's `## IDS DECLARED` footer additionally registers the finding vocabulary
`F-SURF-1 · F-SURF-1b · F-SURF-2 · F-SURF-3 · F-SURF-4 · F-SURF-5` over gates
GATE-SURF-60..69 (see §3).

---

## §2 Band-header & band-level analyses attached to no gate

Header regions and mid-file band-level sections of the nine fragments carry analyses that belong to
no single gate and therefore never entered `gates.yaml`. Captured verbatim, per band.

### 2.1 Known: 03-features contract-drift proof (header, lines 5–8)

```markdown
Contract drift recorded up front:
- FINDING: contract names `src/broadway/features/builder.py` — no such file exists; the actual owner is `src/broadway/features/builders.py`.
- FINDING: contract symbol `build_distance_features` does not exist anywhere in the tree; the distance/log1p transform actually lives in the shared registry `BUILDERS["log_distance"]`.
- FINDING: contract symbol `validate_engineered_schema` does not exist anywhere in the tree; the actual schema trio is `build_engineered_schema` / `engineered_schema_for` / `validate_engineered_frame`.
```

### 2.2 Further header-region and band-level captures (beyond the known trio)

**(a) 01-ingest — pipeline duality statement (header, lines 3–5):**

```markdown
Branch sklearn @ 5016e93; line numbers cited from the CURRENT working tree (carries unrelated WIP).
Two ingest pipelines exist side by side: LEGACY = `project/etl/process.py` (taxi, hardcoded glob),
CONTRACT = `broadway.data.loader` → `broadway.etl.module.run` → `broadway.data.cleaner.canonicalize`.
```

**(b) 02-etl-lookup — provenance/WIP caveat (header, line 3):**

```markdown
Branch sklearn @ 5016e93 claimed by contract; worktree carries unrelated WIP — all lines below cite the CURRENT working tree (verified by opening each file). Git state NOT re-verified (zero git operations mandated).
```

**(c) 03-features — footer findings digest F1–F6 (final line of `## IDS DECLARED`; not machine-encoded — verified absent from gates.yaml by probe):**

```markdown
Findings summary: (F1) contract file/symbol drift — builder.py/build_distance_features/validate_engineered_schema absent, actual owners builders.py + generic.py trio; (F2) log1p transform numerically pinned only via double-log chaining test, no single-hop golden test; (F3) TargetEncoding.fit y=None raise (transformers.py:52-55) unpinned by any test; (F4) ARTIFACT-TRAIN-PARQUET write + features run() have no direct test node id (only _load_split imported); (F5) write-side schema check is unordered while read-side is ordered=True — order drift survives the write gate; (F6) entire engineered read contract (validate_engineered_frame et al.) tested only transitively, zero direct node ids.
```

**(d) 04-training — band scope + fix pointers T-BUG-1/T-BUG-2 (header, lines 3–4):**

```markdown
Branch sklearn @ 5016e93; line numbers cited from the CURRENT working tree (carries unrelated WIP).
Band 30–39 maps every gate in src/broadway/training/ (incl. the project/ glue read path), src/broadway/evaluate/, and the MLflow logging/champion-manifest surface (scripts/check_champion_manifest.sh). T-BUG-1 = HC3 self-fit + idempotency pin (agents/ledger/FIXES.md:91); T-BUG-2 = champion persist⇒promote terminal ordering (FIXES.md:98).
```

**(e) 05-stats — purity-law sweep verdict (header, line 3):**

```markdown
Branch sklearn @ 5016e93 claimed by contract; worktree carries unrelated WIP — all lines below cite the CURRENT working tree (verified by opening each file). Git state NOT re-verified (zero git operations mandated). Purity-law sweep verdict for `src/broadway/stats/`: ZERO env reads (no `os.environ`/`getenv` anywhere in the package); NO direct config reads (every `PipelineConfig`/threshold arrives as a function parameter); IO exists ONLY in the two step-entry legs (`module.run`, `describe.run`), the plan JSON helpers (`plan.py`), and three `fig.savefig` plot writes — each violation of the strict "no IO" reading is flagged inline as `# FINDING`.
```

**(f) 06-timeline — provenance/WIP caveat (header, line 3):**

```markdown
Branch sklearn, HEAD 5016e93. All `file:line` cites are the CURRENT working tree (worktree carries unrelated uncommitted WIP elsewhere; nothing here was modified by this mapping).
```

**(g) 08-config — structural fact: NO environment inheritance chain (header, lines 3–9):**

```markdown
Mapped on branch sklearn @ HEAD 5016e93 against the CURRENT working tree
(worktree also carries unrelated uncommitted WIP; every `path:line` below was
read from the live tree, not from history). Band 70–79, phase `config-schema`.
Structural fact up front: there is NO dev→staging→production inheritance chain
— `load_config` merges EXACTLY ONE `configs/environment/<name>.yaml`
(src/broadway/config/loader.py:193); "layering" is per-file values plus
`${VAR}` env-var indirection resolved by `os.path.expandvars`, nothing more.
```

**(h) 08-config — `## Cross-gate enforcement surface` (mid-file section attached to no gate, lines 179–186):**

```markdown
## Cross-gate enforcement surface

- `scripts/run_local_ci.sh:47-51` (`run configs …`) is the ONLY continuous
  gate that round-trips real shipped configs through load_config (train ×
  every configs/experiment/*.yaml against dataset=test); ci.yml delegates to
  this script per D17b (DECISIONS.md:129-133). It does NOT touch
  staging/production environments — the `${VAR}` path of GATE-CFG-71 runs
  green in CI by construction and is exercised by no test.
```

**(i) 09-infra — meta-gate scope + field legend (header, lines 3–13; the legend defines semantics used by ALL bands' YAML encodings):**

```markdown
Meta-gates guarding the platform itself: the local/CI tier battery, branch-parity
custody, ship law, E2E determinism comparator, review-tier classifier, the
machine-local pre-push chain, and the ci.yml delegation. Provenance: repo
`/home/opc/ONE/broad-way`, branch `sklearn`, HEAD `5016e93`; **all file:line
citations are CURRENT working-tree** (worktree carries unrelated uncommitted WIP,
e.g. modified `agents/ledger/DECISIONS.md`, untracked `project/tests/*`).

Legend: `touched_by` = governance actors/surfaces that own or amend the gate;
`if_changed` = inputs whose modification makes this gate's verdict stale or that
feed it. `validated_by` lists REAL pytest node ids only; `[]` marks an
unexercised guard and carries a `# FINDING:` line.
```

(07-surfaces' charter is captured in §1.1 and is not double-counted here.)

---

## §3 IDS DECLARED footer vocabulary vs gates.yaml top-level vocabulary

Method: every `CFG-*` / `ARTIFACT-*` / `VOCAB-*` token declared in each fragment's `## IDS DECLARED`
footer was extracted mechanically and diffed against the 29-entry top-level `vocabulary:` block of
`gates.yaml`. Gate ids (`GATE-*`) are identifiers of encoded gates, not vocabulary, and are excluded.

| Fragment | Footer vocabulary tokens declared | Additions required (not in gates.yaml vocabulary) |
| --- | --- | --- |
| 01-ingest.md | 9 (CFG-TAXI-PROJECT, CFG-ETL-STEP, CFG-DATASET-CONTRACT, ARTIFACT-RAW-PARQUET, ARTIFACT-RAW-FRAME, ARTIFACT-SELECTED-FRAME, ARTIFACT-CANONICAL-FRAME, ARTIFACT-CLEAN-EVIDENCE, ARTIFACT-COERCION-AUDIT) | **none** |
| 02-etl-lookup.md | 7 distinct (ARTIFACT-CANONICAL-PARQUET, ARTIFACT-JOIN-AUDIT, ARTIFACT-LOOKUP-VALUE-AUDIT, ARTIFACT-COERCION-AUDIT, ARTIFACT-STRUCTURAL-CLEAN, CFG-ENV-PROCESSED-SUBDIR, CFG-ETL-SPLIT-FILES) | **none** |
| 03-features.md | gate ids only + 1 repeated token (ARTIFACT-TRAIN-PARQUET, inside the findings digest quoted in §2.2(c)) | **none** |
| 04-training.md | 7 (CFG-STEP-TRAIN, CFG-STEP-EVALUATE, CFG-DATA-MODE, ARTIFACT-TRAINING-RESULT, ARTIFACT-EVALUATION-RESULT, ARTIFACT-CHAMPION-REGISTRY, ARTIFACT-SAMPLE-CACHE) | **none** |
| 05-stats.md | 5 (ARTIFACT-STATS-PLAN, ARTIFACT-DESCRIBE-SUMMARY, CFG-STATS-FLOOR, CFG-WALKTHROUGH-THRESHOLDS, VOCAB-DECLARED-GROUPS-ABSENT) | **none** |
| 06-timeline.md | gate ids only | **none** |
| 07-surfaces.md | gate ids + F-SURF finding names only | **none** |
| 08-config.md | gate ids only | **none** |
| 09-infra.md | gate ids only | **none** |

**Result of the mandated footer diff: ZERO additions** — every vocabulary token declared in a
`## IDS DECLARED` footer is already present in the `gates.yaml` top-level `vocabulary:` block.

**Adjacent observation (recorded so the knowledge survives the fragments; outside the strict letter
of this category):** four tokens referenced INSIDE gate bodies of 03-features.md — and indeed present
in the encoded `inputs:`/`outputs:` lists of GATE-FEAT-20/23/25 in gates.yaml — are nevertheless
absent from that same top-level vocabulary list: `ARTIFACT-SPLIT`, `ARTIFACT-VAL-PARQUET`,
`ARTIFACT-PIPELINE-PICKLE`, `CFG-FEATURES`. If the vocabulary block is ever regenerated strictly from
footers, these four would be lost.

---

## §4 UNVERIFIED / Verification-status sections, consolidated per band

Only four of the nine fragments carry such a section; the other five (03-features, 06-timeline,
07-surfaces, 08-config, 09-infra) end at their IDS DECLARED footers and carry none. All four are
preserved verbatim below, under per-band headings.

### Band 01 — ingest (Verification status paragraph, 01-ingest.md final line)

```markdown
Verification status: all owner symbols, line numbers, and validated_by node ids above were confirmed by opening the files in this working tree (branch sklearn @ 5016e93). Library versions read from the live environment via importlib.metadata (assumed to be the project env; NOT cross-checked against lockfile pins). Test PASS status not executed — existence verified by grep/read only.
```

### Band 02 — etl-lookup (`## UNVERIFIED`, 4 bullets, 02-etl-lookup.md tail)

```markdown
## UNVERIFIED

- Branch/HEAD (sklearn @ 5016e93): taken from the contract; NOT verified — git operations prohibited by this mandate.
- ARTIFACT-COERCION-AUDIT persistence branch (src/broadway/etl/module.py:145-156): no test asserts the coercion-audit JSON or its lineage record; coverage ends at the collector level (parse_numeric/canonicalize).
- Post-write reload validation: asserted ABSENT after full-tree search (read_parquet/scan_parquet/reload); whether a downstream-band gate owns reload verification was not confirmed here.
- Sentinel matching in GATE-ETL-14 against post-merge pandas-inferred dtypes (string sentinel vs float column): code-read only, no test pins that interaction.
```

### Band 04 — training-eval (Verification status paragraph, 04-training.md final line)

```markdown
Verification status: all owner symbols, line numbers, and validated_by node ids above were confirmed by opening the files in this working tree (branch sklearn @ 5016e93, WIP untouched). Library versions read live via importlib.metadata in the project env (uv run). Test PASS status NOT executed — existence verified by grep/read of tests/ only. Parametrize-id enumeration of tests/test_mlflow_utils_unreachable.py::test_unreachable_http_store_raises_clear_error incomplete (only the 'wrapped-mlflow-exception' param sighted); cited without bracket id for that reason. MLflow runtime behaviors quoted from code comments (duplicate LocalArtifactDatasetSource registration, cause-less connection refusals) were taken on trust from the source annotations, not reproduced live — UNVERIFIED.
```

### Band 05 — stats (`## UNVERIFIED`, 5 bullets, 05-stats.md tail)

```markdown
## UNVERIFIED

- Branch/HEAD (sklearn @ 5016e93): taken from the contract; NOT verified — git operations prohibited by this mandate.
- Purity sweep covers src/broadway/stats/ source lines only (grep: environ/getenv/open/read_parquet/write_text/savefig/mkdir): THIRD-PARTY import-time behavior (matplotlib/seaborn/statsmodels/pingouin/lightgbm reading env or files inside their own import) was not audited.
- baseline.train_lgbm determinism: no seed pinned and no production caller exists; LightGBM default settings are largely deterministic but thread-count float-summation drift is untested here.
- Golden-absence proof is glob+grep based (no tests/*golden*, no "golden" token under stats-relevant tests); a golden reference living outside tests/ (docs/notebooks) was not ruled out.
- tests/test_walkthrough.py::test_run_describe_flags_imbalance_and_absent_groups validates the walkthrough FLAGGING branch, not groups.py directly — GATE-STATS-42's walkthrough-leg coverage is therefore indirect (direct pins exist only for the stats and baseline legs).
```

### Bands with NO such section

03-features, 06-timeline, 07-surfaces, 08-config, 09-infra — none present (verified by full read).

---

## §5 NOTE / OBSERVATION / trailing-comment residue dropped by YAML encoding

The assembler captured every `# FINDING:` comment into the per-gate `findings` key
(`meta.counts.finding_strings = 78`) but dropped `# NOTE` / `# OBSERVATION` / bare trailing notes and
the two `outputs:` trailing comments itemized by the REG-YAML builder. Twelve items total, verbatim:

| # | Gate | Fragment:line | Kind | Verbatim dropped content |
| --- | --- | --- | --- | --- |
| 1 | GATE-INGEST-01 | 01-ingest.md:17 | `# NOTE` | validated_by is indirect (through process_data); no direct test exists for this symbol. |
| 2 | GATE-INGEST-03 | 01-ingest.md:39 | `# NOTE` | touched_by is empty because etl/module.py imports no third-party library at file level (stdlib + broadway only); it operates duck-typed on pandas frames returned by load_with_audit. |
| 3 | GATE-FEAT-21 | 03-features.md:29 | bare trailing note | All four raise sites are pinned by exact-message pytest.raises matches; none dead. |
| 4 | GATE-FEAT-23 | 03-features.md:47 | inline `# NOTE` on touched_by | pipeline.py imports NO sklearn; sklearn enters only via transformers.py (sklearn.base BaseEstimator/TransformerMixin) |
| 5 | GATE-FEAT-24 | 03-features.md:61 | `# OBSERVATION` | (not a dead validation): load_custom_builders(cfg.builder_module) runs INSIDE every transform call (pipeline.py:46) — train and val each re-resolve it; harmless today because importlib caches, but it is per-call work on the hot path. |
| 6 | GATE-TRAIN-30 | 04-training.md:16 | `# NOTE` | baseline-improvement logging inside the run (:121-126) is deliberately silent when load_persisted returns None or the metric is absent — no warning, no metric; accepted design, unrecorded anywhere outside this file. |
| 7 | GATE-TRAIN-31 | 04-training.md:28 | `# NOTE` | val-absent→split fallback is deliberate (docstring module.py:1-6 for evaluate's twin); the asymmetry means train may self-split while evaluate HARD-REQUIRES the val file (GATE-TRAIN-35) — same config can satisfy one step and crash the other. |
| 8 | GATE-TRAIN-32 | 04-training.md:40 | `# NOTE` | the stratified-cache twin (load_stratified_sample :182-203, generate_sample_cache :206-228) DOES carry a params_hash meta check but only WARNS on drift (:195-201) — cache/staleness is report-only, never a gate. |
| 9 | GATE-TRAIN-37 | 04-training.md:97 | `# NOTE` | touched_by empty — promotion.py imports nothing (stdlib-free pure logic, :1-5). |
| 10 | GATE-TRAIN-38 | 04-training.md:108 | `# NOTE` | touched_by lists pandas only — statsmodels@0.14.6 is duck-typed, NOT imported (no import statement in robust.py); the TypeError gate is the only defense against non-statsmodels input, and it is exercised (tests/test_stats_robust.py:98-103 _NoRobustCovariance case). |
| 11 | GATE-SURF-60 | 07-surfaces.md:13 | `outputs:` trailing comment | `# concrete artifact` |
| 12 | GATE-SURF-67 | 07-surfaces.md:83 | `outputs:` trailing comment | `# read-only gate; asserts on reports/** but writes nothing (docstring :8)` |

This sweep confirms the REG-YAML builder's itemization exactly (INGEST-01/03, FEAT-21/23/24,
TRAIN-30/31/32/37/38, SURF-60/67); no further dropped note/comment content exists in the nine fragments.

Encoder-behavior boundary, verified by probe against `gates.yaml` so future readers know what does
NOT need re-preserving: content that SURVIVED encoding includes owner-line trailing comments
(e.g. GATE-SURF-60 owner `…# renders via reports/index.py:61 render_dashboard()`), and all
quoted-string material such as GATE-SURF-66 transforms `NOTE no toggle suppresses emitting a figure —
FINDING F-SURF-5`, GATE-SURF-67 if_changed `.svg currently UNCOVERED, see FINDING F-SURF-4`, and
GATE-SURF-69 if_changed `violations = FINDINGS F-SURF-1/1b/2 below`. Content dropped as pure noise:
id-line band-range markers (`# NN = 60..69`, surfaces band) — formatting only, no analytic content.

---

## §6 PROVENANCE

- **Source fragments (read in full, verbatim excerpts above):**
  - `agents/ledger/gates/01-ingest.md`
  - `agents/ledger/gates/02-etl-lookup.md`
  - `agents/ledger/gates/03-features.md`
  - `agents/ledger/gates/04-training.md`
  - `agents/ledger/gates/05-stats.md`
  - `agents/ledger/gates/06-timeline.md`
  - `agents/ledger/gates/07-surfaces.md`
  - `agents/ledger/gates/08-config.md`
  - `agents/ledger/gates/09-infra.md`
- **Stamp:** branch `sklearn` @ HEAD `5016e93` (as claimed by the preservation contract and by every
  fragment header; consistent with `gates.yaml` `meta.head: 5016e93`). Zero git operations were
  performed during this capture.
- **SSOT statement:** the gate DATA (per-gate 11-key schema: id, phase, order, owner, inputs,
  outputs, transforms, touched_by, validated_by, if_changed, findings) lives in
  `agents/ledger/gates.yaml`, the machine-assembled single source of truth; THIS file holds only the
  non-encodable residue — surface-ownership analysis, refusal conditions, band-level findings and
  verdicts, vocabulary-diff results, verification caveats, and dropped notes/comments — that cannot
  be represented in that schema without distortion.
- **Deletion eligibility:** the nine `agents/ledger/gates/*.md` fragments become eligible for
  deletion once (a) THIS file exists (it now does) and (b) the HPO band integration has landed.
  Until both hold, the fragments remain the citation-of-record for anything quoted above.
