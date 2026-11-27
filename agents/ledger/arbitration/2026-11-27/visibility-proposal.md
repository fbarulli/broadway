# Proposal — pipeline evidence visibility (audit finding #10)

Status: **fourth REVISE addressed, awaiting re-review. Not approved.** Author: build agent.
Base: `time` @ `114ef4d` (the commit is being amended; unpushed).

## Problem

`configs/step/eda.yaml` published EDA's decision record and evidence into
`artifacts/reports/`, which `.gitignore` ignores wholesale. The dashboard
(`src/broadway/reports/experiments_dashboard.py`) discovers only experiment
series: a folder under the experiments root with numbered `NN_*.py` scripts.
A pipeline step is not a series, so a fresh checkout showed no EDA evidence
even though the pipeline had published it.

## What changes

| File | Change |
|---|---|
| `configs/step/eda.yaml:5` | Reverted to the base default `output_dir: artifacts/reports`. |
| `project/config/step/eda.yaml` (new) | Project binding: `output_dir: project/experiments/results/pipeline/eda` only. It is a partial step overlay, not a copy of the knobs. |
| `src/broadway/config/loader.py:109-135` | `_load_yaml` now layers a project `step/*.yaml` over its base file with the module's own `_deep_merge`, so the project file names only the knobs it binds (nested partials merge; lists replace). Dataset/experiment/analysis overlays keep their existing complete-contract semantics. |
| `src/broadway/config/schema.py:405-411` | `EdaStep.output_path` anchors a relative `output_dir` at `repo_root()` (same pattern as `observability/profiling.py:230`), so a run from another cwd cannot write evidence the dashboard never reads. |
| `src/broadway/reports/pipeline_evidence.py` (new, 168 lines) | `PipelineRecord.from_path` validates each `*_decision.json` with the canonical `Decision.model_validate_json`; invalid/inverted/unknown-direction records are skipped with a full-traceback log, once per path per process. `PipelineEvidenceSection` scans `<experiments root>/results/pipeline/*/` and renders step + `provenance.dataset`, metric, value [95%], rule (`>` for `max`, `<` for `min`), null band, n, verdict, figure/sidecar/record links. No new route: the existing `/results` static mount serves them. |
| `src/broadway/reports/experiments_dashboard.py:40,674-677,691,979-990` | Imports the section, adds four CSS rules, calls it once in `_render_dashboard_page`, lets `/` render when no experiment series exist, and returns 404 for any supplied focus that is not a discovered series (including on a no-series root). |
| `src/broadway/eda/module.py:213` | Adds `dataset` to the published provenance, so a row cannot silently be another dataset's. |
| `src/broadway/eda/artifacts.py:8-12` | Corrects the false claim that the record provenance points at the sidecar; the sidecar is the per-feature evidence written next to the singular record. |
| `.gitignore:30-31` | Adds `!project/experiments/results/**/*_feature_decisions.json`, committing the per-feature evidence sidecar (30 KB, smaller than one PNG). `eda.html` stays ignored. |
| `scripts/check_experiment_outputs.py:18-24,65-101,190-290` | No `project` import at module load. `resolve_project_layout()` imports `project.paths` lazily inside `main()` and tolerates `ImportError`/`FileNotFoundError`/`ValueError`. `ExperimentOutputAudit.series()` returns `[]` for an absent root. `PipelineDecisionAudit` accepts a published `<layout results>/pipeline/<step>/*_decision.json` only when the canonical `Decision` validates. `eda` is enforced whenever the project experiments root exists; a vanished results tree fails the missing record instead of skipping. |
| `agents/ledger/gates.yaml` | GATE-SURF-66 owns the new module/test and the committed pipeline evidence; old `artifacts/reports/eda.html` strings replaced. `DIGEST.md` + `index/by_artifact.yaml` re-rendered with `agents/tools/render_gates.py`. |
| `tests/test_pipeline_evidence.py` (new, 127 lines) | One public-behavior test renders the dashboard root page for a pipeline-only checkout, asserts the row/dataset/links/comparators, asserts `focus="../.."` and `focus="alpha"` are 404, and asserts four malformed records are skipped and logged once per path across two requests. |
| `tests/test_experiment_outputs.py` | Pipeline audit tests pass an explicit results root; a new test pins that only a canonical published record flips the status. |
| `tests/test_loader.py` | Pins that a partial project step overlay overrides one knob and keeps the base knobs. |
| `project/tests/test_dashboard.py` | Pins that the EDA writer path equals `load_project_paths().experiments / "results" / "pipeline" / "eda"`, so changing the layout root cannot silently desynchronize the dashboard and the writer. |
| `README.md:329` | EDA output row now says committed decision/evidence + local ignored `eda.html`. |
| Regenerated artifacts under `project/experiments/results/pipeline/eda/` | Produced by the real EDA step at the new path; the sidecar is now tracked. |

What it replaces: the EDA step's output location only. No runtime capability is
removed. The old ignored `artifacts/reports/` copy is left in place (not deleted).

## Response to each required revision

1. **Layering + anchoring.** Base config reverted; project overlay added; the
   loader now layers partial step overlays; `EdaStep.output_path` anchors at
   `repo_root()`. Verified in-process: `output_dir` is the project path,
   `output_path` is the absolute repo path, and a base knob (`alpha=0.05`)
   survives the partial overlay. The real EDA run then logged the absolute
   destination.
2. **Canonical parser.** `PipelineRecord.from_path` no longer parses ad-hoc
   fields. It calls `Decision.model_validate_json`, catches `OSError` /
   `ValidationError`, and logs the full traceback. `[]`, `1e999`, an inverted
   interval, and an unknown direction all fail validation and are skipped; the
   comparator is read from the validated `Direction` literal.
3. **Evidence gap.** The sidecar is committed via the `.gitignore` negation and
   linked from the row ("feature decisions"). `dataset` is in provenance.
   `eda/artifacts.py` no longer claims provenance points at the sidecar.
   The earlier false claim is corrected below: the writer also added
   `unit: null` and `min_sample_size: 100`, not only `min_sample_size`.
4. **Ratchet.** `eda` is enforced; the gate reuses the canonical `Decision`;
   the ledger registry now names the new module/test and the new path, and the
   rendered views were regenerated. The gate prints no `pipeline/eda` note and
   exits 0.
5. **Real-behavior test.** The test calls `ui.index(focus="")` (the actual
   `/` route), so deleting the one-line dashboard call makes it fail. It uses
   contract-valid `Decision` fixtures and asserts malformed records do not
   raise.

### Second review

1. **Focus safety.** `index()` now resolves the focus and returns 404 whenever
   the resolved focus is not a discovered series, unless there is no focus at
   all on a no-series root (the valid pipeline-only page). Verified:
   `focus=""` → 200, `focus="../.."` → 404, `focus="alpha"` → 404 on a
   no-series root. The test pins all three.
2. **Writer↔dashboard path agreement.** `project/tests/test_dashboard.py` now
   asserts `load_config("eda", dataset="taxi").eda.output_path.parent ==
   load_project_paths().experiments / "results" / "pipeline" / "eda"`.
   The gate resolves the same layout lazily at run time (see third review), so
   there is no import-time project coupling and no second literal.

### Third review — platform-only checkout

1. **Lazy, tolerant project resolution.** The module-level `project.paths`
   import is gone. `resolve_project_layout()` imports it inside `main()` and
   catches `ImportError`, `FileNotFoundError`, `ValueError`. Importing the gate
   no longer executes `project/__init__.py`, so it cannot set
   `BROADWAY_CONFIG_OVERLAY_DIR`.
2. **Experiments root guard.** `ExperimentOutputAudit.series()` returns `[]`
   when the root is `None` or not a directory; no `iterdir()` crash.
3. **Conditional pipeline enforcement.** `eda` (the only step in
   `PROJECT_EVIDENCE_STEPS`) is enforced whenever the resolved project
   experiments root exists (fourth review corrected this from the evidence
   directory). Without a project checkout, the gate prints
   `note pipeline/eda: decision record not enforced (project results root
   unavailable)` and exits 0. With a project checkout, a missing results tree
   or record prints `FAIL pipeline/eda: missing decision record` and exits 1.
4. **No import-time coupling.** Loading the gate in a fresh process with
   `BROADWAY_CONFIG_OVERLAY_DIR` unset: `project` is not imported and the env
   stays unset. Running it with a meta-path finder blocking `project` completes
   with exit 0 and the skip note.

### Fourth review — vanished results tree must fail

1. **Enforcement key.** `_enforced_pipeline_steps()` now keys on
   `layout.experiments is not None and layout.experiments.is_dir()`, not on the
   evidence directory. A project checkout whose `project/experiments/results`
   vanished still enforces `eda`, so `main()` prints
   `FAIL pipeline/eda: missing decision record` and exits 1.
2. **Pin.** `tests/test_experiment_outputs.py` now asserts:
   - a layout with an existing experiments root and a missing results root still
     includes `eda` in `_enforced_pipeline_steps()`;
   - a stubbed `main()` over that tree returns 1 with the missing-record FAIL;
   - a stubbed unavailable layout returns 0 with the skip note.
   This also guards the lazy-import regression: if the gate went back to a
   module-level `project` import, the unavailable-layout path would fail.

Cheap/safe extras addressed:
- `_load_yaml` now uses the module's `_deep_merge` for step overlays, so a
  nested partial merges instead of replacing a block and hard-failing; pinned
  by `tests/test_loader.py` with a `decisions.features.action` override.
- The row's step cell renders `provenance.dataset`; a record without it renders
  a plain `<td>step</td>`, pinned by the pipeline-evidence test.
- A permanently malformed record logs its full traceback once per path per
  process (`_LOGGED_UNREADABLE`), not on every `/` request.
- `Decision` float fields still accept infinity: `decision.py` is a separate
  no-touch unit in the parent contract, so this is recorded as a pre-existing
  model gap rather than changed here. The dashboard consumes only validated
  records and renders finite values from the committed pipeline record.

GATE-SURF-66 now owns the new module/test/artifacts (a new row would require
renumbering the dense global order and updating `meta.counts`, which is
invasive for this unit); README corrected; the pipeline-only `/` 404 is fixed.
`configure_from_env`'s first-wins `/results` mount is unchanged and documented
under risks.

## Blast radius

- **Loader:** step configs with a project overlay now layer instead of replace.
  No project `step/*.yaml` existed before this change, so no existing config
  behavior moves; dataset/experiment/analysis overlay semantics are untouched.
- **Config consumers:** `EdaStep.output_path` is read by `EdaRunner._writer`
  (`eda/module.py:188`), so report, tables, figures, decisions and sidecar move
  together and are cwd-independent. Tests that override `output_dir` with an
  absolute temp path (`test_eda.py:115`, `test_eda_decisions.py:227`,
  `test_eda_paths.py:30`) are unaffected.
- **Dashboard:** one import, one call, one tightened focus guard. No route or
  mount added. Existing series behavior is unchanged; an explicit unknown focus
  is 404 on both series and no-series roots.
- **Gate:** `check_experiment_outputs.py` imports `Decision` only; the project
  layout is resolved lazily in `main()`. On a project checkout the EDA record is
  enforced (missing → exit 1); on a data-agnostic checkout the project surfaces
  are skipped (exit 0). Existing code-based pipeline entries still pass.
- **Ledger:** GATE-SURF-66 owner/inputs/outputs/validated_by updated; rendered
  DIGEST/index are regenerated. The registry no longer names a retired path.
- **Tests:** one new dashboard test, one new audit test, one new loader test,
  one new project path-agreement test. No tracked file deleted.

## Evidence (verified against the real artifact)

1. Real EDA run after the layering change (repo-root run; logged destination is
   absolute):

   ```
   $ .venv/bin/python -m project.cli eda --dataset taxi --environment development
   eda: report written to /home/opc/ONE/broad-way/project/experiments/results/pipeline/eda/eda.html in 37.65s
   eda: rows=8545665 sampled=50000 missing_cells=1090 constant=0 duplicates=0
        outlier_columns=6 features_tested=7 decisions={'act': 6, 'investigate': 1} compare=available
   ```

2. Config layering + anchor, verified in-process:

   ```
   output_dir: project/experiments/results/pipeline/eda
   output_path: /home/opc/ONE/broad-way/project/experiments/results/pipeline/eda/eda.html
   absolute: True
   alpha (base knob): 0.05
   ```

3. Regeneration vs the old ignored record (`artifacts/reports/eda_decision.json`)
   — the writer's newer fields plus the new provenance key are the only
   differences:

   ```
   added keys: ['min_sample_size', 'unit']   # unit: null, min_sample_size: 100
   evidence.provenance.dataset: 'taxi'
   every measured value/interval/threshold/null band/segment identical
   ```

4. Files at the new path, and commit-ability:

   ```
   eda_comparison.csv  eda_correlations.csv  eda_correlations.png
   eda_decision.json   eda_effects.csv       eda_effects.png
   eda_feature_decisions.json (tracked now)
   eda.html (ignored)
   eda_outliers.csv    eda_outliers.png
   ```

   `git check-ignore`: all CSVs, PNGs, `*_decision.json` and the
   `*_feature_decisions.json` sidecar are TRACKABLE; only `eda.html` is IGNORED.
   No tracked `eda_decisions.json` (`git ls-files | grep` → empty).

5. HTML references: `eda.html` contains 6 `data:image/png;base64,` payloads and
   **0** relative asset references — self-contained, so the move cannot break it.

6. Focus safety on a no-series root (the pipeline-only case):

   ```
   focus=''     -> 200 (HTMLResponse, pipeline evidence rendered)
   focus='../..' -> 404 (PlainTextResponse)
   focus='alpha' -> 404 (PlainTextResponse)
   ```

7. Writer↔dashboard path agreement:

   ```
   eda output parent: /home/opc/ONE/broad-way/project/experiments/results/pipeline/eda
   layout expected:   /home/opc/ONE/broad-way/project/experiments/results/pipeline/eda
   agreement: True
   ```

8. Gate checkout-shape probes:

   ```
   import (fresh process, project blocked): project_imported=False, overlay=None
   runpy, project unimportable:              exit 0  (skip note)
   runpy, layout.yaml absent:                exit 0  (skip note)
   absent experiments root:                  series=[]; exit 0 (no iterdir crash)
   project checkout, results root absent:    exit 1  (FAIL pipeline/eda: missing decision record)
   normal checkout:                          exit 0  (eda enforced, no eda note)
   record temporarily moved:                 exit 1  (FAIL pipeline/eda: missing decision record)
   ```

9. Sample rendered row (real dashboard HTML):

   ```html
   <tr><td>eda<br><span class="dataset">taxi</span></td><td>association_fare_amount</td>
   <td>0.9396 [0.9367, 0.9425]</td><td>&gt; 0.0500</td><td>0.0089</td>
   <td>50,000</td><td class="verdict">act</td>
   <td><a href="/results/pipeline/eda/eda_correlations.png">correlations</a> ·
       <a href="/results/pipeline/eda/eda_effects.png">effects</a> ·
       <a href="/results/pipeline/eda/eda_outliers.png">outliers</a> ·
       <a href="/results/pipeline/eda/eda_feature_decisions.json">feature decisions</a> ·
       <a href="/results/pipeline/eda/eda_decision.json">record</a></td></tr>
   ```

10. Checks run:

   ```
   pytest tests/test_pipeline_evidence.py tests/test_experiments_ui.py tests/test_eda.py \
          tests/test_eda_decisions.py tests/test_gate_registry.py tests/test_experiment_outputs.py \
          tests/test_loader.py project/tests/test_dashboard.py -q
   → 257 passed, 6 skipped
   pytest tests/test_eda_paths.py -q                      → 4 passed
   python scripts/check_experiment_outputs.py             → EXPERIMENT OUTPUTS OK — 12 enforced steps, 23 advisory gaps (exit 0; no pipeline/eda note)
   ruff check <changed python files>                      → All checks passed
   mypy <gate + changed src files>                        → Success: no issues found in 4 source files
   render_gates.py round trip (test_gate_registry)        → byte-identical, help exit 0
   ```

## Rollback plan

1. `git revert <this commit>` restores config/loader/schema/dashboard/gate/
   ledger/README and removes the new module, tests, sidecar and committed
   pipeline artifacts in one step.
2. The old ignored `artifacts/reports/` copy remains readable.
3. To roll back only the artifact location: remove
   `project/config/step/eda.yaml` and rerun the EDA command; the base default
   writes back to `artifacts/reports/`, and the new results directory is
   removable with `rm -rf project/experiments/results/pipeline` (regenerable).

## Risks / open items

- **`configure_from_env` mount is first-wins:** if `/results` was already
  mounted to a different directory, a later env change does not remount. The
  project launcher (`project/dashboard.py`) sets `BROADWAY_EXPERIMENTS_ROOT`
  before importing the module, so the normal path is correct; a test harness
  that imports early and mutates the root still needs `configure_from_env`
  before its first mount. Documented, not changed.
- **Other steps keep cwd-relative `output_path`:** only `EdaStep` was anchored;
  the other step models are separate units.
- **Malformed-record log is process-global:** `_LOGGED_UNREADABLE` suppresses a
  repeat traceback for the same path for the lifetime of the dashboard process.
  A fixed-then-rebroken file will not re-log until restart; that is the intended
  tradeoff for the no-log-bloat rule.
- **`Decision` still accepts infinity:** `value`/`threshold` are plain floats and
  `1e999` validates. Fixing that belongs in `reports/decision.py`, which the
  parent contract marks as a separate no-touch unit; recorded as a pre-existing
  model gap, not changed here.
- **Gate skip semantics:** a platform-only checkout (or one without an
  experiments root) skips the project experiments and `eda` enforcement and
  exits 0. A project checkout with an experiments root enforces `eda`
  regardless of whether `results/` currently exists; deleting the record or the
  whole results tree exits 1, which is the intended tooth. The `project.paths`
  import happens only inside `main()`, never at module load.
- **No new GATE row:** ownership is folded into GATE-SURF-66 because inserting
  a row means renumbering the dense global order and `meta.counts`; the row now
  names the module, test and committed artifacts.
- **Committed sidecar:** `eda_feature_decisions.json` is tracked by design
  (30 KB). If the reviewer prefers it ignored, the negation and the row link
  can be removed together.
- **DIGEST date:** the rendered digest carries the render date; it was rendered
  on 2026-11-27.
- **Not run:** the full pytest suite; the requested modules plus the touched
  loader/EDA-path tests were run.

## Verification limits

- I did not open a browser; I called the real `/` route (`ui.index`) and the
  in-process dashboard renderer, and checked the anchors point at files that
  exist.
- I did not regenerate other pipeline steps; only EDA publishes a pipeline
  record today.
