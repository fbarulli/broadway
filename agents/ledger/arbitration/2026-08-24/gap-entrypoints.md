# GH-2 GAP-HUNTER — Entry-point census & gate coverage (2026-08-24)

Contract GH-2 @ HEAD 5016e93 · read-only sweep · SSOT cross-checked against
`agents/ledger/gates.yaml` (89 gates) + `DIGEST.md` via `agents/tools/render_gates.py`.
Scope: every user/kernel-reachable entry point — console scripts, `python -m` /
`__main__` blocks, argparse trees, executable shell, containers/k8s, documented ops verbs.

## Census

| # | entry-point class | found | gated-already | NEW-candidate |
|---|---|---|---|---|
| 1 | pyproject `[project.scripts]` | 1 (`ds-pipeline`) | 1 (GATE-TLINE-59 owns `cli.py:105 main()`) | 0 |
| 2 | `ds-pipeline` argparse subcommands (cli.py:26-102) | 19 (+`stats run\|describe`) | 13 dispatched bodies owned (see map below) | 6 (`init`, `causal`, `contracts`, `baseline`, `columns`, bare-CMD dispatch gap) |
| 3 | PREDICTION-mode entries | 3 (`train`, `evaluate` @ training/module.py:92 + evaluate/module.py:85 require `AnalysisMode.PREDICTION`; `baseline` prediction leg) | 2 (TRAIN-30, TRAIN-35 own compute legs) | 1 (serving tail = PROPOSE-GH2-01) |
| 4 | `python -m` / `__main__.py` blocks | ~80 files | 1 (`scripts/tier_classifier.py` → GATE-INFRA-98); outputs convention only for experiments lane (GATE-SURF-68) | 4 (PROPOSE-GH2-08..10,16); `project/scripts/*.py` ×11 = **KNOWN teaching-surface question — referenced, not re-ruled** |
| 5 | executable shell (`scripts/` + shebang scan) | 8 | 5 (GATE-INFRA-90..97) | 3 (`k8s/optuna/{lifecycle,teardown}.sh`, `entrypoint-mlflow.sh` — syntax-only via CI-99 sh -n/shellcheck) |
| 6 | containers / k8s ENTRYPOINT·CMD·command | 9 surfaces | 1 chain (k8s/train-job.yaml → gated `ds-pipeline train`) | 8 (PROPOSE-GH2-01,12,13,14,15 + mlflow/compose/postgres surfaces) |
| 7 | docs/workflow cron-like ops verbs | 5 (`lifecycle.sh train\|up\|dump\|down\|view`, HPO_TRAINING.md:71-124) | 0 | folded into PROPOSE-GH2-12 |
| | **total** | **~45 distinct reachable surfaces** | **22 covered by existing gates** | **16 candidates** |

### Subcommand → gate map (walked EVERY parser in `_build_parser`)

discover→SURF-66(partial) · columns→**none** · lineage→SURF-64 · report→SURF-62+TLINE-59 ·
profile→SURF-66 · audit→SURF-63 · ingest→ETL-19(+INGEST-01) · init→**none** ·
etl→ETL-10 · contracts(step)→**none** (CFG-75/INGEST-09 own pandera build, NOT this runner) ·
features→FEAT-25 · stats run/describe→STATS-40/41/49 · causal→**none** · train→TRAIN-30 ·
evaluate→TRAIN-35 · baseline→**none** · full→CFG-78(dispatch)+composed steps ·
walkthrough→TLINE-54+59 · decide→TLINE-52+59.
GATE-TLINE-59's `inputs` clause enumerates ONLY walkthrough/decide/lineage/report/stats argv
surfaces — the other 14 subparsers have no gate reading their argv contract.

## CANDIDATE GATES (house registry format)

- id: PROPOSE-GH2-01 · owner: `src/broadway/inference/api.py:1` + `k8s/api-deployment.yaml:18 command` · inputs: [MLflow registry champion via models:/<model>@champion (get_champion mlflow_utils.py:152)] · outputs: [HTTP /health /predict /metrics on 0.0.0.0:8000, Service broadway-api :80→8000, HPA 2→10 replicas] · transforms: [uvicorn `inference.api:app` import] · touched_by: [] · validated_by: [] · FINDING: **api.py is a one-line docstring stub — no `app` symbol exists**, so `uvicorn inference.api:app` raises AttributeError ⇒ CrashLoopBackOff on all replicas. This IS the champion predict path the deployment promises; implemented nowhere else (pyfunc_wrapper.predict serves only inside MLflow pyfunc; evaluate loads champions offline under TRAIN-35). Root: serving surface declared in infra before any implementation or gate; blast-radius tool blind (proof below).
- id: PROPOSE-GH2-02 · owner: `src/broadway/onboard/module.py:215 init()` (`ds-pipeline init`) · inputs: [interactive stdin prompts, CSV path, 13 argv flags] · outputs: [OVERWRITES configs/dataset/<name>.yaml + analysis + experiment YAMLs (_write_configs :178-198), artifacts profile JSON :205 with one lineage record :206] · transforms: [infer hints → typed contracts → write_text onto the config SSOT tree consumed by GATE-CFG-70..79] · touched_by: [] · validated_by: [] · FINDING: the ONLY entry point that mutates the contract/config registry itself, yet no gate reads its argv or its writes; a wrong answer silently re-pins every downstream gate's declared surface. Root: unvalidated config-writing scaffolder behind a gated loader.
- id: PROPOSE-GH2-03 · owner: `src/broadway/baseline/module.py:45 run()` (`ds-pipeline baseline`) · inputs: [PipelineConfig(baseline section), dataset frame via data.loader.load] · outputs: [BaselineResult JSON at cfg.baseline.output_dir/output_file (:82-92) + lineage record node_id("baseline") :92] · transforms: [_git_commit() subprocess.run(["git","rev-parse","HEAD"]) :20-27 → ArtifactTrace.commit, silent "unknown" fallback :26-27; mode dispatch hypothesis/causal/prediction :51] · touched_by: [] · validated_by: [] · FINDING: full pipeline step with a subprocess call and an artifact writer, zero gates, zero tests greppable; commit provenance can silently degrade to "unknown". Only `src/broadway/stats/baseline.py` (a different file) is touched by GATE-STATS-48.
- id: PROPOSE-GH2-04 · owner: `src/broadway/causal/module.py:18 run()` (`ds-pipeline causal`) · inputs: [cfg.causal design params, canonical frame] · outputs: [causal design artifact via save_design(out_path) :35 + lineage record :37] · transforms: [assignment/design/multiple/sequential/hte composition] · touched_by: [] · validated_by: [] · FINDING: entire causal step ungated despite configs/analysis/*_causal.yaml shipping; record exists but nothing pins design semantics or the argv entry.
- id: PROPOSE-GH2-05 · owner: `src/broadway/contracts/module.py:19 run()` (`ds-pipeline contracts`) · inputs: [cfg.dataset, cfg.contracts.null_threshold, loaded frame] · outputs: [pass/fail log verdict; raises ValueError on violation :30-33] · transforms: [check_columns + check_nulls (contracts/checks.py — not the gated pandera path)] · touched_by: [] · validated_by: [] · FINDING: duplicate contract-checking surface parallel to gated GATE-CFG-75/GATE-INGEST-09; its checks.py rules and threshold semantics are unpinned, so two "contract" truths coexist.
- id: PROPOSE-GH2-06 · owner: `src/broadway/data/download.py:16 download()` · inputs: [arbitrary URL, EnvironmentConfig.data_dir/raw_subdir/download_chunk_size] · outputs: [<raw_dir>/<URL-basename> file written streaming] · transforms: [requests.get(stream=True) → iter_content → open(dest,"wb"); filename derived from urlparse path :17] · touched_by: [requests env] · validated_by: [tests/test_download.py::test_download_writes_chunks_under_raw_dir (unit-only)] · FINDING: **orphan fetch helper — zero production callers** (repo grep: only tests import it; no step module or CLI wires it), so raw data enters via untracked side channels while this pinned-by-test function sits dead; if ever wired, URL-controlled filename lands in the globbed raw_dir feeding GATE-INGEST-01. Unvalidated as an entry, unreachable as shipped.
- id: PROPOSE-GH2-07 · owner: `src/broadway/discover/columns.py:10 run()` (`ds-pipeline columns --csv`) · inputs: [raw CSV path] · outputs: [stdout column report] · transforms: [read + dtype/null summary] · touched_by: [] · validated_by: [] · FINDING: minor — last ungated read-only subparser; inconsistent with sibling discover/profile which SURF-66 partially owns.
- id: PROPOSE-GH2-08 · owner: `src/broadway/discover/qq.py` `if __name__ == "__main__"` block (tail of module) · inputs: [none — synthetic np.random frame seeded 42] · outputs: [qq_demo_output/ + qq_demo_overview.json written to CWD] · transforms: [plot_numeric_qq demo invocation] · touched_by: [] · validated_by: [] · FINDING: a demo `__main__` block shipped inside library code under src/ — stray artifact writes into whatever CWD it runs from; not covered by SURF-66 (which owns the function's production call sites, not this block).
- id: PROPOSE-GH2-09 · owner: `experiments.py:main` argparse dispatcher (`python experiments.py|python -m experiments <ols|diagnostics|qq_legend|verify>`) · inputs: [argv subcommand] · outputs: [residual Q-Q plots, diagnostics renderings, legend plots, verification JSON; experiments/results/ CSVs] · transforms: [merged four legacy root scripts; dynamic importlib loading of experiment modules] · touched_by: [ruff/mypy battery GATE-INFRA-92 lint-scope only] · validated_by: [] · FINDING: a second full CLI tree at repo root that no gate reads; `verify` even audits the experiments tree itself — self-check without registry backing.
- id: PROPOSE-GH2-10 · owner: `experiments_ui.py:893 __main__` (uvicorn.run(app, host=127.0.0.1, port=8000)) · inputs: [experiments/results/** + numbered NN_*.py series scan] · outputs: [FastAPI HTML dashboard on :8000 incl. per-step pipeline graphs] · transforms: [AST/importlib scan of experiment sources to render graphs] · touched_by: [] · validated_by: [] · FINDING: network-serving entry (localhost) executing source-tree introspection; unpinned like the API it mirrors.
- id: PROPOSE-GH2-11 · owner: `k8s/optuna/lifecycle.sh:train|up|dump|down` + `teardown.sh` · inputs: [kind cluster state, OPTUNA_WORKER_IMAGE, $BACKUP_DIR dumps] · outputs: [applied mlflow.yaml+optuna-init.yaml (:105), optuna.sql.gz/mlflow.sql.gz/artifacts snapshots, deleted cluster] · transforms: [restore→run HPO jobs→snapshot→teardown chain; pg_dump/gzip round-trip] · touched_by: [GATE-INFRA-99 syntax layer only (sh -n/shellcheck)] · validated_by: [] · FINDING: the documented operational entry for ALL HPO training (HPO_TRAINING.md:71-124) has no behavioral gate; a restore-order or dump-whitespace regression passes CI green.
- id: PROPOSE-GH2-12 · owner: `k8s/optuna/optuna-init.yaml:19 command ["python","/app/worker.py"]` · inputs: [optuna-config ConfigMap, optuna-db Secret] · outputs: [pre-created Optuna studies in postgres (schema-race fix per header comment)] · transforms: [--init-only study bootstrap] · touched_by: [] · validated_by: [] · FINDING: **phantom target — no `worker.py` exists anywhere in the repo** (find: only .venv/.uv-cache hits); Dockerfile.base copies only src/ + project/, Dockerfile.worker installs 03_optuna_worker.py at a DIFFERENT path. Image contract unverifiable from source; Job would ModuleNotFound unless built from an untracked layout.
- id: PROPOSE-GH2-13 · owner: `Dockerfile.worker:16 CMD ["python","/app/experiments/mlflow/03_optuna_worker.py"]` (argparse: --model/--config/--secret-dir/--init-only) · inputs: [/etc/broadway/config.yaml ConfigMap, secret dir, model registry choices] · outputs: [MLflow runs + Optuna trials in shared postgres] · transforms: [HPO loop logging ARTIFACT-MLFLOW-RUN candidates] · touched_by: [GATE-INFRA-99 kubeconform/docker-only syntax checks] · validated_by: [] · FINDING: the containerized producer of every candidate model that TRAIN-35/promotion later judges is itself ungated; its MLflow logging path feeds GATE-TRAIN-39 classification downstream but nothing pins the worker's own behavior.
- id: PROPOSE-GH2-14 · owner: `Dockerfile:19 CMD ["ds-pipeline"]` (bare) · inputs: [] · outputs: [argparse exit 2 — required subcommand missing] · transforms: [none] · touched_by: [] · validated_by: [] · FINDING: image default CMD is a guaranteed-crash entry; only correct when overridden (as train-job.yaml does). Container boot path never exercises the gated CLI happy path by default; build-and-boot in CI-99 tests images but the default entrypoint semantics are unpinned.
- id: PROPOSE-GH2-15 · owner: `docker-compose.yml:8 command mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db` + `docker/mlflow/Dockerfile:7 CMD ["mlflow","server"...]` vs `k8s/mlflow-deployment.yaml:20` postgres-backed · inputs: [tracking clients via MLFLOW_TRACKING_URI choice of stack] · outputs: [two divergent champion/trial stores: local sqlite file vs databases.mlflow postgres] · transforms: [server boot] · touched_by: [] · validated_by: [] · FINDING: two first-class mlflow-server entries with different persistence engines; champion manifests listed by GATE-TRAIN-39 differ depending on which entry a user walked in through — split-brain risk for the alias 'champion'.
- id: PROPOSE-GH2-16 · owner: `experiments/**` `__main__` lane (~60 runnable scripts across 7 series dirs) · inputs: [argv-less mains, configs/experiment YAMLs, raw parquet] · outputs: [experiments/results/**/*.csv (tracked via GATE-SURF-68 negation triad), PNGs] · transforms: [ad hoc per-script analysis] · touched_by: [GATE-SURF-68 (outputs convention only); ruff scope INFRA-92 partial] · validated_by: [] · FINDING: largest unaudited entry surface by count. Adjacent to but DISTINCT from the KNOWN project/scripts teaching-surface question (project/scripts/01-12 `__main__` teaching scripts — already ruled; referenced here once, not re-ruled): the experiments lane is the results-tracking production lane whose entry points still have no owner gate.

## Blast-radius cross-check (house law: quote command + output)

Every NEW-candidate owner file was queried:

```
$ python agents/tools/render_gates.py --file src/broadway/inference/api.py
no matches: no gate owns or references 'src/broadway/inference/api.py'
$ python agents/tools/render_gates.py --file src/broadway/baseline/module.py
no matches: no gate owns or references 'src/broadway/baseline/module.py'
$ python agents/tools/render_gates.py --file src/broadway/onboard/module.py
no matches: no gate owns or references 'src/broadway/onboard/module.py'
$ python agents/tools/render_gates.py --file src/broadway/causal/module.py
no matches: no gate owns or references 'src/broadway/causal/module.py'
$ python agents/tools/render_gates.py --file src/broadway/contracts/module.py
no matches: no gate owns or references 'src/broadway/contracts/module.py'
$ python agents/tools/render_gates.py --file src/broadway/data/download.py
no matches: no gate owns or references 'src/broadway/data/download.py'
$ python agents/tools/render_gates.py --file experiments.py
no matches: no gate owns or references 'experiments.py'
$ python agents/tools/render_gates.py --file experiments_ui.py
no matches: no gate owns or references 'experiments_ui.py'
$ python agents/tools/render_gates.py --file k8s/api-deployment.yaml
no matches: no gate owns or references 'k8s/api-deployment.yaml'
$ python agents/tools/render_gates.py --file k8s/optuna/lifecycle.sh
no matches: no gate owns or references 'k8s/optuna/lifecycle.sh'
```

Control (tool works, gates exist where claimed):

```
$ python agents/tools/render_gates.py --file src/broadway/cli.py
GATE-TLINE-59 · src/broadway/cli.py:105 main() · owns
== 1 gate(s) touch src/broadway/cli.py ==
$ python agents/tools/render_gates.py --file scripts/run_local_ci.sh
GATE-INFRA-90 · … · owns   GATE-INFRA-91 · … · owns   GATE-INFRA-92 · … · owns
== 3 gate(s) touch scripts/run_local_ci.sh ==
```

10/10 candidate files invisible to the registry query tool ⇒ real holes, not naming drift.

## Top surprises

1. **Champion predict API is vaporware behind a 2→10-replica HPA** (GH2-01): k8s deploys `uvicorn inference.api:app`; `src/broadway/inference/api.py` contains exactly its docstring. Evidence tail: `wc -l` → 1 line; render_gates → `no matches`.
2. **Two phantom/broken container commands**: optuna-init runs `/app/worker.py` which exists nowhere in-repo (GH2-12); base image CMD `["ds-pipeline"]` exits 2 unconditionally (GH2-14).
3. **`ds-pipeline init` is the only entry that rewrites the gate registry's own inputs** (configs/*.yaml SSOT) and is gate-blind (GH2-02).
4. **Subprocess inside kernel code**: `baseline/module.py:22 git rev-parse HEAD`, in an otherwise fully ungated step, with silent `"unknown"` degradation (GH2-03).
5. **Orphan network fetcher**: `data/download.download()` unit-tested yet called by nothing — raw ingestion happens off-book instead (GH2-06).
