# GAP-HUNTER GH-1 — TOPOLOGY SWEEP (orphan territories beyond the nine-lane map)

- contract: GH-1 · HEAD `5016e93` (verified: `git rev-parse HEAD`) · date 2026-08-24
- method: owner-path extraction from `agents/ledger/gates.yaml` (89 gates) diffed against full `git ls-files` tree (525 paths), then per-cluster consumer/import verification
- write scope honored: this file is the ONLY file created; zero mutating git ops

## METHOD (evidence commands)

```
git ls-files | wc -l                        # 525
grep -n "owner:" agents/ledger/gates.yaml   # 89 owner lines extracted
python3 path-coverage diff                  # direct-path unclaimed: 469/525
python3 textual-scan (path OR basename absent from gates.yaml): 248 never mentioned
grep -rln "<symbol>" src/ tests/ configs/ … # per-cluster consumer checks
```

Coverage arithmetic: 525 tracked − 56 owner-path files − indirect claims (report artifacts owned as gate outputs, tests referenced in validated_by, parity SHARED custody list) ⇒ **10 confirmed behavior-bearing orphan clusters** below + 1 pure-prose cluster (NOT-GATES).

## ORPHAN CLUSTERS (confirmed, with disposition)

### C1 — Step-dispatch core (LIVE, unowned heart)
`src/broadway/pipeline.py` (whole module) + `src/broadway/config/loader.py:37-60` `STEP_MODELS`/`STEP_MODULES`.
Evidence: `grep -rn "STEP_MODULES" src/ tests/` → only `pipeline.py:20 importlib.import_module(STEP_MODULES[step])` and `cli.py:16 STEPS = list(STEP_MODELS.keys())`; `grep -c STEP_MODULES agents/ledger/gates.yaml` → **0**. Every `ds-pipeline <step>` invocation routes through these dicts; the nine lanes own `_load_yaml`, `_merge_section`, `resolve_full_steps` (same file!) but not the registry itself.

### C2 — Dead production packages (import-graph orphans)
`src/broadway/trust/` (7 modules), `src/broadway/unsupervised/` (anomaly/clustering/pca + module), `src/broadway/selection/` (information/learning_curves/nested_cv + module), `src/broadway/inference/api.py`, `src/broadway/data/db.py`, `src/broadway/data/download.py`.
Evidence: `grep -rn "from broadway.(trust|unsupervised|selection|onboard)|broadway.data.(db|download)"` outside own packages → **zero hits**; absent from `STEP_MODULES`. Alive only through their own test suites (`tests/test_init.py`, `tests/test_loud_failures.py`, `tests/test_splitter_extended.py`, …). Contrast: `broadway.viz` + `config.viz` looked orphan but are consumed by owned qq/describe/diagnostics/runners — NOT orphans. `data/splitter.py` likewise consumed by etl/module.py:19 + training/module.py:17.

### C3 — Test-harness guards never registered
`tests/conftest.py` (`_SNAPSHOT_DIRS = ["artifacts","reports"]` snapshot hygiene — runs around EVERY pytest invocation; `grep -c conftest agents/ledger/gates.yaml` → 0), `tests/contract_fixture.py` (demo-contract-derived fixture factory), `tests/test_uv_probe_guard.py` (pins pyproject setuptools exclude vs deepseek-harness symlink-loop stall), plus 17 further never-mentioned suites (test_causal*.py ×3, test_onboard*.py ×2, test_explain, test_feature_selection, test_columns, test_profile, test_baseline_module, test_lineage_module, test_optuna_extended, test_experiments_ui, test_splitter_extended, test_utils_extended…).

### C4 — Container/deployment surface (k8s quartet + root image)
Root `Dockerfile` (produces `broadway:latest`) consumed by `k8s/train-job.yaml:10` and `k8s/api-deployment.yaml:17`; `k8s/mlflow-deployment.yaml:17` / `k8s/postgres-deployment.yaml:18` expect `broadway-mlflow:latest` / `broadway-postgres:latest`. Evidence of omission: kubeconform scope hardcoded `k8s/optuna/` (ci.yml:71-80, flagged in gates.yaml:2886 finding) and `grep -n "docker build" .github/workflows/ci.yml` → only `-f k8s/optuna/Dockerfile.base` + thin COPY layers (:163,:177). The quartet is never built NOR schema-scanned in CI.

### C5 — Local compose stack
`docker-compose.yml` (services build `./docker/mlflow`, `./docker/postgres`; ports 5000/5432; named volumes) + `docker/mlflow/Dockerfile` + `docker/postgres/init.sql` (never mentioned anywhere in gates.yaml). Compose's default image-naming convention is what MINTS `broadway-mlflow:latest`/`broadway-postgres:latest` consumed by C4 manifests — a cross-cluster contract nobody owns.

### C6 — CI orchestrator fixture
`.github/ci-fixtures/k8s-config.yaml` mounted into the worker dry-run: `.github/workflows/ci.yml:201 -v "$PWD/.github/ci-fixtures/k8s-config.yaml:/etc/broadway/config.yaml"`. Consumed by the owned ci.yml gate yet zero mentions in gates.yaml (only other tracked .github path besides ci.yml).

### C7 — Local experiment platform (root-level entry points)
`experiments.py` (subcommand dispatcher ols/diagnostics/qq_legend, writes plots/CSVs into repo tree), `experiments_ui.py` (FastAPI dashboard whose folder-discovery convention serves any `experiments/**/NN_*.py` series from `experiments/results/<series>`), `_common.py`/`_setup.py` helpers across 5 series dirs (~90 numbered scripts), 45 result CSV/JSON artifacts. Zero registry mentions except the `.gitignore` negation-triad gate owning only the tracking CONVENTION (gates.yaml:2224).

### C8 — Sample-generation machinery
`src/broadway/samples/generate.py:95 generate_sample()` (writes parquet + provenance JSON sidecar with `definition_sha256`/`artifact_sha256`), `samples/models.py`, inputs `configs/sample/{fare_prediction_1m,taxi_diagnostic,taxi_estimation}.yaml`. Only caller: `tests/test_samples.py`; no CLI subcommand exposes it. All four files zero-mention in gates.yaml.

### C9 — Legacy `project/` twins beyond process.py
`project/etl/process_config.py` (config reader for the OWNED legacy process.py — reads `configs/project/taxi.yaml`), `project/tests/test_process_config.py`, `project/{basic,working,ml_pipeline,features,boroughs,config}.py`, `project/scripts/01..12_*.py` (11 legacy analysis scripts), input `configs/project/taxi.yaml`. Registry mentions only `project/etl/process.py`, `project/data.py`, `project/tests/test_process.py`.

### C10 — Unowned config families (no owner, OUTSIDE parity SHARED list too)
`configs/sample/` (see C8), `configs/experiments/{mlflow,multivariate,working}.yaml`, `configs/slice/*.yaml` (4), `configs/analysis/taxi_{causal,hypothesis}.yaml`, `configs/experiment/taxi.yaml`, `configs/step/{baseline,contracts,discover,features,full,stats,train,viz,walkthrough}.yaml`. Note: parity SHARED (check_branch_parity.sh:44-67) covers only dataset/test, experiment/{baseline,engineered,hyperopt}, analysis/{test,test_hypothesis,test_causal}, step/{causal,etl}, environment/, flow/. `configs/step/viz.yaml` declares a step type ABSENT from `STEP_MODELS` — dead config pointing at a non-dispatchable step.

## CANDIDATE GATES (house format, provisional numbering)

```yaml
- id: PROPOSE-GH1-01
  phase: config
  order: 1
  owner: 'src/broadway/config/loader.py:50 STEP_MODULES (companion STEP_MODELS :37) + src/broadway/pipeline.py:20 run_step() importlib dispatch'
  inputs:
  - CLI argv (ds-pipeline <step>)
  - configs/step/<name>.yaml
  outputs:
  - imported module.run() execution for every step lane
  transforms:
  - cli.py:16 derives STEPS menu from STEP_MODELS.keys()
  - pipeline.py:20 resolves step-name → dotted module string → importlib.import_module → run()
  - KeyError on unregistered-but-configured step (e.g. configs/step/viz.yaml has no registry entry)
  touched_by: []
  validated_by:
  - tests/test_cli_dispatch.py (in-process parser→delegate wiring, monkeypatched entry points)
  - tests/test_full_dispatch.py
  if_changed:
  - every step lane's reachability
  - PROPOSE-GH1-11
  # FINDING: the single funnel all nine lanes flow through is itself unowned; no test imports STEP_MODULES
  #   directly, so registry↔STEP_MODELS↔config-schema parity is pinned only incidentally.
  #   root: nine-lane sweep owned symbols inside loader.py/pipeline call-sites, never the dispatch table between them.
- id: PROPOSE-GH1-02
  phase: infra
  order: 2
  owner: 'tests/conftest.py:6 _SNAPSHOT_DIRS snapshot hygiene'
  inputs:
  - pytest session start/end
  outputs:
  - cleaned artifacts/ + reports/ trees around tests (prevents stale-report false greens)
  transforms:
  - hashes/prunes _SNAPSHOT_DIRS before/after test runs (hashlib+pathlib walk)
  touched_by: []
  validated_by: []  # grep -c conftest gates.yaml == 0
  if_changed:
  - any test that asserts report/artifact freshness
  # FINDING: global side-effect guard executed on every pytest run is absent from the registry.
  #   root: lanes indexed test FILES as validated_by pointers, never the harness file all tests traverse.
- id: PROPOSE-GH1-03
  phase: infra
  order: 3
  owner: 'tests/test_uv_probe_guard.py:1 uv editable-rebuild stall guard'
  inputs:
  - pyproject.toml [tool.setuptools] packages.find exclude list
  outputs:
  - hard fail if deepseek-harness exclusion removed/weakened
  transforms:
  - asserts foreign tree stays out of setuptools discovery (symlink-loop stall would return on next pyproject touch)
  touched_by: []
  validated_by:
  - itself (the guard IS the validator); docstring records 2026-08-23 incident
  if_changed:
  - pyproject.toml packaging section
  # FINDING: incident-derived guard with zero registry presence; pyproject.toml otherwise claimed only by tier-classifier prefix + parity custody.
  #   root: infra lane enumerated scripts/ and ci.yml but not repo-root guard tests born from incidents.
- id: PROPOSE-GH1-04
  phase: surfaces
  order: 4
  owner: 'experiments_ui.py:1 FastAPI dashboard (series-discovery convention)'
  inputs:
  - experiments/<series>/NN_*.py presence
  - experiments/results/<series>/*.csv
  outputs:
  - HTTP dashboard (?focus=<series-id> routing) over local experiment results
  transforms:
  - folder scan → series registry; renders per-step pages from result CSVs
  touched_by: []
  validated_by:
  - tests/test_experiments_ui.py (exists; never cited by any lane)
  if_changed:
  - experiments/results layout consumed by pages
  # FINDING: sole CONSUMER of the 45 tracked result CSVs is an ungated app; result-format drift breaks it silently.
  #   root: results CSVs treated as terminal artifacts; their reader was outside every lane's boundary.
- id: PROPOSE-GH1-05
  phase: experiments
  order: 5
  owner: 'experiments.py:1 subcommand dispatcher (ols | diagnostics | qq_legend)'
  inputs:
  - raw/derived parquet data
  outputs:
  - plots + CSVs written into repo tree (residual Q-Qs, diagnostics renderings)
  transforms:
  - argparse dispatch merging four former root scripts (experiment_ols.py et al.)
  touched_by: []
  validated_by: []  # zero mentions in gates.yaml; manual-entry research tooling
  if_changed:
  - experiments/results content
  # FINDING: repo-root executable writing tracked artifacts, invisible to the map.
  #   root: lanes stopped at src/broadway + scripts/; bare root-level entry points were never enumerated.
- id: PROPOSE-GH1-06
  phase: ingest
  order: 6
  owner: 'src/broadway/samples/generate.py:95 generate_sample()'
  inputs:
  - configs/sample/{fare_prediction_1m,taxi_diagnostic,taxi_estimation}.yaml specs
  - source parquet
  outputs:
  - deterministic sample parquet + provenance JSON sidecar (definition_sha256, artifact_sha256, created_at)
  transforms:
  - canonical-spec sha256 (:32), filters/derived/exclude_any application (:50-93), provenance write (tail)
  touched_by: []
  validated_by:
  - tests/test_samples.py (sole caller; no CLI subcommand exposes generation)
  if_changed:
  - pinned-sample fixtures consumed by CI determinism legs
  # FINDING: the machinery producing the repo's pinned samples is registry-blind; spec-hash drift would be undetected.
  #   root: sample CONSUMPTION (loader._build_schema) got a gate; sample GENERATION upstream did not.
- id: PROPOSE-GH1-07
  phase: infra
  order: 7
  owner: 'k8s/train-job.yaml:10 + k8s/api-deployment.yaml:17 (image: broadway:latest) + mlflow/postgres-deployment.yaml:17-18'
  inputs:
  - broadway:latest (root Dockerfile), broadway-mlflow:latest / broadway-postgres:latest (compose-built)
  outputs:
  - cluster workloads (training job, API serving, tracking server, DB)
  transforms:
  - manifest → workload binding; depends on images CI NEVER BUILDS (ci.yml builds only k8s/optuna/*)
  touched_by: []
  validated_by: []  # kubeconform hardcoded to k8s/optuna/ (ci.yml:80; gates.yaml:2886 already flags the scope gap)
  if_changed:
  - PROPOSE-GH1-08, PROPOSE-GH1-09
  # FINDING: a four-manifest deployment economy rides on an implicit image-tag contract with the compose stack.
  #   root: infra lane adopted ci.yml's own scope (k8s/optuna/) as the territory boundary instead of git ls-files k8s/.
- id: PROPOSE-GH1-08
  phase: infra
  order: 8
  owner: 'docker-compose.yml:3 services mlflow/postgres (build ./docker/mlflow, ./docker/postgres)'
  inputs:
  - docker/mlflow/Dockerfile, docker/postgres/init.sql
  - env DATABASE_*/MLFLOW_PORT vars
  outputs:
  - local tracking server :5000 + postgres :5432; named volumes mlflow_data/pg_data
  - mints broadway-mlflow:latest / broadway-postgres:latest via compose default tagging
  transforms:
  - two-service build+run; init.sql bootstraps DB schema on first volume init
  touched_by: []
  validated_by: []  # init.sql: zero mentions repo-wide outside compose reference
  if_changed:
  - PROPOSE-GH1-07 image expectations
  # FINDING: compose's implicit image names are a load-bearing cross-file contract nothing validates.
  #   root: docker/ beyond optuna fell outside both ci.yml's docker-only checks and the lane sweep.
- id: PROPOSE-GH1-09
  phase: infra
  order: 9
  owner: 'Dockerfile:1 multi-stage uv builder → broadway:latest'
  inputs:
  - python:3.12-slim, uv.lock, pyproject.toml, src/
  outputs:
  - broadway:latest consumed by k8s train-job + api-deployment
  transforms:
  - uv sync into image layers (FROM/COPY pyproject+lock first for cache)
  touched_by: []
  validated_by: []  # not referenced by ci.yml (only k8s/optuna/Dockerfile.base is), not by compose
  if_changed:
  - PROPOSE-GH1-07
  # FINDING: the production-app image is built by NOTHING automated; drift from k8s/optuna/Dockerfile.base deps unchecked.
  #   root: 'Dockerfile' token matched the optuna base image in lane text; the bare root file escaped both greps and custody semantics.
- id: PROPOSE-GH1-10
  phase: etl
  order: 10
  owner: 'project/etl/process_config.py:1 legacy twin config reader'
  inputs:
  - configs/project/taxi.yaml (taxi knobs), configs/step/etl.yaml (generic etl knobs)
  outputs:
  - typed config object feeding legacy project/etl/process.py path
  transforms:
  - yaml load via broadway.config.loader.CONFIGS_DIR + merge of two sections
  touched_by: []
  validated_by:
  - project/tests/test_process_config.py (exists; zero lane citations)
  if_changed:
  - legacy ETL behavior parity vs CONTRACT pipeline (GATE-INGEST-03 documents the twin grammar)
  # FINDING: the legacy twin's CONFIG leg was unmapped even though the twin's runtime legs are gate-documented.
  #   root: lanes mapped process.py symbols directly and skipped its dedicated config module + private test.
- id: PROPOSE-GH1-11
  phase: infra
  order: 11
  owner: '.github/ci-fixtures/k8s-config.yaml:1 orchestrator dry-run config fixture'
  inputs:
  - rendered worker jobs (RUNNER_TEMP/jobs.yaml)
  outputs:
  - CI verdict: rendered jobs must contain BROADWAY_MLFLOW_CONFIG (ci.yml:96 grep against this fixture's schema)
  transforms:
  - bind-mounted as /etc/broadway/config.yaml inside broadway-optuna-worker during dry-run (ci.yml:201)
  touched_by: []
  validated_by:
  - .github/workflows/ci.yml orchestrator job (indirectly; fixture itself uncited in gates.yaml)
  if_changed:
  - GATE-INFRA ci.yml docker-only verdicts (:2858 lane entry)
  # FINDING: CI correctness depends on a fixture file the registry does not know exists.
  #   root: infra lane recorded ci.yml's steps but not the tracked non-workflow file they consume.
```

## NOT-GATES (pure prose / inert — do NOT re-hunt)

- Repo-root docs: README.md, TODO.md, HPO_TRAINING.md, dataflow.md, synth.md, project.md, SKLEARN_PIPELINES.md (WIP-deleted), readmore/00_ingestion.md, project/STATS.md
- agents/ prose: contracts/ (5), audits/ (3), ledger/ DECISIONS/FIXES/HANDOFF/STATE, notes/ (10) — behavior-free by charter
- `src/broadway/stats/API.md` — prose declaring itself signature SSOT ("implementations must match exactly") but `grep -rln API.md tests/ src/ scripts/` → ZERO enforcement anywhere; document-only until a checker exists
- `.env.example` — zero consumers found (not in compose/scripts/ci)
- `.python-version` — toolchain pin, inert
- `reports/**` md/png/json + lineage graph.* — generated OUTPUTS of already-owned report gates
- `uv.lock` — lockfile; consumed via ci.yml cache hashFiles(:148) and tier prefixes; tooling-managed, no behavioral gate warranted

## TWO-OWNER CONFLICTS (existing gate + new claimant)

1. **C4/C5/GH1-07..09 vs GATE-INFRA branch-parity custody** — SHARED list (scripts/check_branch_parity.sh:44-67; gates.yaml:2690) already claims byte-identity of `k8s/`, `docker/`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/`, `.github/workflows/`. Custody ≠ semantics: parity would happily keep drifted-broken manifests byte-identical. Any adoption of GH1-07..09 must be marked co-tenant with the parity gate, splitting responsibility (bytes ↔ meaning).
2. **GH1-04/GH1-05/C7 vs tier_classifier prefixes** — tier gate (gates.yaml:2832) classifies ALL changes under `experiments/` (and `src/ tests/ project/`), so my experiment-platform claimants overlap its routing jurisdiction; and the `.gitignore` negation-triad gate (gates.yaml:2224) already owns `experiments/results` TRACKING convention — GH1-04 adds the READING side of the same files.
3. **GH1-01 file co-tenancy in config/loader.py** — same file already hosts three owned gates (_load_yaml/_merge_section/resolve_full_steps). Symbol-disjoint (registry vs loader functions) but same-file blast radius; edits for one invalidate the others' line anchors.
4. **demo/ double-existing-claim (flagged, no new gate)** — parity SHARED `demo/` AND dataset-contract input (`configs/dataset/test.yaml:21 path: demo/demo.csv` feeds owned CFG-DATASET-CONTRACT loader gates). Not an orphan; listed so nobody re-hunts it as one.
5. **GH1-03 vs pyproject.toml soft claims** — tier prefix + parity custody + ci.yml cache key all touch pyproject.toml; the uv-guard is the first BEHAVIORAL claimant of its packaging section.

## SUSPECT VERDICTS (brief's list, confirm/deny)

| suspect | verdict | evidence |
|---|---|---|
| demo/ | DENIED (claimed) | configs/dataset/test.yaml:21 consumes it; parity SHARED covers bytes |
| docker/ beyond optuna | CONFIRMED | docker/mlflow/Dockerfile + docker/postgres/init.sql; init.sql zero-mention; only consumer docker-compose.yml |
| .github/ beyond ci.yml | CONFIRMED | exactly one extra tracked path: .github/ci-fixtures/k8s-config.yaml (consumed ci.yml:201, zero registry mention); no templates/other workflows |
| docs/ incl stats/API.md | HALF | no tracked docs/ dir; stats/API.md = prose SSOT with zero enforcement → NOT-GATE |
| scripts/ leftovers | DENIED | all six known scripts gated (check_champion_manifest.sh covered inside champion gate transforms :1289-:1317); setup_hooks does NOT exist; no unaccounted scripts |
| configs/sample machinery | CONFIRMED | generate.py+models.py+3 yamls zero-mention; generator callable only from its test |
| data/ convention files | DENIED (vacuous) | `git ls-files data/` → empty at HEAD |
| src/broadway import orphans | CONFIRMED | trust(7)/unsupervised(4)/selection(4)/inference.api/data.db/data.download: zero external importers, absent from STEP_MODULES |

## COUNTS

- Confirmed orphan clusters: **10** (C1–C10) + 1 prose cluster diverted to NOT-GATES
- Candidate gates emitted: **11** (PROPOSE-GH1-01..11)
- Two-owner conflicts flagged: **5**
- Never-mentioned tracked files (raw textual scan): 248/525 (includes prose; classified above)
