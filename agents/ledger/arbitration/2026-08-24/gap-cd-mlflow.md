# GH-5 — CD reconciliation + MLflow downstream object control

Arbitration date: 2026-08-24 · Repo: /home/opc/ONE/broad-way · HEAD: sklearn @ `5016e93`
Scope honored: read-only except this file; git ops limited to read-only inspection of all refs (`refs/heads`, `refs/remotes`, tag `tier-1-complete`) + GitHub Actions registry API.

---

## PART A — CD reconciliation (GH-1 "image economy never built or scanned" vs the human's cd.yaml claim)

### A1. Workflow enumeration — every reachable ref

| Ref | `.github` workflow files | ci.yml blob |
|---|---|---|
| sklearn @ 5016e93 (HEAD) | `workflows/ci.yml` (+ `ci-fixtures/k8s-config.yaml`) | `c4ee9377` |
| origin/taxi @ 7c5ca95 | same set | `c4ee9377` (identical to HEAD) |
| main == origin/main @ 1860709 | same set | `5efa6469` (older variant) |
| taxi_work @ 5c6370f | same set | `5efa6469` |
| taxi (local) @ 7cbe172 | same set | `57edbee6` (older than its own origin!) |
| sp_probe @ ee014cf | same set | `57edbee6` |
| pr-1 / pr-2 | `workflows/ci.yml` only | `6a0d3e0e` (**no CD job**) |
| tag tier-1-complete | + `.github/parity-era.env` | `8faf3513` |

GitHub Actions registry (`gh api repos/fbarulli/broadway/actions/workflows`): exactly two entries —
`.github/workflows/ci.yml` (active) and `dynamic/dependabot/update-graph` (GitHub-generated). **No `cd.yaml`
exists on any ref, local branch, tag, or in GitHub's registry.** The human's specific suggestion ("cd.yaml now
handles it") is factually wrong about the file; the CD capability is a *job inside* `ci.yml`.

### A2. Build/push/scan steps per workflow@ref

Only one workflow file exists anywhere: `.github/workflows/ci.yml`. Its newest variant (`c4ee9377`,
sklearn@HEAD == origin/taxi) contains:

Builds — job `build-and-boot` (ci.yml@HEAD):
- L158–166 `Build base image`: `docker build -f k8s/optuna/Dockerfile.base … broadway-base:latest .` (comment: ":latest so worker/mlflow `FROM broadway-base:latest` resolves locally"; "broadway-base stays CI-private", L244–245)
- L175–179 `Build worker image`: `-f k8s/optuna/Dockerfile.worker` → `broadway-optuna-worker:${{ github.sha }}` ("immutable sha tag", L174)
- L182–186 `Build mlflow image`: `-f k8s/optuna/Dockerfile.mlflow` → `mlflow-server:${{ github.sha }}`
- Boot tests consume those exact sha tags (L193, L201, L211–212).

Package/publish — job `CD (push images to GHCR)` (same file, not a separate cd.yaml):
- L217–223 `Package verified deployment images`, gated `if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/taxi')` — `docker save broadway-optuna-worker:$sha mlflow-server:$sha` ("so cd publishes bit-for-bit what was verified — zero rebuilds downstream. sklearn/PR runs deploy nothing.", L214–216)
- L247–254 job gate identical + `needs: [platform, build-and-boot]` + concurrency serialization ("overlapping runs must never interleave :latest/:taxi layer pushes into a torn mutable tag")
- L275 `docker/login-action@c94ce9fb… # v3`; L282–301 `Push verified images to GHCR` → `ghcr.io/${{ github.repository }}/…` with `$sha` always, `:latest` only from main, `:taxi` only from taxi (tag law, L243–245).

Older variants still live on other refs (drift evidence):
- `main`/`origin/main` @ `5efa6469` (2026-08-19): builds `:test` tags and the CD job **rebuilds from scratch** then pushes sha+`:latest` only — no artifact packaging, no `:taxi` tag law (L205–217 there).
- local `taxi` @ `57edbee6` predates origin/taxi's current ci.yml (local behind its own upstream).
- pr-1/pr-2 @ `6a0d3e0e`: no CD job at all.

Scans/validation:
- **Vulnerability scanning: ZERO.** `git grep -ilE 'trivy|grype|snyk|cosign|syft|aquasec|dockle|vulnerab'` across every ref under `.github` → zero hits on all of them. Nothing is ever scanned, pushed or not.
- kubeconform v0.6.7 scope = `k8s/optuna/` (minus kind-config.yaml) + rendered worker jobs only (L66–104). The four top-level `k8s/*.yaml` manifests are validated by no ref's CI.

### A3. Image-reference inventory vs builders

| Image | Demanded-by | Built-by (workflow@ref) | Scanned-by |
|---|---|---|---|
| `broadway-optuna-worker` | k8s/optuna/optuna-init.yaml:17; render_worker_jobs.py default + lifecycle.sh:137 | ci.yml@c4ee9377 build-and-boot ($sha); pushed to GHCR by embedded CD job (sha; :latest from main; :taxi from taxi) | NONE |
| `mlflow-server` | k8s/optuna/mlflow.yaml:25 | ci.yml@c4ee9377 build-and-boot ($sha); pushed by same CD job | NONE |
| `broadway-base` | internal FROM parent (Dockerfile.worker/.mlflow) | ci.yml@c4ee9377 (CI-private, never pushed) | NONE |
| `broadway` | k8s/train-job.yaml:10; k8s/api-deployment.yaml:17 | **UNBUILT** (root ./Dockerfile referenced by no workflow) | NONE |
| `broadway-mlflow` | k8s/mlflow-deployment.yaml:17 | **UNBUILT** (name mismatch: CI builds `mlflow-server`) | NONE |
| `broadway-postgres` | k8s/postgres-deployment.yaml:18 | **UNBUILT** | NONE |
| `postgres:16-alpine` | k8s/optuna/postgres.yaml:39 (+ lifecycle.sh view stack L189) | pulled upstream from Docker Hub, never built/scanned locally | NONE |
| compose `mlflow` | docker-compose.yml service mlflow (`build: ./docker/mlflow`) | compose-local build only; never CI | NONE |
| compose `postgres` | docker-compose.yml service postgres (`build: ./docker/postgres`) | compose-local build only; never CI | NONE |

### A4. Part-A verdict

**REFUTED as to building / HOLDS as to scanning ⇒ net PARTIAL.** GH-1's claim that an entire image economy is
"never built" no longer holds at current refs — but not via any `cd.yaml`: the covering publisher is a CD *job
inside* `.github/workflows/ci.yml` (carried identically by sklearn@5016e93 and origin/taxi, blob `c4ee9377`),
which builds 3 images and pushes 2 to GHCR on main/taxi only. What survives of GH-1: (a) **zero vulnerability
scanning exists on every ref**; (b) three of seven demanded images (`broadway`, `broadway-mlflow`,
`broadway-postgres`) remain UNBUILT by any workflow, while the top-level k8s manifests demanding them are
outside kubeconform scope; (c) the covering variant is **not synced to main** (`5efa6469` rebuild-style CD,
no :taxi law) and pr-1/pr-2 carry no CD job — a parity/drift question, with sklearn itself fully covered.

---

## PART B — MLflow downstream object control census

### B1. Object-creation inventory (10 sites, 7 first-party files, 0 deletion calls)

| # | Site | Objects created | Naming/schema law | Declared gate constraining it |
|---|---|---|---|---|
| 1 | src/broadway/training/mlflow_utils.py:54 `setup_mlflow` | experiment (auto-created) | caller-supplied free string | NONE |
| 2 | mlflow_utils.py:79–89 log_params/log_metrics/log_metadata | params/metrics dicts | none (pass-through) | NONE |
| 3 | mlflow_utils.py:92 `log_dataset` | param `dataset_id` + dataset lineage input (from_pandas, context="train") | param key fixed; source path unchecked if missing (warn+continue) | NONE |
| 4 | mlflow_utils.py:135 `log_model` | logged model artifact (sklearn flavor, cloudpickle, name="model") | artifact path convention "model"; signature optional | bucket classification AFTER the fact (classify_champion), not a write gate |
| 5 | mlflow_utils.py:162 `promote_candidate` | **registered model VERSION** + **`champion` ALIAS** (`MlflowClient().set_registered_model_alias`) | registered_model_name = caller arg (= cfg.dataset.name at both call sites); alias literal "champion" | threshold decision `should_promote` (evaluate/promotion.py) + persistence-before-promotion ordering enforced by tests/test_evaluate_contracts.py:339–411 — but NO custody on who may call it |
| 6 | src/broadway/training/hpo.py:78–96 | nested run per trial `{study} trial {n}`; tags {trial, study}; params=trial.params; metrics incl `target_metric` | convention only | NONE |
| 7 | src/broadway/training/module.py:107–145 | unnamed run; params + `data_source.*`; metrics + `baseline_improvement`; logged model "model" | convention only | NONE |
| 8 | experiments/mlflow/_common.py:203–240 `log_best_artifacts` | logged model "model"; artifacts predictions.csv, feature_importance.png | fixed filenames, no schema | NONE |
| 9 | experiments/mlflow/01_model_battle_mlflow.py:56,86,230 + 02_explainability.py:38,166 | runs `run_name=summary["name"]` / `hpo_bandit` / `explainability` in EXPERIMENT=`ratecode1_model_battle`; tags `model`; explainability PNG artifact | constants hard-coded | NONE |
| 10 | experiments/mlflow/03_optuna_worker.py:106–113 (entry: k8s Job via render_worker_jobs.py, or manual) | run `optuna_{model_name}`; metric mae; tags model/study/seed; log_dataset; metadata n_trials; best artifacts | convention only; experiment from configmap.yaml (`ratecode1_model_battle`) | preflight connectivity check only |

Supporting actors (read-only or infra): scripts/check_champion_manifest.sh (manifest reporter — "the manifest
is not a gate"), list_champions/classify_champion (reads), check_e2e_determinism.sh (ephemeral server +
scratch sqlite writes for determinism diffing), k8s/optuna/lifecycle.sh (no MLflow API writes; dumps/restores
mlflow DB + artifacts into unbounded `$BACKUP_DIR` tarballs).

Deletion/retention/stage transitions: **zero call sites repo-wide** (no delete_run, delete_registered_model,
delete_model_version, transition_model_version_stage, no version/tag pruning). Registry state is append-only
by accident, not by law. Stage usage: none (aliases only).

### B2. CANDIDATES — PROPOSE-GH5-NN (real control gaps only)

- **PROPOSE-GH5-01 champion-alias single-writer custody.** The `champion` alias decides what production
  loads; today any process holding the tracking URI can repoint it — including the optuna worker pods, which
  run in-cluster with full MLflow reachability (configmap.yaml `allowed_hosts: "*"`). Gate: alias writes must
  originate solely from the promotion chokepoint (import-lint forbidding `set_registered_model_alias` /
  direct MlflowClient writes outside training.mlflow_utils, plus a registry-side audit trail).
- **PROPOSE-GH5-02 registered-model-name pin.** Registered model names must equal the configured dataset id
  (cfg.dataset.name / configmap `dataset.name`). Currently implicit; nothing rejects `register_model` under an
  ad-hoc name, which would silently fork the champion namespace that get_champion/list_champions scan.
- **PROPOSE-GH5-03 run-tag schema gate.** Required closed tag vocabulary {model, study, seed} on HPO/worker
  runs; unknown keys rejected. Today tags are free strings written by five independent sites (sites 6, 9, 10).
- **PROPOSE-GH5-04 experiment-name pin.** Exactly two lawful namespaces: `ratecode1_model_battle` (experiment
  family incl. workers) and dataset-name experiments derived from configs/dataset/*.yaml. `setup_mlflow` should
  refuse anything else instead of auto-creating orphan experiments.
- **PROPOSE-GH5-05 artifact-retention/lifecycle gate.** No deletion/retention law exists anywhere: registry
  versions accumulate forever and lifecycle.sh snapshot tarballs grow unbounded. Define retention (N snapshot
  generations; version pruning policy for non-champion versions) as an explicit, invoked operation rather than
  accidental append-only growth.
- **PROPOSE-GH5-06 run-name convention gate.** Enforce the de-facto patterns (`optuna_{model}`,
  `{study} trial {n}`, battle names) so check_e2e_determinism and the champion manifest can rely on run
  identity; today a renamed run breaks consumers silently.

### NOT-GATES (considered, deliberately not proposed)

- Vulnerability scanning of published GHCR images — real gap, but it is CD/image-economy territory (Part A
  follow-up), not MLflow object control; belongs in a CD-scoped proposal, not PROPOSE-GH5.
- `should_promote` threshold tuning — already a declared gate (src/broadway/evaluate/promotion.py).
- Champion manifest strictness — scripts/check_champion_manifest.sh explicitly declares itself "not a gate";
  keep it reporting-only until retirement day.
- cloudpickle/signature logging format — already governed post-hoc by classify_champion buckets
  (bare_model / pipeline_signature / ambiguous); adding a second gate would duplicate a checked condition.
- kubeconform scope expansion to top-level k8s/*.yaml — CD validation gap (Part A), cross-referenced there.

### Most ungoverned object

The **champion alias** — a single mutable pointer that every production load resolves, writable by any process
with the tracking URI, guarded only by import convention:

```python
# src/broadway/training/mlflow_utils.py:162-164
def promote_candidate(model_name: str, model_uri: str, alias: str = "champion") -> None:
    version = mlflow.register_model(model_uri, model_name)
    mlflow.tracking.MlflowClient().set_registered_model_alias(model_name, alias, version.version)
```

```text
# k8s/optuna/configmap.yaml (worker pods hold full tracking access)
    mlflow:
      tracking_uri: http://mlflow:5000
      experiment: ratecode1_model_battle
      allowed_hosts: "*"   # local kind cluster; tighten for real deployments

# scripts/check_champion_manifest.sh:11-13 (the only reader-side control declines to be a gate)
# Exit 0 always for reporting — the manifest is not a gate. --strict turns it
# into the retirement-condition check: exit 1 when the bare_model or ambiguous
```

---

## Summary counts

- Workflows found: 1 unique file (`ci.yml`) × 6 distinct blobs across 11 refs; **0 cd.yaml files anywhere**
  (repo refs + GitHub Actions registry). CD capability = embedded job in ci.yml.
- Images inventoried: 9 (7 k8s-demanded + 2 compose-built). Built-by-CI: 3 (base private + 2 published).
  UNBUILT: 3 (`broadway`, `broadway-mlflow`, `broadway-postgres`). Upstream-pulled: 1. Compose-local: 2.
  Scanned-by: 0 for every image.
- MLflow creation sites: 10 (across 7 files), object classes created: experiments, runs, params, metrics,
  dataset-lineage inputs, artifacts, logged models, registered versions, aliases. Deletion/retention calls: 0.
  Declared gates touching them: 2 (should_promote threshold; persistence-before-promotion test contract).
- Candidates emitted: 6 PROPOSE-GH5-NN + 5 NOT-GATES.
