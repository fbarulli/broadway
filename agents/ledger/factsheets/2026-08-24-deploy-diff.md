# FACT SHEET — DEPLOY-DIFF (deployment coherence) — 2026-08-24

Investigator: read-only lane DEPLOY-DIFF @ HEAD 5016e93 (sklearn).
Worktree WIP touches zero deployment paths — tree ≡ committed state for
every k8s/, docker/, configs/, .github/ citation below.

## F-1 verdict: STILL-BROKEN (stub confirmed; two independent breaks)
- src/broadway/inference/api.py:1 is the ENTIRE file — docstring only,
  no app object (committed state identical via git show 5016e93:…).
- k8s/api-deployment.yaml:18-20 invokes bare `inference.api:app`;
  image PYTHONPATH=/app/src exposes package `broadway` only
  (Dockerfile:17) — correct target would be broadway.inference.api:app.
  fastapi/uvicorn ARE installed (pyproject.toml:38-39) — pure wiring.
- api reads model from MLflow registry per docstring yet manifest sets
  no tracking URI and mounts nothing.

## F-2 verdict: STILL-BROKEN
- k8s/train-job.yaml:8-21: no env block, no volumeMounts, runs
  --environment production (:20-21).
- configs/environment/production.yaml:2,6-11 interpolates 7 OS-env vars
  via os.path.expandvars (src/broadway/config/resolver.py:10); unset
  vars pass through literally → database_port int coercion
  (config/schema.py:73) fails at load; ${MLFLOW_TRACKING_URI} feeds
  setup_mlflow (training/module.py:107); no dataset volume mounted.
- Local gates mask it: development defaults carry concrete localhost
  values (loader.py:35; development.yaml:2-11).

## Binding-matrix mismatches (non-matching rows)
| # | REF | PROBLEM |
|---|---|---|
| 1 | postgres-deployment.yaml:25,30,35 → configMap `environment` | no manifest creates it anywhere → CreateContainerConfigError |
| 2 | train-job.yaml env: none vs 7 ${VARS} | MISSING all → crash at config load |
| 3 | api-deployment.yaml env: none vs MLflow URI need | MISSING |
| 4 | HPA cpu 70 (:56) vs api_hpa_cpu_threshold: 80 (production.yaml:16) | MISMATCH — config never wired |
| 5 | mlflow-deployment.yaml:17 broadway-mlflow:latest | no repo build produces that tag (CI tags mlflow-server:<sha>) |
| 6 | postgres-deployment.yaml:18 broadway-postgres:latest | no build source (docker/postgres/ holds only init.sql) |
| 7 | docker-compose.yml:16 build ./docker/postgres | no Dockerfile exists there — BROKEN context |
| 8 | train-job writes artifacts/dataset cwd-relative (discover/module.py:22-23) | no PVC — MISSING durable storage |
| 9 | root Dockerfile broadway:latest consumed by api+train | never built by CI — UNVALIDATED |

Match for record: replicas 2/10 ↔ api_replicas_min/max ✓.

## CI coverage verdict
run_local_ci.sh:6 header claim TRUE (all four docker-only checks exist:
ci.yml :53-62,:71-80,:85-104,:132-212) BUT kubeconform scope is
k8s/optuna/ ONLY (:80) and build-and-boot builds only optuna images
(:163-187). The four top-level manifests are structurally unvalidated
and nothing boots the top-level path — semantic gaps invisible anyway.
**Both findings PASS CI today.**
