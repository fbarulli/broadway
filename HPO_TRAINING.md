# HPO Training — how it works & how to view results

Broadway's hyperparameter optimization is **one config-driven API** shared by the
platform `train` step, the model battle, and the distributed k8s workers — with an
**on-demand lifecycle**: the cluster exists only while training runs, snapshots the
results on finish, and restores them on the next call.

## 1. The model of HPO

```
configs/experiments/mlflow.yaml (hpo block)
        │  search spaces + budgets
        ▼
src/broadway/training/hpo.py  (the unified API)
  make_objective      one objective: train via the model registry, score on val
  run_model_study     per-model Optuna study (TPE, deterministically seeded)
  bandit_allocate     top-k models split the remaining budget
  run_hpo             round 1: initial trials per model (parallel)
                      round 2: bandit gives the rest to the leaders
```

- **One objective, one search space per model** — no hardcoded ranges in code.
- **Deterministic**: each study's sampler is seeded at construction
  (`TPESampler(seed=…)`), so `(study_name, random_state)` reproduces the same
  trajectory, including on resume.
- **Interruption-safe**: studies persist to RDB storage with heartbeats; a worker
  killed mid-trial has its stale `RUNNING` trial recovered on the next run.

## 2. Where the configuration lives

| Concern | Location |
|---|---|
| Search spaces + budgets (k8s / battle) | `configs/experiments/mlflow.yaml` → `hpo:` (models, spaces, `total_trials`, `initial_trials_per_model`, `top_k`, `target_metric`) |
| Search spaces (platform `train`) | `configs/experiment/hyperopt.yaml` → `hpo:` |
| Allowed params + base defaults per model | `src/broadway/training/models/registry.py` (`MODEL_META`) |
| k8s infra only (DB URLs, mlflow URI, dataset path) | `k8s/optuna/configmap.yaml` |

A model's HPO block looks like:

```yaml
hpo:
  engine: optuna
  direction: minimize
  target_metric: mae
  total_trials: 100
  initial_trials_per_model: 20
  top_k: 2
  models:
    - name: lgbm
      search_space: {n_estimators: [50, 200], max_depth: [3, 8], learning_rate: [0.05, 0.3]}
    - name: xgb
      search_space: {n_estimators: [50, 200], max_depth: [3, 8], learning_rate: [0.05, 0.3], reg_lambda: [1e-3, 10]}
    - name: linear
      search_space: {}   # no tunable hyperparameters
```

Model names are **registry keys** (`linear`, `lgbm`, `xgb`, `rf`); display labels
(e.g. `ols` for `linear`) live in the registry.

## 3. Run paths

| Path | Command | Storage |
|---|---|---|
| Platform `train` + HPO | `uv run ds-pipeline train --dataset … --experiment hyperopt` | in-memory |
| Model battle (bandit demo) | `uv run python experiments/mlflow/01_model_battle_mlflow.py` | in-memory |
| Distributed worker (one study per model) | `uv run python experiments/mlflow/03_optuna_worker.py --model {lgbm,ols,xgb} --config …` | shared Postgres (k8s) or `sqlite:///…` locally |

## 4. The on-demand lifecycle

The k8s stack (kind cluster: postgres + mlflow + optuna workers) is **not kept
running**. `k8s/optuna/lifecycle.sh` activates it for a run and tears it down
after, snapshotting the results:

```bash
k8s/optuna/lifecycle.sh train   # up → restore → run HPO → dump → down
k8s/optuna/lifecycle.sh up      # create cluster (if absent), restore snapshot
k8s/optuna/lifecycle.sh dump    # snapshot DBs + artifacts (no teardown)
k8s/optuna/lifecycle.sh down    # dump, then delete the cluster
```

Durable state lives in `data/optuna-backup/` (gitignored):

```
optuna.sql.gz           the Optuna studies + trials (resume-able)
mlflow.sql.gz           mlflow run metadata (params, metrics)
mlflow-artifacts.tar.gz models, figures, artifacts
```

The next `train`/`up` restores these (custom-format dumps via `pg_restore`,
artifacts via `tar`), so studies **resume across cycles** — verified: after a full
teardown + recreate + restore, `ratecode1_lgbm` still had its 40 trials, `ols` 43,
`xgb` 40.

The worker image mirrors the repo layout (single source of truth — no remapped
paths); workers auto-delete ~2 min after finishing (`ttlSecondsAfterFinished`).

## 4a. CI boundary (what is and isn't automated)

- **CI — structural determinism, no cluster, no training:** the platform gates
  (ruff/mypy/pytest+coverage, shellcheck, kubeconform, frozen lockfile) plus the
  **Build & Boot** job — builds the three images and boots the worker container
  (imports, CLI, config-load simulation) to catch layout/import bugs.
- **Manual — runtime semantics:** `lifecycle.sh train` (kind cluster + workers +
  dump/restore) runs only on demand. **There is no scheduled retraining, by
  design** — HPO answers the question you ask, when you ask it.

## 5. Viewing mlflow results

**The one command — no Kubernetes involved:**

```bash
k8s/optuna/lifecycle.sh view
# restores the snapshot into a throwaway docker postgres, runs mlflow locally,
# and prints:  mlflow UI (all dumped results): http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` — you'll see every completed run (params + metrics)
from the snapshot (`data/optuna-backup/`), and the Optuna studies are restored in
the same postgres. Stop when done:

```bash
docker rm -f broadway-view-pg          # the throwaway postgres
kill $(pgrep -f 'mlflow server.*15433')  # the local mlflow
```

Notes:
- The cluster is **never** involved in viewing — it only exists for distributed
  training (`train`/`up`/`down`).
- Runs + metrics always display. Model **artifact links may 404** — the dump
  records the pod's `/mlflow` paths, which a local server can't resolve; the
  numeric results are unaffected.
- While the cluster happens to be up (during `train`), the same runs are also at
  `http://<node-ip>:30500` / `kubectl port-forward svc/mlflow 5000:5000`.

**Per-model results without the UI** — the worker logs print the best run:

```bash
kubectl logs job/optuna-lgbm | grep DONE
# [worker] DONE model=lgbm best_mae=2.0041 params={...}
```

**Committed evidence** — the model battle writes a metrics CSV + leaderboard to
`experiments/results/mlflow/` (tracked).

## 6. Tuning a model

1. Edit the search space in the `hpo:` block (`configs/experiments/mlflow.yaml` or
   `configs/experiment/hyperopt.yaml`).
2. Run the path you want (local battle, `train`, or `lifecycle.sh train`).
3. Read the leaderboard / UI; the bandit already shifted budget to the leaders.

Parameter names must be in the model's `allowed_params` (registry) or the config
fails validation.
