# HPO Training — how it works & how to run it

Broadway's hyperparameter optimization is **one config-driven API** shared by the
platform `train` step and the GPU lane's fan-out workers.

## 1. The model of HPO

```
project/config/experiments/amazon_hpo.yaml (hpo block)   # project spec
configs/experiment/hyperopt.yaml (hpo block)             # platform train
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
| Project dataset/experiment | `project/config/dataset/amazon.yaml`, `project/config/experiment/amazon.yaml` |
| Search spaces + budgets (GPU lane) | `project/config/experiments/amazon_hpo.yaml` → `hpo:` |
| Search spaces (platform `train`) | `configs/experiment/hyperopt.yaml` → `hpo:` |
| Allowed params + base defaults per model | `src/broadway/training/models/registry.py` (`MODEL_META`) |
| GPU lane study definition | `gpu_training/config/hpo_kernel.yaml` |

A model's HPO block looks like:

```yaml
hpo:
  engine: optuna
  direction: minimize
  target_metric: rmse
  total_trials: 60
  initial_trials_per_model: 20
  top_k: 2
  models:
    - name: lgbm
      search_space: {n_estimators: [50, 500], max_depth: [3, 12], learning_rate: [0.01, 0.3]}
    - name: xgb
      search_space: {n_estimators: [50, 500], max_depth: [3, 12], learning_rate: [0.01, 0.3], reg_lambda: [1e-3, 10]}
    - name: linear
      search_space: {}   # no tunable hyperparameters
```

Model names are **registry keys** (`linear`, `lgbm`, `xgb`, `rf`); display labels
(e.g. `ols` for `linear`) live in the registry.

## 3. Run paths

| Path | Command | Storage |
|---|---|---|
| Platform `train` + HPO | `uv run ds-pipeline train --dataset amazon --experiment hyperopt` | in-memory |
| GPU lane dry-run (preflight) | `uv run python gpu_training/run_hpo.py --baseline baseline/memory_baseline.json --config gpu_training/config/hpo_run.yaml` | none |
| GPU lane fan-out | the same command with `--execute` | shared URL or local SQLite |

The GPU lane reads the study in `gpu_training/config/hpo_kernel.yaml`: the HPO
spec, the project dataset/experiment names, and the storage policy. Its adapter
(`gpu_training/adapters/project.py`) loads the named project contracts and returns
the canonical `broadway.training.hpo.make_objective`; it never re-implements a fit.

## 4. Distributed k8s stack — retired with the predecessor dataset

The former `k8s/optuna` worker/mini-cluster served the predecessor project's model
battle. It is **not wired to Amazon**: the project HPO worker and its configmap
were removed with the dataset, so the workload-image steps auto-skip when the
project HPO spec is absent. The generic manifests and Dockerfiles remain in-tree
as platform infrastructure; re-enabling a distributed study for Amazon is future
work, not a claim that it runs today.

## 5. Viewing results

**The one command — no cluster involved:**

```bash
uv run python -m project.dashboard
```

Experiment outputs are committed under `project/experiments/results/` and read by
the dashboard (`project/dashboard.py` → `broadway.reports.experiments_dashboard`).

**Per-model results without the UI** — the GPU lane's kernel prints one JSON run
record (study, model, worker, sampler seed, trial count, dataset identity, best
value/params, elapsed seconds).

## 6. Tuning a model

1. Edit the search space in the `hpo:` block (`project/config/experiments/amazon_hpo.yaml`
   or `configs/experiment/hyperopt.yaml`).
2. Run the path you want (platform `train` or the GPU lane).
3. Read the leaderboard / run record; the bandit already shifted budget to the leaders.

Parameter names must be in the model's `allowed_params` (registry) or the config
fails validation.

Search spaces may also tune preprocessing when the experiment declares a
`preprocessing:` recipe: keys prefixed `pre__<step>__<param>` (e.g.
`pre__imputation_0__knn_neighbors`) are applied to the Pipeline, not the
model. Registry validation (`allowed_params`) covers the model segment only;
unknown `pre__` keys fail loud at trial time via `set_params`.
