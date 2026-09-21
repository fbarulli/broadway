# Contributing — make changes, test, CI/CD

Single source of truth for gates is `scripts/run_local_ci.sh`. `.github/workflows/ci.yml` calls that script — never edit the gate list in YAML.

## 1. Make changes

- Branch model: `main` (frozen platform) / `taxi` (active line) / feature branches like `time` (branched from `taxi`). Merge back into `taxi` when done.
- Before changing: `git status`, `git diff`, `git log --oneline -10`. Stage only intended files, never secrets (`.env`, `k8s/optuna/secret.local.yaml`).
- One logical change per commit. Green before commit — no broken intermediate states.
- Rules: no hardcoded values (config YAML in `configs/` + `src/broadway/config/schema.py` + env only), data-agnostic `src/broadway/`, shared helpers imported not duplicated, fail loud (no silent `get(key, default)`), tests with every `src/` change.
- Workers (subagents): zero git writes — working-tree changes + report only. Main agent tests, commits, pushes.

```bash
git checkout -b <feature>      # from taxi
# edit ...
git status && git diff
bash scripts/run_local_ci.sh --tier=fast   # before commit
git add <intended files> && git commit -m "SCOPE: what changed"
```

## 2. Testing

Install once:

```bash
bash scripts/uv.sh sync --extra dev
```

Run gates:

| Command | What it runs |
|---|---|
| `bash scripts/run_local_ci.sh --tier=fast` | parity + ruff + mypy + pyright-advisory + vulture + configs + shell + docker-paths (<90s) |
| `bash scripts/run_local_ci.sh` | fast + `pytest tests/ -n 4` with `--cov=src/broadway --cov-fail-under=95` + `project/tests` |
| `bash scripts/run_local_ci.sh --static` | doc-only edits |
| `bash scripts/uv.sh run --extra dev pytest tests/test_x.py -q` | single-file debug only — never a substitute for the full gate |

Coverage floor: ≥95% on `src/broadway` (enforced in the script, not in ad-hoc pytest). `mlruns/`, `artifacts/`, `data/processed/` are gitignored regenerable state — safe to delete when a stale champion skews results.

## 3. CI/CD (`.github/workflows/ci.yml`)

Triggers on push/PR to `main`, `taxi`, `sklearn`; `**/*.md` pushes skip CI. Concurrent runs on the same ref cancel in-flight ones.

- `platform` job: install (`uv sync --all-extras --frozen`) → `run_local_ci.sh` (SSOT) → shellcheck → kubeconform manifests → orchestrator dry-run → taxi-only `project/experiments.py verify`.
- `build-and-boot` job: builds docker images (worker/mlflow/platform/postgres), runs import/CLI/config boot checks. No cluster, never trains. Workload images skip loudly on project-less lines.
- `cd` job (push to `main`/`taxi` only): ships bit-for-bit the tested images to GHCR — `:sha` always, plus `:latest` (main) or `:taxi` (taxi). PRs deploy nothing.

Local red = remote red. Run `bash scripts/run_local_ci.sh` before every push; fix `FAIL` banners locally rather than iterating on CI.
