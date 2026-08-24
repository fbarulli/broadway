#!/usr/bin/env bash
# run_local_ci.sh — SINGLE SOURCE of the platform gate list (D16a pattern).
# ci.yml invokes THIS script; edit the gate list HERE, never in ci.yml —
# editing YAML alone reopens two-source drift (D16d rejected C7).
# CI-only (need docker; live in ci.yml BY NAME): shellcheck + k8s sh -n,
# kubeconform, orchestrator dry-run, build-and-boot. experiments.py verify
# runs on taxi only. Usage: run_local_ci.sh [--static]; exit 0 green / 1 red.
set -euo pipefail
cd "$(dirname "$0")/.."
STATIC=0; [[ "${1:-}" == "--static" ]] && STATIC=1   # doc-only micro edits
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/broadway-mpl}"; mkdir -p "$MPLCONFIGDIR"
fail=0
run() {  # run <name> <cmd...>: loud banner, tail on fail, aggregate, stop never
  local log; log="$(mktemp)"; echo "== $1"
  if "${@:2}" >"$log" 2>&1; then echo "PASS $1"
  else echo "FAIL $1 — tail:"; tail -15 "$log"; fail=1; fi
  rm -f "$log"
}
run parity  bash scripts/check_branch_parity.sh
run ruff    uv run ruff check src tests experiments/mlflow experiments/fare_prediction \
            experiments.py experiments_ui.py project/working.py project/data.py
run mypy    uv run mypy src/broadway
run configs uv run python -c "
from pathlib import Path
from broadway.config.loader import load_config
ps = sorted(Path('configs/experiment').glob('*.yaml')); assert ps, 'no configs'
[load_config('train', dataset='test', experiment=p.stem) or print(f'OK {p.stem}') for p in ps]"
if [[ $STATIC -eq 0 ]]; then
  run pytest uv run pytest tests/ --cov=src/broadway --cov-report=term-missing \
             --cov-fail-under=95
fi
[[ $fail -eq 0 ]] && echo "LOCAL-CI GREEN" || echo "LOCAL-CI RED — fix above before commit/push"
exit "$fail"
