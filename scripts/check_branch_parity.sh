#!/usr/bin/env bash
# Branch parity gate — keep main and taxi's SHARED surface in lockstep.
#
# The repo has a deliberate split:
#   * sklearn — the only active development line (MAIN_AGENT_CONTRACT.md §2);
#             taxi fast-forwards to it after each green push; main is frozen
#             until declared main-day.
#
# src/, tests/, demo/, scripts/, the synthetic-demo configs, and the deployment
# files (k8s/, docker/, .github/workflows/) are meant to be IDENTICAL on both
# branches. This script fails loudly the moment they drift — including
# deletions and content changes — so a change made on one branch cannot
# silently diverge.
#
# Usage:
#   scripts/check_branch_parity.sh          # check main vs taxi (current repo)
#   scripts/check_branch_parity.sh --sync   # copy taxi's shared surface onto main
#
# Exit codes: 0 = in sync, 1 = drift detected (check), 2 = usage error.

set -euo pipefail

MODE="check"
if [[ "${1:-}" == "--sync" ]]; then
  MODE="sync"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--sync]" >&2
  exit 2
fi

# The intended shared surface — paths that must be byte-identical on both
# branches. Keep this list explicit: anything NOT listed is deliberately
# taxi-only or main-only.
SHARED=(
  src/
  tests/
  demo/
  configs/dataset/test.yaml
  configs/experiment/baseline.yaml
  configs/experiment/engineered.yaml
  configs/experiment/hyperopt.yaml
  configs/analysis/test.yaml
  configs/analysis/test_hypothesis.yaml
  configs/analysis/test_causal.yaml
  configs/step/causal.yaml
  configs/step/etl.yaml
  configs/environment/
  configs/flow/
  k8s/
  docker/
  .github/workflows/
  pyproject.toml
  Dockerfile
  docker-compose.yml
  .gitignore
  .dockerignore
  README.md
  scripts/
)

check() {
  local drifted=0
  local path
  for path in "${SHARED[@]}"; do
    # Byte-identical between the two branch tips (deletions included: if one
    # side lacks the path, git diff reports it and we fail).
    if ! git diff --exit-code --quiet "origin/main" "origin/taxi" -- "$path" 2>/dev/null; then
      echo "DRIFT: $path differs between origin/main and origin/taxi" >&2
      drifted=1
    fi
  done
  if [[ $drifted -ne 0 ]]; then
    echo "PARITY FAILED — shared surface drifted. Run: $0 --sync" >&2
    return 1
  fi
  echo "PARITY OK — shared surface is identical on origin/main and origin/taxi"
}

sync_to_main() {
  local path
  # Work from a clean main checkout against the latest taxi.
  git fetch origin
  git checkout main
  git checkout taxi -- "${SHARED[@]}"
  # Deletions do not propagate with checkout — mirror them too.
  local f
  while IFS= read -r f; do
    if ! git cat-file -e "origin/taxi:$f" 2>/dev/null; then
      git rm -f --ignore-unmatch "$f" >/dev/null 2>&1 || true
    fi
  done < <(git diff --name-only --diff-filter=AD "origin/main" "origin/taxi" -- "${SHARED[@]}" || true)
  echo "SYNCED taxi -> main for shared surface. Review, run gates, commit, push."
}

if [[ "$MODE" == "sync" ]]; then
  sync_to_main
else
  check
fi
