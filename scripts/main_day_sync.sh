#!/usr/bin/env bash
# MAIN-DAY SYNC — resync main from dev (sklearn) via WHITELIST
# Run from main worktree root. Requires: git, bash.
# Design: copy only what main should contain, then re-apply main-only files.
# Idempotent: safe to re-run.

set -euo pipefail

echo "==> MAIN-DAY SYNC: resyncing main from sklearn (whitelist)"

# --- 0) Preconditions
current_branch=$(git symbolic-ref --short HEAD)
if [[ "$current_branch" != "main" ]]; then
  echo "Must be on main branch (currently on $current_branch)" >&2; exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "Working tree not clean — commit or stash first" >&2; exit 1
fi

# --- 1) Snapshot main's slate BEFORE the swap (in case we need to restore)
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
echo "==> Snapshotting main's slate to $TMPDIR"

# README: main has its own (governance-flavored); preserve it
cp README.md "$TMPDIR/README.md" 2>/dev/null || true

# Governance pointer
cp GOVERNANCE-POINTER.md "$TMPDIR/GOVERNANCE-POINTER.md" 2>/dev/null || true

# --- 2) Fetch dev
echo "==> Fetching origin/sklearn"
git fetch origin sklearn

# --- 3) WHITELIST: only what main should contain
echo "==> Checking out platform surface from sklearn (whitelist)"
# Platform core + tests + demo + gates + shared infra
git checkout origin/sklearn -- \
  src/ \
  tests/ \
  demo/ \
  scripts/ \
  pyproject.toml \
  uv.lock \
  Dockerfile \
  docker-compose.yml \
  .github/workflows/ \
  .gitignore \
  .dockerignore \
  2>&1 | tail -5

# Configs (platform-owned, generic)
git checkout origin/sklearn -- \
  configs/dataset/ \
  configs/analysis/ \
  configs/environment/ \
  configs/flow/ \
  configs/sample/demo.yaml \
  configs/step/ \
  configs/experiment/ \
  2>&1 | tail -3

# --- 4) Restore main's slate (overwrite what dev brought)
echo "==> Restoring main's slate"

# a) README (main owns)
cp "$TMPDIR/README.md" README.md

# b) Governance pointer (main owns)
cp "$TMPDIR/GOVERNANCE-POINTER.md" GOVERNANCE-POINTER.md

# c) Remove development-only bindings. ``project/`` is dataset-specific and
# belongs only on the development line; main keeps the generic demo SampleSpec.
git rm -r --ignore-unmatch project experiments configs/project configs/experiments
git rm -f --ignore-unmatch experiments.py experiments_ui.py

# --- 5) Stage everything for one atomic commit
git add -A

echo "==> DONE. Working tree ready. Review: git diff --cached --stat"
echo "==> Next: run full CI (bash scripts/run_local_ci.sh), then commit + push."
