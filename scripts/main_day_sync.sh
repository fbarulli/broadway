#!/usr/bin/env bash
# MAIN-DAY SYNC — resync main from dev (taxi) via WHITELIST
# Run from main worktree root. Requires: git, bash.
# Design: copy only what main should contain, then re-apply main-only files.
# Idempotent: safe to re-run.
# WHITELIST mirrors the SHARED surface in scripts/check_branch_parity.sh —
# keep the two lists in lockstep.

set -euo pipefail

echo "==> MAIN-DAY SYNC: resyncing main from taxi (whitelist)"

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

# README: main has its own (blank-slate); preserve it
cp README.md "$TMPDIR/README.md" 2>/dev/null || true

# Governance slate (main-owned, never shared)
for f in GOVERNANCE-POINTER.md BROADWAY.md AGENT_CONTRACT.md AGENT_WORKER_CONTRACT.md CONTRACT_TEMPLATE.md; do
  cp "$f" "$TMPDIR/$f" 2>/dev/null || true
done

# --- 2) Fetch dev
echo "==> Fetching origin/taxi"
git fetch origin taxi

# --- 3) WHITELIST: only what main should contain
echo "==> Checking out platform surface from taxi (whitelist)"
# Platform core + tests + demo + generic governance + shared infra
git checkout origin/taxi -- \
  src/ \
  tests/ \
  demo/ \
  scripts/ \
  configs/ \
  agents/contracts/ \
  agents/tools/ \
  k8s/ \
  docker/ \
  .github/workflows/ \
  pyproject.toml \
  uv.lock \
  pyrightconfig.json \
  Dockerfile \
  docker-compose.yml \
  .python-version \
  .env.example \
  HPO_TRAINING.md \
  .gitignore \
  .dockerignore \
  2>&1 | tail -5

# --- 4) Restore main's slate (overwrite what dev brought)
echo "==> Restoring main's slate"

# a) README + governance slate (main owns)
cp "$TMPDIR/README.md" README.md
for f in GOVERNANCE-POINTER.md BROADWAY.md AGENT_CONTRACT.md AGENT_WORKER_CONTRACT.md CONTRACT_TEMPLATE.md; do
  cp "$TMPDIR/$f" "$f" 2>/dev/null || true
done

# b) Remove development-only bindings. ``project/`` is dataset-specific and
# belongs only on the development line; main keeps the generic demo SampleSpec.
# reports/, readmore/, read.md, synth.md, TODO.md, project.md are reference-
# use-case artifacts — never on the data-agnostic line.
git rm -r --ignore-unmatch project reports readmore experiments configs/project configs/experiments beads
git rm -f --ignore-unmatch experiments.py experiments_ui.py project.md read.md synth.md TODO.md dataflow.md

# --- 5) Stage everything for one atomic commit
git add -A

echo "==> DONE. Working tree ready. Review: git diff --cached --stat"
echo "==> Next: run full CI (bash scripts/run_local_ci.sh), then commit + push."
