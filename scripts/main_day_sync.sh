#!/usr/bin/env bash
# MAIN-DAY SYNC — resync main from dev (taxi) via WHITELIST SSOT
# Run from main worktree root. Requires: git, bash, python3+yaml.
# Design: copy only what main should contain, then re-apply main-only files.
# Idempotent: safe to re-run.
# Deterministic law: WHITELIST is scripts/main_whitelist.txt (via
# configs/tooling.yaml shared_surface.file) — the SAME file the parity
# checker and promote_to_main.sh read. Never maintain a second inline list.

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

# --- 3) WHITELIST SSOT: only what main should contain
echo "==> Checking out platform surface from taxi (whitelist SSOT)"
WHITELIST_FILE="$(python3 -c 'import yaml; print(yaml.safe_load(open("configs/tooling.yaml"))["shared_surface"]["file"])' 2>/dev/null || echo scripts/main_whitelist.txt)"
[[ -f "$WHITELIST_FILE" ]] || { echo "FATAL: whitelist not found: $WHITELIST_FILE" >&2; exit 1; }
mapfile -t WHITELIST < <(grep -v '^\s*#' "$WHITELIST_FILE" | grep -v '^\s*$' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ ${#WHITELIST[@]} -gt 0 ]] || { echo "FATAL: whitelist empty: $WHITELIST_FILE" >&2; exit 1; }
echo "==> Whitelist ($WHITELIST_FILE): ${#WHITELIST[@]} entries"
git checkout origin/taxi -- "${WHITELIST[@]}" 2>&1 | tail -5

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
