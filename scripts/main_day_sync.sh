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

# Evidence: main has its own synthetic 5KB blob + provenance sidecar
mkdir -p "$TMPDIR/evidence"
cp experiments/results/univariate/sample_evidence/sample_evidence.parquet "$TMPDIR/evidence/" 2>/dev/null || true
cp experiments/results/univariate/sample_evidence/sample_evidence.json "$TMPDIR/evidence/" 2>/dev/null || true

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
  configs/step/ \
  configs/experiment/ \
  2>&1 | tail -3

# --- 4) Restore main's slate (overwrite what dev brought)
echo "==> Restoring main's slate"

# a) README (main owns)
cp "$TMPDIR/README.md" README.md

# b) Governance pointer (main owns)
cp "$TMPDIR/GOVERNANCE-POINTER.md" GOVERNANCE-POINTER.md

# c) Main's generic dataset bindings (configs/experiments/)
mkdir -p configs/experiments
cat > configs/experiments/working.yaml <<'YAML'
parquet: experiments/results/univariate/sample_evidence/sample_evidence.parquet
columns:
  target: target
  pickup_datetime: pickup_datetime
  dropoff_datetime: dropoff_datetime
min_target_value: 0.0
max_duration_minutes: 240
time_buckets:
  day: 0.0
  peak: 0.0
  overnight: 0.0
time_bucket_default: day
YAML

# d) Main's generic project bindings (project/working.py only)
mkdir -p project
cat > project/working.py <<'PY'
from pathlib import Path
import yaml

_CFG = None
def _load_cfg():
    global _CFG
    if _CFG is None:
        with open("configs/experiments/working.yaml") as f:
            _CFG = yaml.safe_load(f)
    return _CFG

def require_keys(d, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")

def load_working():
    cfg = _load_cfg()
    require_keys(cfg, ["parquet", "columns", "min_target_value", "max_duration_minutes", "time_buckets", "time_bucket_default"])
    require_keys(cfg["columns"], ["target", "pickup_datetime", "dropoff_datetime"])
    df = __import__("pandas").read_parquet(cfg["parquet"])
    TARGET_COL = cfg["columns"]["target"]
    PICKUP_DATETIME_COL = cfg["columns"]["pickup_datetime"]
    DROPOFF_DATETIME_COL = cfg["columns"]["dropoff_datetime"]
    df = df.rename(columns={
        PICKUP_DATETIME_COL: "pickup_datetime",
        DROPOFF_DATETIME_COL: "dropoff_datetime",
        TARGET_COL: "target",
    })
    MIN_TARGET_VALUE = float(cfg["min_target_value"])
    return df[df["target"] > MIN_TARGET_VALUE]

def load_metered():
    cfg = _load_cfg()
    require_keys(cfg, ["parquet", "columns", "min_target_value", "max_duration_minutes", "time_buckets", "time_bucket_default"])
    require_keys(cfg["columns"], ["target", "pickup_datetime", "dropoff_datetime"])
    df = __import__("pandas").read_parquet(cfg["parquet"])
    TARGET_COL = cfg["columns"]["target"]
    PICKUP_DATETIME_COL = cfg["columns"]["pickup_datetime"]
    DROPOFF_DATETIME_COL = cfg["columns"]["dropoff_datetime"]
    df = df.rename(columns={
        PICKUP_DATETIME_COL: "pickup_datetime",
        DROPOFF_DATETIME_COL: "dropoff_datetime",
        TARGET_COL: "target",
    })
    return df

def time_buckets():
    cfg = _load_cfg()
    return cfg["time_buckets"], cfg["time_bucket_default"]
PY

# e) Synthetic evidence (restore main's 5KB blob)
mkdir -p experiments/results/univariate/sample_evidence
cp "$TMPDIR/evidence/sample_evidence.parquet" experiments/results/univariate/sample_evidence/
cp "$TMPDIR/evidence/sample_evidence.json" experiments/results/univariate/sample_evidence/

# --- 5) Stage everything for one atomic commit
git add -A

echo "==> DONE. Working tree ready. Review: git diff --cached --stat"
echo "==> Next: run full CI (bash scripts/run_local_ci.sh), then commit + push."