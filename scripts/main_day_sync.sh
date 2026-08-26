#!/usr/bin/env bash
# MAIN-DAY SYNC — resync main from dev (sklearn) with slate exceptions
# Run from main worktree root. Requires: git, python3, uv sync'd.
# Safe to run multiple times; idempotent where possible.

set -euo pipefail

echo "==> MAIN-DAY SYNC: resyncing main from sklearn"

# --- 0) Preconditions
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repo" >&2; exit 1
fi
current_branch=$(git symbolic-ref --short HEAD)
if [[ "$current_branch" != "main" ]]; then
  echo "Must be on main branch (currently on $current_branch)" >&2; exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "Working tree not clean — commit or stash first" >&2; exit 1
fi

# --- 1) Snapshot main-only files BEFORE we overwrite them
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
echo "==> Snapshotting main-only files to $TMPDIR"
cp README.md "$TMPDIR/README.md" 2>/dev/null || true
cp GOVERNANCE-POINTER.md "$TMPDIR/GOVERNANCE-POINTER.md" 2>/dev/null || true
# Snapshot main's synthetic evidence blob + sidecar (5KB parquet)
mkdir -p "$TMPDIR/evidence"
cp experiments/results/univariate/sample_evidence/sample_evidence.parquet "$TMPDIR/evidence/" 2>/dev/null || true
cp experiments/results/univariate/sample_evidence/sample_evidence.json "$TMPDIR/evidence/" 2>/dev/null || true

# --- 2) Fetch dev and checkout its full tree
echo "==> Fetching origin/sklearn and checking out full tree"
git fetch origin sklearn
git checkout origin/sklearn -- .

# --- 3) Delete the taxi use-case payload (dev owns it; main is blank slate)
echo "==> Deleting use-case payload"
git rm -rf --ignore-unmatch \
  experiments/univariate \
  experiments/multivariate \
  experiments/fare_prediction \
  experiments/polynomial_regression_et_all \
  experiments/mlflow/_common.py \
  project/data.py \
  project/working.py \
  configs/experiments \
  HPO_TRAINING.md \
  SKLEARN_PIPELINES.md \
  BROADWAY.md \
  dataflow.md \
  2>/dev/null || true

# --- 4) Restore main's slate (its own files)
echo "==> Restoring main's slate"

# a) Governance
cp "$TMPDIR/GOVERNANCE-POINTER.md" GOVERNANCE-POINTER.md

# b) Generic dataset configs
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

# c) Project bindings (generic, reads working.yaml above)
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

# d) Synthetic evidence blob + provenance (restore main's 5KB blob)
mkdir -p experiments/results/univariate/sample_evidence
cp "$TMPDIR/evidence/sample_evidence.parquet" experiments/results/univariate/sample_evidence/
cp "$TMPDIR/evidence/sample_evidence.json" experiments/results/univariate/sample_evidence/

# e) README (main's governance + platform description)
cp "$TMPDIR/README.md" README.md

echo "==> Slate restored. Next: run full CI on this tree, then commit + push."