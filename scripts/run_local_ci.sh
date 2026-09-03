#!/usr/bin/env bash
# run_local_ci.sh — SINGLE SOURCE of the platform gate list (D16a pattern).
# ci.yml invokes THIS script; edit the gate list HERE, never in ci.yml —
# editing YAML alone reopens two-source drift (D16d rejected C7).
#
# CI-only (need docker; live in ci.yml BY NAME):
#   kubeconform, orchestrator dry-run, build-and-boot.
#
# experiments.py verify runs on the project branch only.
#
# The full tier additionally runs project-tests (project/tests, collected
# WITHOUT --cov; the >=95 floor stays on tests/).
#
# Usage:
#   run_local_ci.sh [--static|--tier=fast|--tier=full] [--clean-lint]
#                   [-h|--help]
#
# Flags combine:
#   --tier=fast --clean-lint
#
# --clean-lint:
#   ruff+mypy scan a throwaway HEAD worktree snapshot instead of this possibly
#   dirty tree (WIP-immune); every other gate stays tree-bound.
#
# Alone, --clean-lint keeps its historical coupling to the FULL tier
# (pytest itself stays tree-bound).
#
# CONFLICT RULE:
#   --static together with any --tier=X is refused loudly (exit 2).
#
# Exit codes:
#   0 = green
#   1 = gate failure
#   2 = usage/configuration error

set -Eeuo pipefail

cd "$(dirname "$0")/.."

STATIC=0
TIER="full"
CLEAN_LINT=0
SEEN_STATIC=0
SEEN_TIER=0
fail=0

# ---------------------------------------------------------------------------
# ERROR / USAGE
# ---------------------------------------------------------------------------

# shellcheck disable=SC2317  # reached via the ERR trap
on_err() {
  local rc=$?
  echo "LOCAL-CI FAILED: unexpected command failure (exit $rc) at line ${BASH_LINENO[0]:-unknown}." >&2
  exit "$rc"
}
trap on_err ERR

usage() {
  cat <<'EOF'
usage: run_local_ci.sh [--static|--tier=fast|--tier=full] [--clean-lint] [-h|--help]

  --static                 Run static/doc-only gate set.
  --tier=fast              Run fast tier gates.
  --tier=full              Run full tier gates (default).
  --clean-lint             Run ruff+mypy against a pristine HEAD snapshot.
  -h, --help               Show this help.

Flags combine:
  --tier=fast --clean-lint

--clean-lint alone retains its historical coupling to the FULL tier.

CONFLICT:
  --static together with --tier=X is refused (exit 2).

Exit codes:
  0 = green
  1 = gate failure
  2 = usage/configuration error
EOF
}

usage_error() {
  echo "REFUSED: $*" >&2
  usage >&2
  exit 2
}

# ---------------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------------

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;

    --static)
      if [[ "$SEEN_TIER" -eq 1 ]]; then
        usage_error "--static conflicts with already-parsed --tier=$TIER"
      fi
      STATIC=1
      SEEN_STATIC=1
      ;;

    --tier=fast|--tier=full)
      if [[ "$SEEN_STATIC" -eq 1 ]]; then
        usage_error "$arg conflicts with already-parsed --static"
      fi
      TIER="${arg#--tier=}"
      SEEN_TIER=1
      ;;

    --clean-lint)
      CLEAN_LINT=1
      ;;

    *)
      usage_error "unknown argument: '$arg'"
      ;;
  esac
done

# Defensive invariant. The parser above should make this unreachable.
if [[ "$STATIC" -eq 1 && "$SEEN_TIER" -eq 1 ]]; then
  usage_error "--static and --tier=$TIER are mutually exclusive"
fi

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

FAST_BANNERS="FAST-GREEN"
FULL_BANNERS="LOCAL-CI GREEN"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/broadway-mpl}"
mkdir -p "$MPLCONFIGDIR" || {
  echo "LOCAL-CI REFUSED: cannot create MPLCONFIGDIR=$MPLCONFIGDIR" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# GATE RUNNER
# ---------------------------------------------------------------------------
#
# run <name> <command...>
#
# Every gate:
#   - gets an explicit banner;
#   - captures output;
#   - prints a useful tail on failure;
#   - contributes to aggregate status;
#   - never stops subsequent gates.
#
# This intentionally does NOT use `set -e` semantics for the gate command:
# failures are data here, not shell-control-flow events.

run() {
  local name="$1"
  shift

  local log
  log="$(mktemp "${TMPDIR:-/tmp}/broadway-ci.XXXXXX")"

  echo "== $name"

  if "$@" >"$log" 2>&1; then
    echo "PASS $name"
  else
    echo "FAIL $name — tail:" >&2
    tail -40 "$log" >&2
    fail=1
  fi

  rm -f "$log"
}

# ---------------------------------------------------------------------------
# CLEAN-LINT SNAPSHOT
# ---------------------------------------------------------------------------
#
# Opt-in:
#   ruff + mypy execute against pristine HEAD rather than the dirty worktree.
#
# The snapshot lives outside the repository so WIP cannot affect lint.
#
# pytest deliberately remains tree-bound.

CLEAN_SNAP=""

# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
cleanup_clean_snapshot() {
  local rc=0

  if [[ -n "$CLEAN_SNAP" ]]; then
    if [[ -d "$CLEAN_SNAP/head" ]]; then
      git worktree remove --force "$CLEAN_SNAP/head" \
        >/dev/null 2>&1 || rc=1
    fi

    rm -rf "$CLEAN_SNAP" || rc=1
  fi

  return "$rc"
}

trap cleanup_clean_snapshot EXIT

# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
ensure_clean_snapshot() {
  [[ -n "$CLEAN_SNAP" ]] && return 0

  CLEAN_SNAP="$(mktemp -d "${TMPDIR:-/tmp}/broadway-clean-lint.XXXXXX")"

  local head="$CLEAN_SNAP/head"

  git worktree add --detach "$head" HEAD >/dev/null

  echo \
    "== clean-lint: HEAD @ $(git rev-parse --short HEAD) -> $head (worktree WIP excluded)" \
    >&2
}

# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
in_clean_snapshot() {
  ensure_clean_snapshot

  (
    cd "$CLEAN_SNAP/head" || exit 1
    exec env \
      UV_PROJECT_ENVIRONMENT="$CLEAN_SNAP/venv" \
      "$@"
  )
}

# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
dispatch() {
  if [[ "$CLEAN_LINT" -eq 1 ]]; then
    in_clean_snapshot "$@"
  else
    "$@"
  fi
}

# Build eagerly. This ensures --clean-lint is visible even when ruff/mypy
# subsequently pass and their output is captured by `run`.
if [[ "$CLEAN_LINT" -eq 1 ]]; then
  ensure_clean_snapshot
fi

# ---------------------------------------------------------------------------
# F1b PARITY GUARD
# ---------------------------------------------------------------------------
#
# The parity gate must NOT trust the tree-local checker.
#
# The track ref is declared exactly once:
#
#   PARITY_TRACK_BRANCH=...
#
# in check_branch_parity.sh.
#
# We retrieve that checker from origin/<track>, verify that it is a
# post-D16 checker, and execute THAT pinned copy.
#
# Rename-day law:
#   edit the ONE PARITY_TRACK_BRANCH declaration in check_branch_parity.sh.
#
# No fallback to the local checker is permitted.

# shellcheck disable=SC2317  # reached via `run parity gate_parity` indirection
gate_parity() {
  local track
  local dest
  local rc

  track="$(
    sed -n \
      's/^PARITY_TRACK_BRANCH=\([A-Za-z0-9_.-]*\).*/\1/p' \
      scripts/check_branch_parity.sh |
      head -1
  )"

  if [[ ! "$track" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo \
      "FAIL parity (F1b): no parsable PARITY_TRACK_BRANCH declaration in tree-local checker" \
      >&2
    return 1
  fi

  dest="$(mktemp "${TMPDIR:-/tmp}/f1b_parity.XXXXXX")"

  if ! git show \
      "refs/remotes/origin/$track:scripts/check_branch_parity.sh" \
      >"$dest" 2>/dev/null; then
    echo \
      "FAIL parity (F1b): origin/$track unavailable — cannot pin checker" \
      >&2
    rm -f "$dest"
    return 1
  fi

  if ! grep -q '^PARITY_ERA=' "$dest"; then
    echo \
      "FAIL parity (F1b): legacy pre-D16 checker on track ref" \
      >&2
    rm -f "$dest"
    return 1
  fi

  if bash "$dest"; then
    rc=0
  else
    rc=$?
  fi

  rm -f "$dest"
  return "$rc"
}

run parity gate_parity

# ---------------------------------------------------------------------------
# STATIC / FAST / FULL GATES
# ---------------------------------------------------------------------------

# project/experiments mirrors project/config/layout.yaml `experiments` SSOT.
run ruff \
  dispatch \
  bash scripts/uv.sh run --extra dev \
  ruff check src tests project/experiments scripts

run mypy \
  dispatch \
  bash scripts/uv.sh run --extra dev \
  mypy src/broadway

run vulture \
  dispatch \
  bash scripts/uv.sh run --extra dev \
  vulture src/broadway project scripts --min-confidence 95

run configs \
  bash scripts/uv.sh run --extra dev \
  python -c '
from pathlib import Path
from broadway.config.loader import load_config

paths = sorted(Path("configs/experiment").glob("*.yaml"))
assert paths, "no configs"

for path in paths:
    load_config("train", dataset="test", experiment=path.stem)
    print(f"OK {path.stem}")
'

# Gate-divergence law:
# Keep this command-identical to ci.yml's "Shell scripts" step.
# shellcheck disable=SC2016  # single quotes intended: globs expand under bash -c
run shell-scripts \
  bash -c '
    set -euo pipefail
    shopt -s nullglob
    files=(scripts/*.sh)
    (( ${#files[@]} > 0 )) || {
      echo "no scripts/*.sh files found" >&2
      exit 1
    }
    for f in "${files[@]}"; do
      bash -n "$f"
    done
    shellcheck "${files[@]}"
  '

# data-refs enforces build/deploy SSOT:
# project/config/layout.yaml build:*.
run data-refs \
  bash scripts/uv.sh run --extra dev \
  python scripts/check_data_refs.py

# graphify reconciles callable symbols against gates.yaml owners.
run graphify \
  bash scripts/uv.sh run --extra dev \
  python scripts/check_graphify_surfaces.py

# ---------------------------------------------------------------------------
# FULL-TIER TESTS
# ---------------------------------------------------------------------------
#
# --static deliberately excludes pytest.
# --tier=fast deliberately excludes pytest.
# --tier=full runs both test suites.
#
# --clean-lint only changes ruff/mypy; tests remain tree-bound.

if [[ "$STATIC" -eq 0 && "$TIER" == "full" ]]; then
  run pytest \
    bash scripts/uv.sh run --extra dev \
    pytest tests/ \
      -n 4 \
      --dist worksteal \
      --cov=src/broadway \
      --cov-report=term-missing \
      --cov-fail-under=95

  run project-tests \
    bash scripts/uv.sh run --extra dev \
    pytest project/tests \
      -q \
      --dist worksteal
fi

# ---------------------------------------------------------------------------
# RESULT
# ---------------------------------------------------------------------------

if [[ "$fail" -eq 0 ]]; then
  CL_NOTE=""
  if [[ "$CLEAN_LINT" -eq 1 ]]; then
    CL_NOTE=" + clean-lint(ruff+mypy@HEAD-snapshot)"
  fi

  if [[ "$STATIC" -eq 1 ]]; then
    echo "STATIC-GREEN (tiers: parity/ruff/mypy/vulture/configs/shell-scripts/data-refs/graphify)$CL_NOTE"
  elif [[ "$TIER" == "fast" ]]; then
    echo "$FAST_BANNERS (tiers: parity/ruff/mypy/vulture/configs/shell-scripts/data-refs/graphify)$CL_NOTE"
  else
    echo "$FULL_BANNERS$CL_NOTE"
  fi
else
  echo "LOCAL-CI RED — fix above before commit/push" >&2
fi

exit "$fail"
