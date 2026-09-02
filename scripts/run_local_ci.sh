#!/usr/bin/env bash
# run_local_ci.sh — SINGLE SOURCE of the platform gate list (D16a pattern).
# ci.yml invokes THIS script; edit the gate list HERE, never in ci.yml —
# editing YAML alone reopens two-source drift (D16d rejected C7).
# CI-only (need docker; live in ci.yml BY NAME): kubeconform,
# orchestrator dry-run, build-and-boot. experiments.py verify
# runs on the project branch only. The full tier additionally runs 'project-tests'
# (project/tests, collected WITHOUT --cov; the >=95 floor stays on tests/).
# Usage: run_local_ci.sh [--static|--tier=fast|--tier=full] [--clean-lint]
#        [-h|--help]; flags COMBINE (e.g. --tier=fast --clean-lint);
# exit 0 green / 1 red / 2 usage error. --clean-lint: ruff+mypy scan a
# throwaway HEAD worktree snapshot instead of this possibly-dirty tree
# (WIP-immune); every other gate stays tree-bound. Alone, --clean-lint keeps
# its historical coupling to the FULL tier (pytest itself stays tree-bound).
# CONFLICT RULE: --static together with any --tier=X is refused loudly
# (exit 2) — they select disjoint gate sets by design. Never silent.
set -euo pipefail
cd "$(dirname "$0")/.."
STATIC=0; TIER="full"; CLEAN_LINT=0
SEEN_STATIC=0; SEEN_TIER=0
usage() {  # stdout for -h|--help; error paths call `usage >&2`
  cat <<'EOF'
usage: run_local_ci.sh [--static|--tier=fast|--tier=full] [--clean-lint] [-h|--help]
  flags combine (e.g. --tier=fast --clean-lint); --clean-lint alone = full tier;
  CONFLICT: --static together with --tier=X is refused (exit 2)
EOF
}
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --static)
      if [[ $SEEN_TIER -eq 1 ]]; then
        echo "REFUSED: --static conflicts with already-parsed --tier=$TIER" >&2; usage >&2; exit 2
      fi
      STATIC=1; SEEN_STATIC=1 ;;               # doc-only micro edits
    --tier=fast|--tier=full)
      if [[ $SEEN_STATIC -eq 1 ]]; then
        echo "REFUSED: $arg conflicts with already-parsed --static" >&2; usage >&2; exit 2
      fi
      TIER="${arg#--tier=}"; SEEN_TIER=1 ;;    # fast: parity+ruff+mypy+vulture+configs+shell (<30s) / full: +pytest+cov>=95 + project-tests
    --clean-lint) CLEAN_LINT=1 ;;              # ruff+mypy vs pristine HEAD snapshot (teeth 5)
    *) echo "unknown argument: '$arg'" >&2; usage >&2; exit 2 ;;
  esac
done
FAST_BANNERS="FAST-GREEN"; FULL_BANNERS="LOCAL-CI GREEN"   # distinct vocabularies
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/broadway-mpl}"; mkdir -p "$MPLCONFIGDIR"
fail=0
run() {  # run <name> <cmd...>: loud banner, tail on fail, aggregate, stop never
  local log; log="$(mktemp)"; echo "== $1"
  if "${@:2}" >"$log" 2>&1; then echo "PASS $1"
  else echo "FAIL $1 — tail:"; tail -40 "$log"; fail=1; fi
  rm -f "$log"
}
# --clean-lint machinery (opt-in): snapshot HEAD into a throwaway worktree
# OUTSIDE the repo so concurrent shared-tree WIP cannot false-red/green
# ruff+mypy (ratified teeth 5). Built eagerly at mode start (idempotent) and
# torn down by the EXIT trap; pytest deliberately stays tree-bound.
CLEAN_SNAP=""
# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
ensure_clean_snapshot() {  # create-once, IN THIS SHELL: a $( ) capture would
                           # discard the CLEAN_SNAP assignment (subshell)
  [[ -n $CLEAN_SNAP ]] && return 0
  CLEAN_SNAP="$(mktemp -d "${TMPDIR:-/tmp}/broadway-clean-lint.XXXXXX")"
  trap 'git worktree remove --force "$CLEAN_SNAP/head" >/dev/null 2>&1 || true; rm -rf "$CLEAN_SNAP"' EXIT
  git worktree add --detach "$CLEAN_SNAP/head" HEAD >/dev/null
  echo "== clean-lint: HEAD @ $(git rev-parse --short HEAD) -> $CLEAN_SNAP/head (worktree WIP excluded)" >&2
}
# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
in_clean_snapshot() {  # in_clean_snapshot <cmd...>: run cmd rooted at the snapshot
  ensure_clean_snapshot
  (
    cd "$CLEAN_SNAP/head" || exit 1
    exec env UV_PROJECT_ENVIRONMENT="$CLEAN_SNAP/venv" "${@}"
  )
}
# shellcheck disable=SC2317  # reached via `run … dispatch …` indirection
dispatch() {  # default: verbatim here; --clean-lint: same command in the snapshot
  if [[ $CLEAN_LINT -eq 1 ]]; then in_clean_snapshot "$@"; else "$@"; fi
}
# Announce + build the snapshot EAGERLY: doing it lazily inside a gate would
# bury the banner in that gate's pass/fail log (green runs must still show
# that --clean-lint engaged — never silent).
[[ $CLEAN_LINT -eq 1 ]] && ensure_clean_snapshot
# F1b guard (D21): the parity gate must NOT trust the tree-local checker —
# a checkout of main (or any stale ref) carries the PRE-D16 legacy script.
# Pin the checker to refs/remotes/origin/sklearn and reject it unless it
# carries the post-D16/D21 inline era declaration (`^PARITY_ERA=` marker).
# shellcheck disable=SC2317  # reached via `run parity gate_parity` indirection
gate_parity() {
  local dest rc
  dest=$(mktemp "${TMPDIR:-/tmp}/f1b_parity.XXXXXX")
  git show refs/remotes/origin/sklearn:scripts/check_branch_parity.sh >"$dest" 2>/dev/null || {
    echo "FAIL parity (F1b): origin/sklearn unavailable — cannot pin checker"; return 1;
  }
  grep -q '^PARITY_ERA=' "$dest" || {
    echo "FAIL parity (F1b): legacy pre-D16 checker on track ref"; rm -f "$dest"; return 1;
  }
  bash "$dest"; rc=$?
  rm -f "$dest"
  return "$rc"
}
run parity gate_parity
# project/experiments mirrors project/config/layout.yaml ``experiments`` (SSOT).
run ruff    dispatch bash scripts/uv.sh run --extra dev ruff check src tests project/experiments \
            scripts
run mypy    dispatch bash scripts/uv.sh run --extra dev mypy src/broadway
run vulture dispatch bash scripts/uv.sh run --extra dev vulture src/broadway project scripts --min-confidence 95
run configs bash scripts/uv.sh run --extra dev python -c "
from pathlib import Path
from broadway.config.loader import load_config
ps = sorted(Path('configs/experiment').glob('*.yaml')); assert ps, 'no configs'
[load_config('train', dataset='test', experiment=p.stem) or print(f'OK {p.stem}') for p in ps]"
# Gate-divergence law: keep command-identical to ci.yml's 'Shell scripts' step.
# shellcheck disable=SC2016  # single quotes intended: globs must expand under bash -c
run shell-scripts bash -c 'for f in scripts/*.sh; do bash -n "$f"; done; shellcheck scripts/*.sh'
# data-refs enforces the build/deploy SSOT (project/config/layout.yaml build:*).
run data-refs bash scripts/uv.sh run --extra dev python scripts/check_data_refs.py
# graphify reconciles callable symbols (graphify-out/graph.json) against
# gates.yaml owners — every governed src/ + project/ file with callables must
# be OWNER-mapped; regenerated via `graphify extract . --code-only --no-cluster`.
run graphify bash scripts/uv.sh run --extra dev python scripts/check_graphify_surfaces.py
if [[ $STATIC -eq 0 && $TIER == "full" ]]; then
  run pytest bash scripts/uv.sh run --extra dev pytest tests/ -n 4 --dist worksteal \
             --cov=src/broadway --cov-report=term-missing --cov-fail-under=95
  run project-tests bash scripts/uv.sh run --extra dev pytest project/tests -q --dist worksteal
fi
if [[ $fail -eq 0 ]]; then
  CL_NOTE=""
  [[ $CLEAN_LINT -eq 1 ]] && CL_NOTE=" + clean-lint(ruff+mypy@HEAD-snapshot)"
  [[ $TIER == "fast" ]] && echo "$FAST_BANNERS (tiers: parity/ruff/mypy/vulture/configs/shell-scripts)$CL_NOTE" \
                        || echo "$FULL_BANNERS$CL_NOTE"
else
  echo "LOCAL-CI RED — fix above before commit/push"
fi
exit "$fail"
