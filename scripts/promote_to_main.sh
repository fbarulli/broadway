#!/usr/bin/env bash
# promote_to_main.sh — DETERMINISTIC single-command taxi->main promotion.
# Reads the shared surface from scripts/main_whitelist.txt via
# configs/tooling.yaml (same SSOT as check_branch_parity.sh + main_day_sync.sh).
# Anchor bumps live on TAXI ONLY (syncing one to main reads as drift).
#
# Usage:
#   scripts/promote_to_main.sh [--dry-run] [--execute --main-worktree <path>] [-h|--help]
#   Default is --dry-run: fetch, whitelist diffstat, parity custody, fast gates.
#   --execute: run main_day_sync.sh inside <path> (a clean main checkout),
#     re-run fast gates there, stage for one atomic commit, and print the
#     exact taxi-side anchor-bump line. Committing/pushing stays with the
#     main agent per custody (this script never commits or pushes).
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=1
MAIN_WORKTREE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --execute) DRY_RUN=0 ;;
    --main-worktree=*) MAIN_WORKTREE="${arg#--main-worktree=}" ;;
    -h|--help)
      echo "usage: $0 [--dry-run] [--execute --main-worktree <path>]"
      echo "  --dry-run (default): report only, no mutations"
      echo "  --execute: sync <path> (clean main checkout), fast-gate, stage, print anchor bump"
      exit 0 ;;
    *)
      if [[ -z "$MAIN_WORKTREE" && "$arg" != -* ]]; then MAIN_WORKTREE="$arg";
      else echo "unknown argument: '$arg'" >&2; exit 2; fi ;;
  esac
done

WHITELIST_FILE="$(python3 -c 'import yaml; print(yaml.safe_load(open("configs/tooling.yaml"))["shared_surface"]["file"])' 2>/dev/null || echo scripts/main_whitelist.txt)"
[[ -f "$WHITELIST_FILE" ]] || { echo "FATAL: whitelist not found: $WHITELIST_FILE" >&2; exit 1; }
mapfile -t WHITELIST < <(grep -v '^\s*#' "$WHITELIST_FILE" | grep -v '^\s*$' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ ${#WHITELIST[@]} -gt 0 ]] || { echo "FATAL: whitelist empty" >&2; exit 1; }

echo "==> promote_to_main ($([ $DRY_RUN -eq 1 ] && echo dry-run || echo execute)): ${#WHITELIST[@]} whitelist entries from $WHITELIST_FILE"
git fetch origin main taxi --quiet

echo "==> whitelist diffstat origin/main..origin/taxi"
git diff --stat "origin/main" "origin/taxi" -- "${WHITELIST[@]}" || true

echo "==> parity custody (frozen-main, track=taxi)"
bash scripts/check_branch_parity.sh || { echo "PROMOTE REFUSED: parity red" >&2; exit 1; }

echo "==> pyright consultant (strict, advisory)"
bash scripts/pyright_advisory.sh

echo "==> fast gates (parity/ruff/mypy/pyright-advisory/vulture/configs/shell)"
bash scripts/run_local_ci.sh --tier=fast || { echo "PROMOTE REFUSED: fast tier red" >&2; exit 1; }

if [[ $DRY_RUN -eq 1 ]]; then
  echo "==> DRY-RUN COMPLETE. To execute: $0 --execute --main-worktree <clean-main-checkout>"
  echo "    Next (deterministic): run main_day_sync.sh in the main worktree, fast-gate there,"
  echo "    commit ONE atomic sync, push main, then bump PARITY_MAIN_ANCHOR on taxi only."
  exit 0
fi

[[ -n "$MAIN_WORKTREE" ]] || { echo "FATAL: --execute requires --main-worktree <path>" >&2; exit 2; }
[[ -d "$MAIN_WORKTREE" ]] || { echo "FATAL: no such worktree: $MAIN_WORKTREE" >&2; exit 1; }
if [[ "$(git -C "$MAIN_WORKTREE" symbolic-ref --short HEAD)" != "main" ]]; then
  echo "FATAL: $MAIN_WORKTREE is not on main" >&2; exit 1
fi
if [[ -n "$(git -C "$MAIN_WORKTREE" status --porcelain)" ]]; then
  echo "FATAL: $MAIN_WORKTREE not clean — commit or stash first" >&2; exit 1
fi

echo "==> executing main_day_sync.sh in $MAIN_WORKTREE"
bash -c "cd \"$MAIN_WORKTREE\" && bash \"$PWD/scripts/main_day_sync.sh\""
echo "==> fast gates in main worktree"
bash -c "cd \"$MAIN_WORKTREE\" && bash \"$PWD/scripts/run_local_ci.sh\" --tier=fast" || {
  echo "PROMOTE HALTED: fast tier red in main worktree — fix before commit" >&2; exit 1
}
NEW_TIP="$(git -C "$MAIN_WORKTREE" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
echo "==> STAGED for one atomic commit in $MAIN_WORKTREE. Review: git -C \"$MAIN_WORKTREE\" diff --cached --stat"
echo "==> NEXT (taxi checkout ONLY, never on main): bump the anchor to the ratified sync tip:"
echo "    sed -i 's/^PARITY_MAIN_ANCHOR=.*/PARITY_MAIN_ANCHOR=<new-main-tip>  # <reason>/' scripts/check_branch_parity.sh"
echo "    (current main HEAD here: $NEW_TIP — re-resolve AFTER the sync commit lands)"
