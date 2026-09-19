#!/usr/bin/env bash
# promote_to_main.sh — DETERMINISTIC single-command taxi->main promotion.
# Reads the shared surface from scripts/main_whitelist.txt via
# configs/tooling.yaml (same SSOT as check_branch_parity.sh + main_day_sync.sh).
# Anchor bumps live on TAXI ONLY (syncing one to main reads as drift).
#
# Usage:
#   scripts/promote_to_main.sh [--dry-run] [--execute --main-worktree <path>] [--close-out] [-h|--help]
#   Default is --dry-run: fetch, whitelist diffstat, parity custody, fast gates.
#   --execute: run main_day_sync.sh inside <path> (a clean main checkout),
#     re-run fast gates there, stage for one atomic commit, and print the
#     exact taxi-side anchor-bump line. Committing/pushing stays with the
#     main agent per custody (this script never commits or pushes).
#   --close-out (opt-in, requires --execute --main-worktree <path>): AFTER the
#     sync commit has landed on main, re-resolve the tip via
#     `git rev-parse origin/main`, bump PARITY_MAIN_ANCHOR on taxi only,
#     verify parity, then re-run + poll the failed main workflow.
#     This script still never commits or pushes.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=1
MAIN_WORKTREE=""
CLOSE_OUT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --execute) DRY_RUN=0; shift ;;
    --close-out) CLOSE_OUT=1; shift ;;
    --main-worktree=*) MAIN_WORKTREE="${1#--main-worktree=}"; shift ;;
    --main-worktree)
      [[ $# -ge 2 ]] || { echo "FATAL: --main-worktree requires a <path>" >&2; exit 2; }
      MAIN_WORKTREE="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [--dry-run] [--execute --main-worktree <path>] [--close-out]"
      echo "  --dry-run (default): report only, no mutations"
      echo "  --execute: sync <path> (clean main checkout), fast-gate, stage, print anchor bump"
      echo "  --close-out (with --execute --main-worktree <path>): after the sync commit lands on main,"
      echo "    bump PARITY_MAIN_ANCHOR to origin/main, verify parity, re-run + poll main CI"
      exit 0 ;;
    *)
      if [[ -z "$MAIN_WORKTREE" && "$1" != -* ]]; then MAIN_WORKTREE="$1"; shift
      else echo "unknown argument: '$1'" >&2; exit 2; fi ;;
  esac
done

if [[ $CLOSE_OUT -eq 1 && $DRY_RUN -eq 1 ]]; then
  echo "CLOSE-OUT REFUSED: --close-out requires --execute (dry-run mutates nothing)" >&2; exit 2
fi

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

if [[ $CLOSE_OUT -eq 0 ]]; then
  exit 0
fi

# --- Close-out (opt-in): runs ONLY after the sync commit landed on main. ---
# Never commits or pushes; only bumps the taxi-side anchor value + polls CI.
if [[ $DRY_RUN -eq 1 ]]; then
  echo "CLOSE-OUT REFUSED: --close-out requires --execute (dry-run mutates nothing)" >&2; exit 2
fi
echo "==> close-out: resolving ratified sync tip via origin/main"
git fetch origin main --quiet
MAIN_TIP="$(git rev-parse origin/main)"
WORKTREE_HEAD="$(git -C "$MAIN_WORKTREE" rev-parse HEAD)"
if [[ "$MAIN_TIP" != "$WORKTREE_HEAD" ]]; then
  echo "CLOSE-OUT REFUSED: origin/main ($MAIN_TIP) != main-worktree HEAD ($WORKTREE_HEAD) —" >&2
  echo "  commit + push the staged sync in $MAIN_WORKTREE first, then re-run with --close-out" >&2
  exit 1
fi
echo "==> close-out: bumping PARITY_MAIN_ANCHOR to $MAIN_TIP (taxi only)"
sed -i -E "s/^(PARITY_MAIN_ANCHOR=)[0-9a-f]{40}(.*)$/\1${MAIN_TIP}\2/" scripts/check_branch_parity.sh
grep -q "^PARITY_MAIN_ANCHOR=${MAIN_TIP}" scripts/check_branch_parity.sh || {
  echo "CLOSE-OUT FAILED: anchor line did not update to $MAIN_TIP" >&2; exit 1
}
echo "==> close-out: verifying parity"
bash scripts/check_branch_parity.sh | tee /tmp/promote_closeout_parity.log
grep -q "PARITY OK" /tmp/promote_closeout_parity.log || {
  echo "CLOSE-OUT FAILED: parity gate did not print PARITY OK" >&2; exit 1
}
if ! command -v gh >/dev/null 2>&1; then
  echo "==> close-out: 'gh' not found — re-run main CI manually:"
  echo "    TIP=\$(git rev-parse origin/main)"
  echo "    RUN_ID=\$(gh api repos/fbarulli/broadway/actions/runs -f head_sha=\"\$TIP\" -f branch=main --jq '.workflow_runs[0].id')"
  echo "    gh run rerun \"\$RUN_ID\" --failed"
  echo "    gh run watch \"\$RUN_ID\" --exit-status"
  exit 0
fi
echo "==> close-out: locating failed main workflow run for tip $MAIN_TIP"
RUNS_JSON="$(gh api repos/fbarulli/broadway/actions/runs -f head_sha="$MAIN_TIP" -f branch=main -f per_page=10)"
RUN_ID="$(printf '%s' "$RUNS_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); rs=d.get("workflow_runs",[]); print(rs[0]["id"] if rs else "")')"
if [[ -z "$RUN_ID" ]]; then
  echo "CLOSE-OUT: no workflow run found for $MAIN_TIP — trigger/poll manually:" >&2
  echo "    TIP=\$(git rev-parse origin/main)"
  echo "    RUN_ID=\$(gh api repos/fbarulli/broadway/actions/runs -f head_sha=\"\$TIP\" -f branch=main --jq '.workflow_runs[0].id')"
  echo "    gh run rerun \"\$RUN_ID\" --failed"
  echo "    gh run watch \"\$RUN_ID\" --exit-status"
  exit 1
fi
echo "==> close-out: re-running failed jobs for run $RUN_ID"
gh run rerun "$RUN_ID" --failed
echo "==> close-out: polling run $RUN_ID (max ~10 min)"
ATTEMPT=0
MAX_ATTEMPTS=30
SLEEP_SECS=20
while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
  ATTEMPT=$((ATTEMPT + 1))
  POLL_JSON="$(gh api repos/fbarulli/broadway/actions/runs -f head_sha="$MAIN_TIP" -f branch=main -f per_page=10)"
  RUN_STATE="$(RUN_ID_ENV="$RUN_ID" printf '%s' "$POLL_JSON" | python3 -c 'import json,os,sys; rid=os.environ.get("RUN_ID_ENV",""); d=json.load(sys.stdin); m={str(r.get("id")): r for r in d.get("workflow_runs",[])}; r=m.get(str(rid),{}); print((r.get("status") or "")+"|"+(r.get("conclusion") or ""))')"
  RUN_STATUS="${RUN_STATE%%|*}"
  RUN_CONCLUSION="${RUN_STATE##*|}"
  echo "    [${ATTEMPT}/${MAX_ATTEMPTS}] status=${RUN_STATUS:-unknown} conclusion=${RUN_CONCLUSION:-pending}"
  if [[ "$RUN_STATUS" == "completed" ]]; then
    if [[ "$RUN_CONCLUSION" == "success" ]]; then
      echo "CLOSE-OUT COMPLETE: main CI success for $MAIN_TIP (run $RUN_ID)"
      exit 0
    fi
    echo "CLOSE-OUT FAILED: main CI concluded '${RUN_CONCLUSION}' for $MAIN_TIP (run $RUN_ID)" >&2
    exit 1
  fi
  sleep "$SLEEP_SECS"
done
echo "CLOSE-OUT TIMEOUT: run $RUN_ID not completed after ~10 min — poll manually:" >&2
echo "    gh api repos/fbarulli/broadway/actions/runs -f head_sha=\"${MAIN_TIP}\" -f branch=main"
echo "    gh run watch \"$RUN_ID\" --exit-status" >&2
exit 1
