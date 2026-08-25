#!/usr/bin/env bash
# ship.sh — THE push path. Full tier gates; exit codes decide; no prose parsing.
# Born from two push-on-red recurrences (2026-08-24): keyword-grep gating and
# unconditional newline-chained pushes. This wrapper makes both impossible:
# the push lines execute ONLY if run_local_ci.sh exits 0.
#
# Usage: bash scripts/ship.sh [remote] [refspec...]   (default: origin +refs/heads/sklearn)
set -euo pipefail
cd "$(dirname "$0")/.."
# Single sanctioned uv cache root: $HOME/.cache/uv — a repo-local UV_CACHE_DIR
# export is prohibited (MAIN_AGENT_CONTRACT "Ledger & artifact hygiene").
# MPLCONFIGDIR stays: matplotlib font-cache determinism for the plotting path.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

REMOTE="${1:-origin}"; shift || true
REFSPECS=("$@")
[ ${#REFSPECS[@]} -eq 0 ] && REFSPECS=("sklearn:sklearn")

echo "== ship.sh: full tier gates every push =="
if ! bash scripts/run_local_ci.sh; then
  echo "" >&2
  echo "SHIP REFUSED: LOCAL-CI RED. Fix above. No flag exists to override this;" >&2
  echo "raw 'git push' is a policy violation (MAIN_AGENT_CONTRACT §push-custody)." >&2
  exit 1
fi

echo "== pushing $REMOTE ${REFSPECS[*]} (single invocation -> one hook gate) =="
git push "$REMOTE" "${REFSPECS[@]}"
# --- POST-PUSH MONITOR (#5/D25): watch the pushed tip's Actions run -------
# Best-effort by law: absent gh, unparsable remote, or a silent API degrade
# to a notice + exit 0; only an OBSERVED red/cancelled run exits non-zero.
if ! command -v gh >/dev/null 2>&1; then
  echo "SHIP MONITOR SKIPPED: gh CLI absent — post-push Actions watch off"
elif ! slug="$(git remote get-url "$REMOTE" \
      | sed -E 's#^(https?://[^/]+/|git@[^:]+:)##; s#[.]git$##')" || [[ "$slug" != */* ]]; then
  echo "SHIP MONITOR SKIPPED: cannot derive owner/repo from '$REMOTE' URL"
else
  branch="${REFSPECS[0]##*:}"; branch="${branch#refs/heads/}"
  PUSHED_SHA="$(git rev-parse --short HEAD)"
  echo "== ship monitor: Actions for $slug@$PUSHED_SHA (branch $branch) =="
  appeared=0; polls=0
  while [ "$polls" -lt 20 ]; do
    polls=$((polls + 1))
    seen="$(gh api "repos/$slug/actions/runs?branch=$branch&per_page=3" \
      --jq "[.workflow_runs[] | select(.head_sha | startswith(\"$PUSHED_SHA\"))][0] | .status + \" \" + (.conclusion // \"none\")" 2>/dev/null || true)"
    if [ -n "$seen" ]; then appeared=1; fi
    case "$seen" in
      "") ;;                                # run not created yet — keep polling
      "completed success")
        echo "SHIP MONITOR OK: $slug@$PUSHED_SHA green on $branch"
        break ;;
      completed*)
        echo "SHIP MONITOR RED: $slug@$PUSHED_SHA reported '$seen'" >&2
        echo "Push landed red — D25 fix-forward duty: reconcile before next ship." >&2
        exit 1 ;;
      *) ;;                                 # queued/in_progress — keep polling
    esac
    sleep 45
  done
  if [ "$appeared" -eq 0 ]; then
    echo "WARNING: monitor exhausted 20x45s; NO Actions run for $PUSHED_SHA ever appeared (Actions slow/down?). Push STANDS — check github.com/$slug/actions manually." >&2
  elif [ "$seen" != "completed success" ]; then
    echo "WARNING: monitor exhausted 20x45s; run still '$seen' (incomplete). Push STANDS — check github.com/$slug/actions manually." >&2
  fi
fi
echo "SHIP OK"
