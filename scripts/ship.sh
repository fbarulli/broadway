#!/usr/bin/env bash
# ship.sh — THE push path. Full tier gates; exit codes decide; no prose parsing.
# Born from two push-on-red recurrences (2026-08-24): keyword-grep gating and
# unconditional newline-chained pushes. This wrapper makes both impossible:
# the push lines execute ONLY if run_local_ci.sh exits 0.
#
# Usage: bash scripts/ship.sh [remote] [refspec...]   (default: origin +refs/heads/sklearn)
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}" MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"

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

for spec in "${REFSPECS[@]}"; do
  echo "== pushing $REMOTE $spec =="
  git push "$REMOTE" "$spec"
done
echo "SHIP OK"
