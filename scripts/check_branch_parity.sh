#!/usr/bin/env bash
# Branch parity gate — keep main and taxi's SHARED surface in lockstep.
#
# The repo has a deliberate split:
#   * sklearn — the only active development line (agents/contracts/MAIN_AGENT_CONTRACT.md §2);
#             taxi fast-forwards to it after each green push; main is frozen
#             until declared main-day.
#
# src/, tests/, demo/, scripts/, experiments/more_modeling/, the synthetic-demo
# configs, and the deployment files (k8s/, docker/, .github/workflows/) are
# meant to be IDENTICAL on both
# branches. This script fails loudly the moment they drift — including
# deletions and content changes — so a change made on one branch cannot
# silently diverge.
#
# ERA-AWARE (D16): behaviour is gated by the era declaration, INLINED below
# (D21 relocated it from .github/parity-era.env — see "Era declaration").
# PARITY_ERA=dev means
# sklearn is the active line and main is frozen (every event runs frozen-main
# custody, then branch-aware pass-along guards); PARITY_ERA=main is lockstep
# day (stock check / --sync). There is no environment-variable dialect.
#
# Usage:
#   scripts/check_branch_parity.sh          # era-aware parity check
#   scripts/check_branch_parity.sh --sync   # main-day only: copy the track
#                                           # branch's shared surface onto main
#
# Exit codes: 0 = in sync, 1 = drift / custody violation, 2 = usage error.

set -euo pipefail

MODE="check"
if [[ "${1:-}" == "--sync" ]]; then
  MODE="sync"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--sync]" >&2
  exit 2
fi

# The intended shared surface — paths that must be byte-identical on both
# branches. Keep this list explicit: anything NOT listed is deliberately
# taxi-only or main-only.
SHARED=(
  src/
  tests/
  demo/
  configs/dataset/test.yaml
  configs/experiment/baseline.yaml
  configs/experiment/engineered.yaml
  configs/experiment/hyperopt.yaml
  configs/analysis/test.yaml
  configs/analysis/test_hypothesis.yaml
  configs/analysis/test_causal.yaml
  configs/step/causal.yaml
  configs/step/etl.yaml
  configs/environment/
  configs/flow/
  k8s/
  docker/
  .github/workflows/
  pyproject.toml
  Dockerfile
  docker-compose.yml
  .gitignore
  .dockerignore
  README.md
  scripts/
  experiments/more_modeling/
)

check() {
  local drifted=0
  local path
  for path in "${SHARED[@]}"; do
    # Byte-identical between the two branch tips (deletions included: if one
    # side lacks the path, git diff reports it and we fail).
    if ! git diff --exit-code --quiet "origin/main" "origin/taxi" -- "$path" 2>/dev/null; then
      echo "DRIFT: $path differs between origin/main and origin/taxi" >&2
      drifted=1
    fi
  done
  if [[ $drifted -ne 0 ]]; then
    echo "PARITY FAILED — shared surface drifted. Run: $0 --sync" >&2
    return 1
  fi
  echo "PARITY OK — shared surface is identical on origin/main and origin/taxi"
}

sync_to_main() {
  local path
  # Work from a clean main checkout against the latest taxi.
  git fetch origin
  git checkout main
  git checkout taxi -- "${SHARED[@]}"
  # Deletions do not propagate with checkout — mirror them too.
  local f
  while IFS= read -r f; do
    if ! git cat-file -e "origin/taxi:$f" 2>/dev/null; then
      git rm -f --ignore-unmatch "$f" >/dev/null 2>&1 || true
    fi
  done < <(git diff --name-only --diff-filter=AD "origin/main" "origin/taxi" -- "${SHARED[@]}" || true)
  echo "SYNCED taxi -> main for shared surface. Review, run gates, commit, push."
}

# --- Era declaration INLINE (D21: no separate env file, zero SHARED lines) ---
# D21 relocated the former .github/parity-era.env verbatim into this script:
# exactly one era vocabulary (D16a), and editing these four lines below IS
# the main-day flip act (D16c). NOTE for scripts/run_local_ci.sh's F1b guard:
# the `^PARITY_ERA=` line is the staleness marker — a track ref whose checker
# lacks it predates D16/D21 and must not gate CI.
PARITY_ERA=dev                 # dev: sklearn active, main frozen | main: lockstep day
PARITY_TRACK_BRANCH=sklearn    # active development line during dev era
PARITY_ALLOWLIST=()            # SHARED paths exempt from custody; extend only by cited ruling
PARITY_MAIN_ANCHOR=18607091ddbb2602ad4475341ad377bafee5ec4b  # last ratified state of frozen main

# Preserved validations (D16 rider). The old ENV_FILE readability test,
# `source`, ${VAR:?} trio, and declare -p existence check are DEAD here —
# constants cannot be unset or missing; only shape/resolution remain.
# Anchor shape + resolution: a garbled pin must fail as CONFIG ERROR here,
# not later as a misleading ROGUE MAIN WRITE from the diff guard.
[[ "$PARITY_MAIN_ANCHOR" =~ ^[0-9a-f]{40}$ ]] || {
  echo "FATAL: PARITY_MAIN_ANCHOR must be a full 40-hex sha (got '$PARITY_MAIN_ANCHOR')" >&2
  exit 1
}
git cat-file -e "$PARITY_MAIN_ANCHOR^{commit}" 2>/dev/null || {
  echo "FATAL: PARITY_MAIN_ANCHOR $PARITY_MAIN_ANCHOR does not resolve to a commit (stale pin? re-anchor at the next ratified main-day flip-back)" >&2
  exit 1
}
case "$PARITY_ERA" in
  dev|main) ;;
  *) echo "FATAL: unknown PARITY_ERA '$PARITY_ERA'" >&2; exit 1 ;;
esac

custody() {
  # Frozen-main custody (D16b/F2-revised) — two independent alarms, tens of
  # ms total.
  #
  # (1) Anchor drift guard: diff main against PARITY_MAIN_ANCHOR — the last
  #     ratified state of frozen main (seeded at the frozen tip; updated ONLY
  #     in the same commit as a ratified main-day sync/flip-back). Catches
  #     adds/deletes/mods/smuggling; self-diff at seed is zero by
  #     construction, and a stale pin can only false-red loudly AFTER an
  #     unanchored ratified change — the safe failure direction.
  #     (Merge-base anchoring rejected: 21/24 SHARED entries legitimately
  #     diverge between merge-base 7758d1a and the sanctioned main tip.)
  if ! git diff --exit-code --quiet "$PARITY_MAIN_ANCHOR" origin/main -- "${SHARED[@]}"; then
    echo "ROGUE MAIN WRITE: frozen main changed since anchor $PARITY_MAIN_ANCHOR (adds/deletes/mods)" >&2
    exit 1
  fi
  #
  # (2) Blob provenance: every blob on main's tip under SHARED must already
  #     exist somewhere in taxi's object universe (comm -23 = in main, not in
  #     universe). Allowlisted paths are skipped. Secondary layer — catches
  #     novel-content adds; the anchor diff above catches deletions/mods.
  # D21 rider: main and track lines are DISJOINT histories (no common
  # ancestor since the dev-era reset), so the universe below can never
  # contain main's blob versions — the check would false-positive on all
  # shared files forever. Its value is catching POST-FREEZE writes to
  # main; when main sits exactly at the anchor, the freeze-intact
  # shortcut above makes this layer unreachable-by-definition. Run it
  # ONLY when main moved off the anchor (then any novel blob IS rogue).
  if [[ "$(git rev-parse --verify origin/main)" == "$PARITY_MAIN_ANCHOR" ]]; then
    return 0
  fi
  local novel
  novel=$(comm -23 \
    <(git ls-tree -r origin/main -- "${SHARED[@]}" \
        | awk -F'\t' -v allow="${PARITY_ALLOWLIST[*]:-}" '
            {
              split($1, meta, " ")
              path = $2
              skipped = 0
              n = split(allow, A, " ")
              for (i = 1; i <= n; i++)
                if (A[i] != "" && index(path, A[i]) == 1) { skipped = 1; break }
              if (!skipped) print meta[3]
            }' | sort -u) \
    <( git rev-list --objects "origin/$PARITY_TRACK_BRANCH" | cut -d' ' -f1 | sort -u))
  if [[ -n "$novel" ]]; then
    echo "ROGUE MAIN WRITE: novel blob(s) on frozen main absent from the origin/$PARITY_TRACK_BRANCH universe:" >&2
    printf '%s\n' "$novel" | head -10 >&2
    exit 1
  fi
}

# --- Dispatch -----------------------------------------------------------------
branch="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")}"
if [[ -z "$branch" || "$branch" == "HEAD" ]]; then
  branch=""   # detached HEAD / unnamed ref — handled explicitly (F4)
fi

case "$PARITY_ERA" in
  main)
    # Lockstep day: stock behaviour, verbatim.
    if [[ "$MODE" == "sync" ]]; then
      sync_to_main
    else
      check
    fi
    ;;
  dev)
    if [[ "$MODE" == "sync" ]]; then
      echo "REFUSED: --sync is a main-day act (era=$PARITY_ERA declared inline per D21) — nothing was modified" >&2
      exit 1
    fi
    custody   # every event, every branch: frozen-main custody first
    # Branch-aware pass-along (F4: GITHUB_REF_NAME-first).
    if [[ -z "$branch" || "$branch" == "main" ]]; then
      :  # dead case, explicit: empty / main / PR-merge refs — custody-only
    elif [[ "$branch" == "taxi" ]]; then
      # Pushing TO taxi requires the fast-forward to already be complete.
      if ! git diff --exit-code --quiet "origin/$PARITY_TRACK_BRANCH" origin/taxi -- "${SHARED[@]}"; then
        echo "TAXI DRIFT: origin/taxi is not byte-identical to origin/$PARITY_TRACK_BRANCH on the shared surface — complete the fast-forward before pushing to taxi" >&2
        exit 1
      fi
    elif [[ "$branch" == "$PARITY_TRACK_BRANCH" ]]; then
      # Pushing TO the track branch: taxi may lag, never fork.
      if ! git merge-base --is-ancestor origin/taxi "origin/$branch"; then
        echo "FORK: origin/taxi is not an ancestor of origin/$branch — taxi may lag, never fork" >&2
        exit 1
      fi
    else
      :  # any other named ref — custody-only
    fi
    ;;
esac

echo "PARITY OK (era=$PARITY_ERA branch=$branch)"

# KNOWN RESIDUAL (D16, deliberately NOT fixed here): pushes TO main execute
# main's own checked-out legacy script until main-day delivers this file via
# --sync; a push to frozen main is itself the violation, so legacy red there
# is a correct alarm — noted in D16.
