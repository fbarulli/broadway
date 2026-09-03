#!/usr/bin/env bash
# ship.sh — THE push path.
#
# Contract:
#   1. Local CI must exit 0.
#   2. The configured pre-push guard must be installed.
#   3. The exact unpushed batch must pass the tier gate.
#   4. The exact unpushed batch must show ledger activity: >=1 STATE row
#      added AND >=1 STATE row terminally dispositioned (closed/void).
#   5. Exactly one non-deletion branch refspec is pushed.
#   6. The exact pushed commit is monitored in GitHub Actions when possible.
#
# Usage:
#   bash scripts/ship.sh [remote] [refspec]
#
# Default:
#   remote  = origin
#   refspec = HEAD:<current branch>
#
# Important:
#   A successful git push followed by a red/unknown Actions result is reported
#   as a POST-PUSH failure. The remote has already accepted the push.

set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "SHIP REFUSED: not inside a Git repository." >&2
  exit 1
}
cd "$REPO_ROOT"

on_err() {
  local rc=$?
  echo "SHIP FAILED: unexpected command failure (exit $rc) at line ${BASH_LINENO[0]:-unknown}." >&2
  exit "$rc"
}
trap on_err ERR

# Single sanctioned uv cache root:
#   $HOME/.cache/uv
#
# Do not export UV_CACHE_DIR here; repository contract prohibits it.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

# shellcheck source=scripts/tier_gate.sh
. scripts/tier_gate.sh

die() {
  echo "SHIP REFUSED: $*" >&2
  exit 1
}

REMOTE="${1:-origin}"
if (($# > 0)); then
  shift
fi

# Exactly one refspec. Multiple pushes must be separate invocations so every
# invocation gets its own complete gate + monitoring lifecycle.
if (($# > 1)); then
  die "multiple refspecs provided ($#); exactly one is allowed per invocation."
fi

if (($# == 0)); then
  current_branch="$(git symbolic-ref --quiet --short HEAD)" || {
    die "detached HEAD and no refspec was supplied."
  }
  REFSPEC="HEAD:refs/heads/$current_branch"
else
  REFSPEC="$1"
fi

# ---------------------------------------------------------------------------
# REFSPEC VALIDATION
# ---------------------------------------------------------------------------

# Supported forms:
#   source:refs/heads/branch
#   source:branch
#   HEAD:branch
#
# Deletions, tags, and arbitrary refs are deliberately rejected.
normalized_refspec="${REFSPEC#+}"

[[ "$normalized_refspec" != :* ]] ||
  die "deletion refspec '$REFSPEC' is forbidden."

if [[ "$normalized_refspec" != *:* ]]; then
  source_ref="$normalized_refspec"
  destination_ref="$normalized_refspec"
else
  source_ref="${normalized_refspec%%:*}"
  destination_ref="${normalized_refspec#*:}"
fi

[[ -n "$source_ref" ]] ||
  die "empty source in refspec '$REFSPEC'."

[[ -n "$destination_ref" ]] ||
  die "empty destination in refspec '$REFSPEC'."

destination_ref="${destination_ref#refs/heads/}"

git check-ref-format --branch "$destination_ref" >/dev/null 2>&1 || {
  die "destination '$destination_ref' is not a valid branch name."
}

[[ "$destination_ref" != refs/* ]] ||
  die "destination '$destination_ref' is not an ordinary branch."

branch="$destination_ref"

# Resolve the exact object that will be pushed BEFORE git push.
PUSH_SHA="$(git rev-parse --verify "${source_ref}^{commit}" 2>/dev/null)" || {
  die "source '$source_ref' does not resolve to a commit."
}

# ---------------------------------------------------------------------------
# REMOTE VALIDATION
# ---------------------------------------------------------------------------

git remote get-url "$REMOTE" >/dev/null 2>&1 || {
  die "remote '$REMOTE' does not exist."
}

# ---------------------------------------------------------------------------
# FULL LOCAL CI
# ---------------------------------------------------------------------------

echo "== ship.sh: full tier gates every push =="

if ! bash scripts/run_local_ci.sh; then
  echo >&2
  echo "SHIP REFUSED: LOCAL-CI RED. No push was attempted." >&2
  echo "Fix the failures above; there is no override flag." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# L1 PRE-PUSH GUARD
# ---------------------------------------------------------------------------

echo "== ship.sh: L1 pre-push hook guard =="

# Respect core.hooksPath instead of assuming .git/hooks.
HOOK_PATH="$(git rev-parse --git-path hooks/pre-push)" || {
  die "could not resolve Git's pre-push hook path."
}

if [[ ! -x "$HOOK_PATH" ]]; then
  echo "SHIP REFUSED: pre-push hook missing or not executable: $HOOK_PATH" >&2
  echo "Remediation — install the tracked template:" >&2
  echo "  cp agents/contracts/hooks-pre-push.template \"$HOOK_PATH\"" >&2
  echo "  chmod +x \"$HOOK_PATH\"" >&2
  exit 1
fi

if ! grep -Fq 'scripts/run_local_ci.sh' "$HOOK_PATH"; then
  echo "SHIP REFUSED: pre-push hook does not invoke scripts/run_local_ci.sh." >&2
  echo "Hook: $HOOK_PATH" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# TIER GATE ON EXACT BATCH
# ---------------------------------------------------------------------------

echo "== ship.sh: TIER-GATE on unpushed batch =="

REMOTE_BRANCH="$REMOTE/$branch"

if git rev-parse --verify -q "$REMOTE_BRANCH" >/dev/null 2>&1; then
  commits="$(git rev-list "$REMOTE_BRANCH..$PUSH_SHA")" || {
    die "failed to enumerate '$REMOTE_BRANCH..$PUSH_SHA'."
  }

  if [[ -z "$commits" ]]; then
    echo "== ship.sh: no commits ahead of $REMOTE_BRANCH; TIER-GATE skipped =="
  else
    if ! printf '%s\n' "$commits" | tg_run; then
      echo "SHIP REFUSED: TIER-GATE rejected the batch." >&2
      echo "No push was attempted." >&2
      exit 1
    fi
  fi

  # -------------------------------------------------------------------------
  # LEDGER BATCH LAW: the batch must add >=1 STATE row AND terminally
  # disposition >=1 (close/void). The law and its vocabulary live in
  # scripts/tier_gate.sh (tg_ledger_batch) so the pre-push hook and this
  # script share one grammar — one law, two doors.
  # -------------------------------------------------------------------------
  echo "== ship.sh: LEDGER-GATE on unpushed batch =="

  if [[ -n "$commits" ]]; then
    if ! ledger_reason="$(tg_ledger_batch "$REMOTE_BRANCH" "$PUSH_SHA")"; then
      echo "SHIP REFUSED: LEDGER-GATE rejected the batch." >&2
      echo "  $ledger_reason" >&2
      echo "No push was attempted." >&2
      exit 1
    fi
  fi
else
  echo "SHIP NOTICE: $REMOTE_BRANCH does not exist locally." >&2
  echo "TIER-GATE cannot compare against a remote baseline; continuing because" >&2
  echo "this appears to be the first push of this remote branch." >&2
fi

# ---------------------------------------------------------------------------
# PUSH
# ---------------------------------------------------------------------------

echo "== pushing $REMOTE $REFSPEC =="
echo "== exact pushed commit: $PUSH_SHA =="

git push "$REMOTE" "$REFSPEC"

# ---------------------------------------------------------------------------
# POST-PUSH ACTIONS MONITOR
# ---------------------------------------------------------------------------

# Best effort by policy, but never silently swallow CLI/API failures.
if ! command -v gh >/dev/null 2>&1; then
  echo "SHIP MONITOR SKIPPED: gh CLI absent — post-push Actions watch off" >&2
else
  remote_url="$(git remote get-url "$REMOTE" 2>/dev/null)" || {
    echo "SHIP MONITOR SKIPPED: cannot read URL for remote '$REMOTE'" >&2
    echo "Push succeeded; CI status is UNKNOWN." >&2
    exit 0
  }

  # Handle:
  #   https://github.com/owner/repo.git
  #   http://github.com/owner/repo.git
  #   git@github.com:owner/repo.git
  #   ssh://git@github.com:owner/repo.git
  slug="$(
    printf '%s\n' "$remote_url" |
      sed -E \
        -e 's#^[^@]+@[^:]+:##' \
        -e 's#^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]+/##' \
        -e 's#[.]git$##'
  )"

  if [[ ! "$slug" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    echo "SHIP MONITOR SKIPPED: cannot derive owner/repo from remote URL." >&2
    echo "Remote URL: $remote_url" >&2
    echo "Push succeeded; CI status is UNKNOWN." >&2
    exit 0
  fi

  echo "== ship monitor: Actions for $slug@$PUSH_SHA (branch $branch) =="

  appeared=0
  polls=0
  seen=""
  monitor_failed=0

  while ((polls < 20)); do
    polls=$((polls + 1))

    api_err="$(mktemp)"
    api_out=""

    # Deliberately put the command in an if-condition so set -e does not
    # terminate the script before we can report the API failure.
    if api_out="$(
      gh api "repos/$slug/actions/runs?branch=$branch&per_page=10" \
        --jq "[.workflow_runs[] | select(.head_sha == \"$PUSH_SHA\")][0] |
              if . == null then
                \"\"
              else
                .status + \" \" + (.conclusion // \"none\")
              end" \
        2>"$api_err"
    )"; then
      api_exit=0
    else
      api_exit=$?
    fi

    api_err_content="$(cat "$api_err")"
    rm -f "$api_err"

    if ((api_exit != 0)); then
      echo "SHIP MONITOR ERROR: 'gh api' failed (exit $api_exit)." >&2
      if [[ -n "$api_err_content" ]]; then
        echo "$api_err_content" >&2
      fi
      echo "Push STANDS, but CI status is UNKNOWN." >&2
      echo "Check the Actions page for $slug manually." >&2
      monitor_failed=1
      break
    fi

    seen="$api_out"

    if [[ -n "$seen" ]]; then
      appeared=1
    fi

    case "$seen" in
      "")
        # Run has not been created yet.
        ;;

      "completed success")
        echo "SHIP MONITOR OK: $slug@$PUSH_SHA green on $branch"
        break
        ;;

      completed\ *)
        echo "SHIP MONITOR RED: $slug@$PUSH_SHA reported '$seen'" >&2
        echo "Push landed red — fix forward before the next ship." >&2
        exit 1
        ;;

      *)
        # queued / in_progress / waiting / etc.
        ;;
    esac

    if ((polls < 20)); then
      sleep 45
    fi
  done

  if ((monitor_failed == 0)); then
    if ((appeared == 0)); then
      echo "WARNING: monitor exhausted 20 polls (~15 minutes)." >&2
      echo "No Actions run for exact SHA $PUSH_SHA appeared." >&2
      echo "Push STANDS; check $slug Actions manually." >&2
    elif [[ "$seen" != "completed success" ]]; then
      if [[ "$seen" == completed\ * ]]; then
        echo "SHIP MONITOR RED: $slug@$PUSH_SHA reported '$seen'" >&2
        exit 1
      else
        echo "WARNING: monitor exhausted 20 polls (~15 minutes)." >&2
        echo "Actions run is still '$seen'." >&2
        echo "Push STANDS; check $slug Actions manually." >&2
      fi
    fi
  fi
fi

echo "SHIP OK"
