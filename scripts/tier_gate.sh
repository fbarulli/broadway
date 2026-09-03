#!/usr/bin/env bash
# tier_gate.sh — D34/D35 "teeth ①" TIER-GATE validator. Sourced by
# scripts/ship.sh; deliberately a plain-sourced library so a future
# prepare-commit-msg hook reuses the exact same grammar.
#
# Ratified grammar (DECISIONS.md D35(2)): every gated commit must carry a
# computed 'Tier:' trailer whose value is exactly one of FULL|FAST|STATIC|DOCS
# (case-sensitive). Tier FULL additionally requires a resolvable
# reviewer-verdict id, which RESOLVES IFF (a) it is a row id present in the
# STATE.md ## EVENTS table AT THAT COMMIT, and (b) that same event-id string
# appears in the commit message carrying the Tier: trailer.
#
# Public surface:
#   tg_extract_trailer KEY          (message on stdin) -> last 'KEY:' value
#   tg_events_has EVENTS_TEXT TOKEN                   -> rc0 iff row id found
#   tg_check_message EVENTS_TEXT MESSAGE              -> rc0 pass / rc1 + reason
#   tg_run                          (SHAs on stdin)   -> gates each via git
#   tg_batch_adds_row BASE SHA                        -> rc0 iff the batch's
#       ledger diff adds >=1 STATE row id (new CURRENT row; archive lines and
#       in-place edits do not count)
#   tg_batch_terminal BASE SHA                       -> rc0 iff the batch's
#       ledger diff terminally dispositions >=1 STATE row (a new
#       STATE-ARCHIVE marker, or a CURRENT row's status becoming closed/void)
#   tg_ledger_batch BASE SHA                        -> batch-level ledger
#       custody law: rc0 iff BOTH helpers pass (item added AND one closed)
# shellcheck shell=bash

_TG_TIERS='FULL|FAST|STATIC|DOCS'
_TG_EVENTS_HEADER='## EVENTS'
_TG_EVENTS_FILE='agents/ledger/STATE.md'

# Value of the LAST 'KEY:' line (trailer blocks close a message); '' if absent.
tg_extract_trailer() {
  sed -n "s/^$1:[[:space:]]*//p" | tail -n 1
}

# rc0 iff TOKEN opens a table row ('| TOKEN |') inside EVENTS_TEXT.
tg_events_has() {
  printf '%s\n' "$1" | grep -Eq "^\|[[:space:]]*$2[[:space:]]*\|"
}

# Resolution-via-authority-row (D35(2) "resolves iff row ... cited"): the
# verdict token resolves iff it IS a row id OR it occurs inside the EVENTS
# section body — i.e. an authority-class registration row whose stated scope
# is "valid Reviewer:-trailer resolution target". Unregistered ids still
# fail closed; registry-context occurrence is legal hex discipline.
tg_events_resolves() {
  tg_events_has "$@" && return 0
  printf '%s\n' "$1" | grep -Eq "(^|[^0-9a-fA-F])$2([^0-9a-fA-F]|$)"
}

# FULL-tier reviewer-verdict resolution; echoes a refusal reason, '' iff resolved.
_tg_reviewer_reason() {
  local reviewer token
  reviewer="$(printf '%s\n' "$2" | tg_extract_trailer Reviewer)"
  if [ -z "$reviewer" ] || [ "$reviewer" = "none" ]; then
    echo "Tier FULL requires a 'Reviewer:' verdict id (found '${reviewer:-<none>}')"
    return 0
  fi
  token="$(printf '%s\n' "$reviewer" | awk '{print $1}')"
  case "$token" in
    [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]) ;;
    *)
      echo "Reviewer verdict id '$token' is not an 8-hex event-id"
      return 0
      ;;
  esac
  token="$(printf '%s' "$token" | tr 'A-F' 'a-f')"
  if ! tg_events_resolves "$1" "$token"; then
    echo "reviewer verdict $token does NOT resolve in $_TG_EVENTS_FILE $_TG_EVENTS_HEADER at that commit (no row id, no authority-row registration)"
    return 0
  fi
  case "$2" in
    *"$token"*) return 0 ;;
    *)
      echo "event-id $token (row-resolved) absent from the commit message carrying 'Tier:'"
      ;;
  esac
}

# Pure grammar check: rc0 pass, rc1 refuse (single-line reason on stdout).
tg_check_message() {
  local tier reason
  tier="$(printf '%s\n' "$2" | tg_extract_trailer Tier)"
  case "$tier" in
    FULL|FAST|STATIC|DOCS) ;;
    '')
      echo "missing required 'Tier:' trailer (vocabulary: $_TG_TIERS)"
      return 1
      ;;
    *)
      echo "bad tier word '$tier' (vocabulary: $_TG_TIERS; case-sensitive)"
      return 1
      ;;
  esac
  if [ "$tier" != FULL ]; then
    return 0
  fi
  reason="$(_tg_reviewer_reason "$1" "$2")"
  if [ -n "$reason" ]; then
    echo "$reason"
    return 1
  fi
  return 0
}

# EVENTS-table text of STATE.md AS OF a commit (its own tree, not the worktree).
_tg_events_at() {
  git show "$1:$_TG_EVENTS_FILE" \
    | sed -n "/^$_TG_EVENTS_HEADER/,/^## /p" | sed '1d;/^## /,$d'
}

# --- Batch-level ledger custody (2026-09-03 owner law: every pushed batch
# must update the ledger with an item AND terminally disposition one) -------
# Diff-based, never worktree-content-based: the batch is judged by what its
# commits change BETWEEN the remote baseline and the pushed SHA. Pure diffs
# over git plumbing; STATE ids are the STATE-YYYYMMDD-NNN grammar owned by
# agents/tools/state_records.py.

# rc0 iff the batch adds at least one STATE row id to CURRENT (STATE.md).
tg_batch_adds_row() {
  local base="$1" sha="$2"
  git diff "$base" "$sha" -- agents/ledger/STATE.md     | grep -E '^\+\| STATE-[0-9]{8}-[0-9]{3} \|' | grep -qv '^+\+\+'
  # '^+| STATE-... |' = a newly added CURRENT table row (an added +++ file
  # header can never match, but grep -qv keeps the guard explicit).
}

# rc0 iff the batch terminally dispositions >=1 STATE row: either a new
# STATE-ARCHIVE marker in the archive, or a CURRENT row's status column
# flipping to closed/void.
tg_batch_terminal() {
  local base="$1" sha="$2"
  # (a) new archive marker for any month file
  if git diff "$base" "$sha" -- agents/ledger/archive/        | grep -Eq '^\+.*STATE-ARCHIVE:STATE-[0-9]{8}-[0-9]{3}'; then
    return 0
  fi
  # (b) a CURRENT row's status cell edited to closed or void
  git diff "$base" "$sha" -- agents/ledger/STATE.md     | grep -E '^\+\| STATE-[0-9]{8}-[0-9]{3} \| (closed|void) \|' | grep -q .
}

# Batch ledger-custody law: an added item AND a terminal disposition. Echoes
# the missing piece on refusal so ship.sh reports it without duplicating law.
tg_ledger_batch() {
  local base="$1" sha="$2"
  if ! tg_batch_adds_row "$base" "$sha"; then
    echo "ledger: batch adds no STATE row (add one via: state_records.py record add)"
    return 1
  fi
  if ! tg_batch_terminal "$base" "$sha"; then
    echo "ledger: batch terminally dispositions no STATE row (close or void one via: state_records.py record close/void)"
    return 1
  fi
  return 0
}

# Gate every SHA read on stdin against ITS OWN tree's EVENTS table; all-or-nothing.
tg_run() {
  local sha msg tier reason count=0 full=0 light=0
  while IFS= read -r sha; do
    [ -n "$sha" ] || continue
    count=$((count + 1))
    msg="$(git log -1 --format=%B "$sha")"
    reason="$(tg_check_message "$(_tg_events_at "$sha")" "$msg")"
    if [ -n "$reason" ]; then
      echo "TIER-GATE REFUSED $(git rev-parse --short "$sha"): $reason" >&2
      return 1
    fi
    tier="$(printf '%s\n' "$msg" | tg_extract_trailer Tier)"
    case "$tier" in
      FULL) full=$((full + 1)) ;;
      *) light=$((light + 1)) ;;
    esac
  done
  echo "TIER-GATE PASS: $count commit(s) gated (FULL=$full, lighter tiers=$light)"
  return 0
}
