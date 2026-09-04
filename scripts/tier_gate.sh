#!/usr/bin/env bash
# tier_gate.sh — D34/D35 "teeth ①" TIER-GATE validator.
#
# Sourced by scripts/ship.sh. Deliberately a plain-sourced library so future
# hooks can reuse the exact same grammar.
#
# Ratified grammar (DECISIONS.md D35(2)):
#
#   Every gated commit must carry a computed `Tier:` trailer whose value is
#   exactly one of FULL|FAST|STATIC|DOCS (case-sensitive).
#
#   Tier FULL additionally requires a resolvable Reviewer verdict id.
#
#   A Reviewer id RESOLVES IFF:
#     (a) it is an 8-hex event id registered as a row id in STATE.md ## EVENTS
#         at that commit, OR
#     (b) it is explicitly registered by an authority row in the EVENTS body;
#   AND:
#     (c) that same event-id string appears in the commit message carrying
#         the Tier: trailer.
#
# Public surface:
#
#   tg_extract_trailer KEY
#       Message on stdin -> last `KEY:` value.
#
#   tg_events_has EVENTS_TEXT TOKEN
#       rc0 iff TOKEN is an EVENTS table row id.
#
#   tg_events_resolves EVENTS_TEXT TOKEN
#       rc0 iff TOKEN resolves as an authorized event id.
#
#   tg_check_message EVENTS_TEXT MESSAGE
#       rc0 pass / rc1 + single-line refusal reason.
#
#   tg_run
#       SHAs on stdin -> gates every commit against its own EVENTS table.
#
#   tg_batch_adds_row BASE SHA
#       rc0 iff the batch diff adds >=1 CURRENT STATE row.
#
#   tg_batch_terminal BASE SHA
#       rc0 iff the batch terminally dispositions >=1 STATE row.
#
#   tg_ledger_batch BASE SHA
#       rc0 iff BOTH batch custody requirements pass.
#
# shellcheck shell=bash

_TG_TIERS='FULL|FAST|STATIC|DOCS'
_TG_EVENTS_HEADER='## EVENTS'
_TG_EVENTS_FILE='agents/ledger/STATE.md'

# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

# Escape a literal string for use inside a basic/extended grep regex.
# This prevents caller-controlled tokens from becoming regex syntax.
_tg_regex_escape() {
  printf '%s' "$1" | sed 's/[][\\.^$*+?(){}|]/\\&/g'
}

# Return rc0 iff VALUE is exactly an 8-hex event id.
_tg_valid_event_id() {
  [[ "$1" =~ ^[0-9A-Fa-f]{8}$ ]]
}

# ---------------------------------------------------------------------------
# TRAILER GRAMMAR
# ---------------------------------------------------------------------------

# Value of the LAST `KEY:` line.
#
# Trailer keys are literal strings, not regexes.
# Empty output means absent or empty.
tg_extract_trailer() {
  local key
  key="$1"

  # Only permit ordinary trailer keys. This prevents accidental regex/sed
  # injection and keeps the grammar intentionally narrow.
  [[ "$key" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || return 1

  awk -v key="$key" '
    index($0, key ":") == 1 {
      value = substr($0, length(key) + 2)
      sub(/^[[:space:]]+/, "", value)
      last = value
    }
    END {
      if (last != "") print last
    }
  '
}

# ---------------------------------------------------------------------------
# EVENTS AUTHORITY
# ---------------------------------------------------------------------------

# rc0 iff TOKEN opens a table row:
#
#   | TOKEN | ...
#
# TOKEN is matched literally.
tg_events_has() {
  local events token escaped
  events="$1"
  token="$2"

  _tg_valid_event_id "$token" || return 1

  escaped="$(_tg_regex_escape "$token")"

  printf '%s\n' "$events" |
    grep -Eq "^\\|[[:space:]]*${escaped}[[:space:]]*\\|"
}

# rc0 iff TOKEN is explicitly registered as an authority-class resolution
# target in the EVENTS section.
#
# This intentionally does NOT accept arbitrary textual occurrence of TOKEN.
# Merely mentioning an id in prose must not make it authoritative.
tg_events_resolves() {
  local events token escaped
  events="$1"
  token="$2"

  _tg_valid_event_id "$token" || return 1

  # Ordinary EVENTS row id is authoritative.
  tg_events_has "$events" "$token" && return 0

  escaped="$(_tg_regex_escape "$token")"

  # Authority registration must occur in a table row. Keep this deliberately
  # narrow: arbitrary prose or another event's description cannot register an
  # id merely by mentioning it. (-q: this helper is rc-only; its caller
  # captures stdout as a refusal reason, so a match must NOT print.)
  printf '%s\n' "$events" |
    grep -Eiq \
      "^\|.*${escaped}.*valid[[:space:]]+Reviewer:-trailer[[:space:]]+resolution[[:space:]]+target.*\|"
}

# ---------------------------------------------------------------------------
# FULL-TIER REVIEWER RESOLUTION
# ---------------------------------------------------------------------------

# Echoes a refusal reason; emits nothing iff resolved.
_tg_reviewer_reason() {
  local events="$1"
  local message="$2"
  local reviewer token

  reviewer="$(printf '%s\n' "$message" | tg_extract_trailer Reviewer || true)"

  if [[ -z "$reviewer" || "$reviewer" == "none" ]]; then
    echo "Tier FULL requires a 'Reviewer:' verdict id (found '${reviewer:-<none>}')"
    return 0
  fi

  # The grammar accepts an id followed by optional annotation, but the first
  # token itself must be the exact event id.
  token="${reviewer%%[[:space:]]*}"

  if ! _tg_valid_event_id "$token"; then
    echo "Reviewer verdict id '$token' is not an 8-hex event-id"
    return 0
  fi

  token="${token,,}"

  if ! tg_events_resolves "$events" "$token"; then
    echo "reviewer verdict $token does NOT resolve in $_TG_EVENTS_FILE $_TG_EVENTS_HEADER at that commit"
    return 0
  fi

  # Require the exact event-id string in the commit message. Case-insensitive
  # matching is intentionally avoided: the canonical token is lowercase after
  # normalization, so the message must carry that canonical representation.
  if [[ "$message" == *"$token"* ]]; then
    return 0
  fi

  echo "event-id $token (row-resolved) absent from the commit message carrying 'Tier:'"
}

# ---------------------------------------------------------------------------
# PURE COMMIT MESSAGE CHECK
# ---------------------------------------------------------------------------

# rc0 pass, rc1 refuse.
# Exactly one single-line refusal reason is printed on stdout.
tg_check_message() {
  local events="$1"
  local message="$2"
  local tier
  local reason

  tier="$(printf '%s\n' "$message" | tg_extract_trailer Tier || true)"

  case "$tier" in
    FULL|FAST|STATIC|DOCS)
      ;;
    '')
      echo "missing required 'Tier:' trailer (vocabulary: $_TG_TIERS)"
      return 1
      ;;
    *)
      echo "bad tier word '$tier' (vocabulary: $_TG_TIERS; case-sensitive)"
      return 1
      ;;
  esac

  if [[ "$tier" != "FULL" ]]; then
    return 0
  fi

  reason="$(_tg_reviewer_reason "$events" "$message")"

  if [[ -n "$reason" ]]; then
    printf '%s\n' "$reason"
    return 1
  fi

  return 0
}

# ---------------------------------------------------------------------------
# EVENTS SNAPSHOT
# ---------------------------------------------------------------------------

# Emit the EVENTS table body from STATE.md AS OF COMMIT.
#
# This deliberately reads the committed tree, never the worktree.
_tg_events_at() {
  local sha="$1"
  local state

  state="$(git show "$sha:$_TG_EVENTS_FILE" 2>/dev/null)" || {
    echo "TIER-GATE INTERNAL ERROR: $_TG_EVENTS_FILE does not exist at $sha" >&2
    return 1
  }

  awk '
    /^## EVENTS[[:space:]]*$/ {
      in_events = 1
      next
    }

    in_events && /^## / {
      exit
    }

    in_events {
      print
    }
  ' <<<"$state"
}

# ---------------------------------------------------------------------------
# BATCH LEDGER CUSTODY
# ---------------------------------------------------------------------------
#
# Owner law:
#
#   Every pushed batch must:
#     1. add at least one CURRENT ledger item;
#     2. terminally disposition at least one ledger item.
#
# These checks are deliberately diff-based. Worktree contents are irrelevant.
#
# STATE ids are owned by agents/tools/state_records.py and use:
#
#   STATE-YYYYMMDD-NNN
#

# rc0 iff batch adds at least one STATE row id to CURRENT.
tg_batch_adds_row() {
  local base="$1"
  local sha="$2"

  git diff --no-ext-diff --unified=0 "$base" "$sha" -- "$_TG_EVENTS_FILE" |
    grep -Eq '^\+\|[[:space:]]*STATE-[0-9]{8}-[0-9]{3}[[:space:]]*\|'
}

# rc0 iff batch terminally dispositions >=1 STATE row:
#
#   (a) adds a STATE-ARCHIVE marker, OR
#   (b) changes a CURRENT row status to closed/void.
#
tg_batch_terminal() {
  local base="$1"
  local sha="$2"

  # New archive marker.
  if git diff --no-ext-diff --unified=0 "$base" "$sha" -- agents/ledger/archive/ |
    grep -Eq '^\+.*STATE-ARCHIVE:STATE-[0-9]{8}-[0-9]{3}'; then
    return 0
  fi

  # CURRENT row status becoming closed/void.
  if git diff --no-ext-diff --unified=0 "$base" "$sha" -- "$_TG_EVENTS_FILE" |
    grep -Eq '^\+\|[[:space:]]*STATE-[0-9]{8}-[0-9]{3}[[:space:]]*\|[[:space:]]*(closed|void)[[:space:]]*\|'; then
    return 0
  fi

  return 1
}

# Batch ledger custody.
#
# rc0 iff BOTH helpers pass.
# Refusal reason is emitted once, here, so callers don't duplicate the law.
tg_ledger_batch() {
  local base="$1"
  local sha="$2"

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

# ---------------------------------------------------------------------------
# COMMIT-LEVEL TIER GATE
# ---------------------------------------------------------------------------

# Gate every SHA read on stdin against ITS OWN tree's EVENTS table.
#
# All-or-nothing:
#   the first invalid commit immediately fails the entire invocation.
#
# Input contract:
#   one commit-ish per line.
tg_run() {
  local sha msg events reason tier
  local count=0
  local full=0
  local light=0
  local resolved_sha

  while IFS= read -r sha; do
    [[ -n "$sha" ]] || continue

    # Fail closed on malformed/non-commit input.
    resolved_sha="$(
      git rev-parse --verify "${sha}^{commit}" 2>/dev/null
    )" || {
      echo "TIER-GATE REFUSED: '$sha' does not resolve to a commit" >&2
      return 1
    }

    count=$((count + 1))

    msg="$(git log -1 --format=%B "$resolved_sha")" || {
      echo "TIER-GATE INTERNAL ERROR: cannot read commit message for $resolved_sha" >&2
      return 1
    }

    events="$(_tg_events_at "$resolved_sha")" || {
      echo "TIER-GATE REFUSED $(git rev-parse --short "$resolved_sha"): cannot read EVENTS authority" >&2
      return 1
    }

    if ! reason="$(tg_check_message "$events" "$msg")"; then
      echo "TIER-GATE REFUSED $(git rev-parse --short "$resolved_sha"): $reason" >&2
      return 1
    fi

    tier="$(printf '%s\n' "$msg" | tg_extract_trailer Tier || true)"

    case "$tier" in
      FULL)
        full=$((full + 1))
        ;;
      FAST|STATIC|DOCS)
        light=$((light + 1))
        ;;
      *)
        # tg_check_message already guarantees this cannot happen.
        echo "TIER-GATE INTERNAL ERROR: accepted unknown tier '$tier'" >&2
        return 1
        ;;
    esac
  done

  echo "TIER-GATE PASS: $count commit(s) gated (FULL=$full, lighter tiers=$light)"
  return 0
}
