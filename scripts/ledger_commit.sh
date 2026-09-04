#!/usr/bin/env bash
# ledger_commit.sh — THE commit path (front door), companion to
# scripts/ship.sh (back door). Every commit is LEDGER-LINKED and TIER-GATED
# BEFORE git commit runs, so a refused message never creates a commit to fix.
#
# Contract (the commit-side law, mirroring ship.sh's push-side contract):
#   1. The subject must be TYPE(scope): summary, TYPE from the ratified
#      vocabulary (history-derived): FEAT|FIX|STATE|GOVERNANCE|ANALYSIS|EVAL|
#      CHORE|REGISTRY|STYLE|REFACTOR.
#   2. The message must carry a 'Tier:' trailer: FULL|FAST|STATIC|DOCS.
#   3. Tier FULL must also carry a resolvable 'Reviewer:' verdict id.
#   4. The message must cite >=1 STATE row id (STATE-YYYYMMDD-NNN) that is a
#      CURRENT row at commit time — commit and ledger stay linked.
#
# Templates: agents/templates/commit-msg.md is the sanctioned scaffold.
#   bash scripts/ledger_commit.sh --template     # prints it
#
# Usage:
#   bash scripts/ledger_commit.sh --template     # scaffold
#   bash scripts/ledger_commit.sh -               # message on stdin
#   bash scripts/ledger_commit.sh <msg-file> [git-commit flags...]
# shellcheck shell=bash

set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "LEDGER-COMMIT REFUSED: not inside a Git repository." >&2
  exit 1
}
cd "$REPO_ROOT"

# shellcheck source=scripts/tier_gate.sh
. scripts/tier_gate.sh

die() { echo "LEDGER-COMMIT REFUSED: $*" >&2; exit 1; }

COMMIT_TEMPLATE="agents/templates/commit-msg.md"
STATE_FILE="agents/ledger/STATE.md"
TYPE_VOCAB='FEAT|FIX|STATE|GOVERNANCE|ANALYSIS|EVAL|CHORE|REGISTRY|STYLE|REFACTOR'

if [[ "${1:-}" == "--template" ]]; then
  [[ -f "$COMMIT_TEMPLATE" ]] || die "template $COMMIT_TEMPLATE missing."
  # print the scaffold section of the template
  awk '/^# SCAFFOLD/{flag=1; next} flag' "$COMMIT_TEMPLATE" | sed '/^$/{1d}'
  exit 0
fi

# ---------------------------------------------------------------------------
# MESSAGE ACQUISITION
# ---------------------------------------------------------------------------
args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
  die "no message given. Usage: bash scripts/ledger_commit.sh '-' or a message file (or --template for the scaffold)"
fi

message_file="${args[0]}"
cleanup=0
if [[ "$message_file" == "-" ]]; then
  tmp="$(mktemp)"; cleanup=1
  cat > "$tmp"; message_file="$tmp"
elif [[ ! -f "$message_file" && "$message_file" != -* ]]; then
  tmp="$(mktemp)"; cleanup=1
  printf '%s\n' "$message_file" > "$tmp"; message_file="$tmp"
fi
trap '[[ $cleanup -eq 1 ]] && rm -f "$message_file"' EXIT
[[ -f "$message_file" ]] || die "message file '$message_file' not found (pass '-' for stdin or a bare message string)."
message="$(cat "$message_file")"

# ---------------------------------------------------------------------------
# GATE 1: SUBJECT LINE GRAMMAR (TYPE(scope): summary)
# ---------------------------------------------------------------------------
echo "== ledger_commit: subject grammar =="
subject="$(printf '%s\n' "$message" | head -n 1)"
if ! printf '%s\n' "$subject" | grep -Eq "^($TYPE_VOCAB)\([a-z0-9_.-]+\): .+"; then
  echo "subject '$subject' violates the grammar." >&2
  echo "Required: TYPE(scope): summary — TYPE from: $TYPE_VOCAB" >&2
  die "see scaffold: bash scripts/ledger_commit.sh --template"
fi

# ---------------------------------------------------------------------------
# GATE 2: TIER TRAILER (ratified D34/D35 grammar via tier_gate.sh)
# ---------------------------------------------------------------------------
echo "== ledger_commit: tier-gate grammar =="
reason="$(tg_check_message "$(_tg_events_at HEAD)" "$message")"
if [[ -n "$reason" ]]; then
  echo "$reason" >&2
  die "tier-gate refused the message (fix the 'Tier:'/'Reviewer:' trailers)."
fi

# ---------------------------------------------------------------------------
# GATE 3: STATE ROW LINKAGE
# ---------------------------------------------------------------------------
echo "== ledger_commit: STATE-row linkage =="
state_ids="$(printf '%s\n' "$message" | grep -Eo 'STATE-[0-9]{8}-[0-9]{3}' | sort -u || true)"
if [[ -z "$state_ids" ]]; then
  die "message cites no STATE-YYYYMMDD-NNN row id. Add a 'State: STATE-...' trailer citing the CURRENT row this commit advances (create one first via: agents/tools/state_records.py record add)."
fi
while IFS= read -r sid; do
  [[ -n "$sid" ]] || continue
  if grep -Eq "^\|[[:space:]]*${sid}[[:space:]]*\|" "$STATE_FILE"; then
    continue   # a CURRENT row — the normal case
  fi
  # Terminal case: a close/void commit cites the row it dispositions; that
  # row has LEFT CURRENT and lives in the archive under its marker. The id
  # must resolve SOMEWHERE (current or archive), never free-float.
  if ! grep -rq "STATE-ARCHIVE:${sid}" agents/ledger/archive/; then
    die "cited $sid resolves in neither $STATE_FILE CURRENT nor the archive (create it first via: agents/tools/state_records.py record add)."
  fi
done <<< "$state_ids"

# ---------------------------------------------------------------------------
# COMMIT
# ---------------------------------------------------------------------------
echo "== ledger_commit: message gated — committing =="
git commit -F "$message_file" "${args[@]:1}"
commit_sha="$(git rev-parse HEAD)"
echo "== ledger_commit: committed $(git rev-parse --short "$commit_sha") =="
echo "== remember: close the loop — record close/void your STATE row when landed, then ship via scripts/ship.sh =="
