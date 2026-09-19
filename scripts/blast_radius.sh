#!/usr/bin/env bash
# blast_radius.sh — THE go-to blast-radius command (single entry point).
# Runs BOTH mandated queries for a touched surface:
#   1. governance: render_gates.py --blast-radius (owners, gates, if_changed)
#   2. structural: graphify affected (reverse traversal: real callers w/ file:line)
# Every worker contract pastes this output BEFORE implementing (see
# CONTRACT_TEMPLATE.md); reviewers re-run it. No blast-radius evidence,
# no implementation reasoning.
# Usage: bash scripts/blast_radius.sh <repo-relative-path> [symbol]
#   path:   e.g. src/broadway/contracts/pandera.py (governance query)
#   symbol: e.g. DatasetContract (structural reverse traversal; skipped
#           with notice when omitted)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <repo-relative-path> [symbol]" >&2
  exit 2
fi
TARGET="$1"
SYMBOL="${2:-}"

echo "==> blast radius: governance (gates/owners)"
bash scripts/uv.sh run --extra dev python agents/tools/render_gates.py --blast-radius "$TARGET"

echo ""
echo "==> blast radius: structural (graphify reverse traversal)"
if [[ -z "$SYMBOL" ]]; then
  echo "SKIP structural: no symbol given (usage: $0 <path> [symbol])"
elif [[ ! -f graphify-out/graph.json ]]; then
  echo "SKIP structural: graphify-out/graph.json absent — rebuild with: graphify update . --no-cluster"
else
  graphify affected "$SYMBOL" --depth 2 2>/dev/null || graphify affected "$SYMBOL" --depth 2
fi
