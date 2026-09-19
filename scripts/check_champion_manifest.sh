#!/usr/bin/env bash
# Champion manifest — lists deployed champions by logging path (Slice 4,
# decision 3: ModelPyFunc retirement is a checked condition, not a vibe).
#
# Usage:
#   scripts/check_champion_manifest.sh --tracking-uri <uri>
#   scripts/check_champion_manifest.sh --tracking-uri <uri> --strict
#
# Prints one line per bucket (bare_model / pipeline_signature / ambiguous)
# with champion names + artifact URIs, and the reason for any ambiguous item.
# Exit 0 always for reporting — the manifest is not a gate. --strict turns it
# into the retirement-condition check: exit 1 when the bare_model or ambiguous
# bucket is non-empty. Exit 2 = usage error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" != "--tracking-uri" || $# -lt 2 ]]; then
  echo "usage: $0 --tracking-uri <uri> [--strict]" >&2
  exit 2
fi

TRACKING_URI="$2"
STRICT=0
if [[ "${3:-}" == "--strict" ]]; then
  STRICT=1
elif [[ -n "${3:-}" ]]; then
  echo "usage: $0 --tracking-uri <uri> [--strict]" >&2
  exit 2
fi

uv run --project "$REPO_ROOT" python - "$TRACKING_URI" "$STRICT" <<'PY'
import sys

from broadway.training.mlflow_utils import (
    AMBIGUOUS,
    BARE_MODEL,
    PIPELINE_SIGNATURE,
    list_champions,
)

tracking_uri, strict = sys.argv[1], sys.argv[2] == "1"
champions = list_champions(tracking_uri)
buckets = {BARE_MODEL: [], PIPELINE_SIGNATURE: [], AMBIGUOUS: []}
for champion in champions:
    buckets[champion.bucket].append(champion)

print(f"champion manifest — {len(champions)} champion(s) by logging path ({tracking_uri})")
for bucket, label in (
    (BARE_MODEL, "bare_model (ModelPyFunc-wrapped)"),
    (PIPELINE_SIGNATURE, "pipeline_signature (new path)"),
    (AMBIGUOUS, "ambiguous"),
):
    print(f"{label} ({len(buckets[bucket])}):")
    for champion in sorted(buckets[bucket], key=lambda c: c.model_name):
        reason = f"  reason: {champion.reason}" if champion.reason else ""
        artifact = champion.artifact_uri or "<unavailable>"
        print(f"  - {champion.model_name} (v{champion.version}) artifact: {artifact}{reason}")

if strict:
    if buckets[BARE_MODEL] or buckets[AMBIGUOUS]:
        print("STRICT: bare_model or ambiguous champions present — ModelPyFunc retirement NOT cleared")
        sys.exit(1)
    print("STRICT: OK — no bare_model or ambiguous champions")
PY
