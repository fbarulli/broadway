#!/usr/bin/env bash
# pyright_advisory.sh — Pyright ADVISORY gate (compare-vs-mypy period).
# NEVER FAILS the build by design: every path exits 0. mypy remains the
# enforcing checker; this gate only reports counts for the keep-vs-switch
# comparison. Scope/options live in pyrightconfig.json (single source).
# Skips cleanly when node is absent (e.g. node-less CI images).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v node >/dev/null 2>&1; then
  echo "pyright-advisory: SKIP (node absent — advisory only, not failing build)"
  exit 0
fi

out="$(mktemp)"
trap 'rm -f "$out"' EXIT
# NOTE: pyright exits 1 when findings exist — that is the NORMAL advisory
# case, not a failure. Only rc>1 (crash/config error) takes the degraded path.
rc=0
npx --yes pyright@1.1.414 --outputjson src/broadway >"$out" 2>/dev/null || rc=$?
if [[ $rc -gt 1 ]]; then
  echo "pyright-advisory: pyright run failed (rc=$rc; advisory — not failing build)"
  exit 0
fi

if ! python3 - "$out" <<'EOF'
import collections
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
except (OSError, ValueError) as exc:
    print(f"pyright-advisory: unreadable output ({exc}) — not failing build")
    raise SystemExit(0)
summary = data.get("summary", {})
errors = summary.get("errorCount", "?")
files = summary.get("filesAnalyzed", "?")
rules = collections.Counter(
    g.get("rule", "<no-rule>") for g in data.get("generalDiagnostics", [])
)
top = ", ".join(f"{rule}={count}" for rule, count in rules.most_common(5))
print(
    f"pyright-advisory: {errors} errors across {files} files "
    f"(advisory, non-blocking; mypy enforces). top: {top}"
)
EOF
then
  echo "pyright-advisory: summary failed (advisory — not failing build)"
fi
exit 0
