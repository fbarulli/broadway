#!/usr/bin/env bash
# pyright_advisory.sh — Pyright CONSULTANT gate (strict, advisory-only).
# NEVER FAILS the build by design: every path exits 0. mypy remains the
# enforcing checker; pyright runs at highest strictness to surface silent
# errors, data coercion, unexpected behavior, and data drops — as ADVICE.
# NO HARDCODED VALUES: version + scope resolve from configs/tooling.yaml
# (SSOT) and pyrightconfig.json include. Same config on taxi and main
# (shared via parity surface). Bump the version ONLY in configs/tooling.yaml.
# Skips cleanly when node is absent (e.g. node-less CI images).
set -euo pipefail
cd "$(dirname "$0")/.."

TOOLING="configs/tooling.yaml"

if ! command -v node >/dev/null 2>&1; then
  echo "pyright-advisory: SKIP (node absent — advisory only, not failing build)"
  exit 0
fi

# Resolve version + scope from the SSOT (no inline defaults to drift).
resolved="$(python3 - "$TOOLING" <<'EOF'
import json
import sys
from pathlib import Path
tooling = Path(sys.argv[1])
try:
    import yaml
    data = yaml.safe_load(tooling.read_text(encoding="utf-8"))
    version = data["pyright"]["version"]
    scope = data["pyright"]["scope"]
    mode = data["pyright"].get("mode", "strict")
except Exception as exc:  # advisory: report, never fail
    print(f"pyright-advisory: unreadable {sys.argv[1]} ({exc}) — not failing build")
    raise SystemExit(3)
if not version or not scope:
    print(f"pyright-advisory: empty pyright.version/scope in {sys.argv[1]} — not failing build")
    raise SystemExit(3)
print(json.dumps({"version": str(version), "scope": [str(s) for s in scope], "mode": str(mode)}))
EOF
)" || { exit 0; }
PYRIGHT_VERSION="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["version"])' "$resolved")"
PYRIGHT_MODE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["mode"])' "$resolved")"
mapfile -t PYRIGHT_SCOPE < <(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["scope"]))' "$resolved")

out="$(mktemp)"
trap 'rm -f "$out"' EXIT
# NOTE: pyright exits 1 when findings exist — that is the NORMAL advisory
# case, not a failure. Only rc>1 (crash/config error) takes the degraded path.
rc=0
npx --yes "pyright@${PYRIGHT_VERSION}" --outputjson "${PYRIGHT_SCOPE[@]}" >"$out" 2>/dev/null || rc=$?
if [[ $rc -gt 1 ]]; then
  echo "pyright-advisory: pyright run failed (rc=$rc; advisory — not failing build)"
  exit 0
fi

if ! PYRIGHT_VERSION="$PYRIGHT_VERSION" PYRIGHT_MODE="$PYRIGHT_MODE" python3 - "$out" <<'EOF'
import collections
import json
import os
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
    f"(consultant {os.environ.get('PYRIGHT_MODE', '?')}/"
    f"pyright@{os.environ.get('PYRIGHT_VERSION', '?')}, non-blocking; "
    f"mypy enforces). top: {top}"
)
EOF
then
  echo "pyright-advisory: summary failed (advisory — not failing build)"
fi
exit 0
