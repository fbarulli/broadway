#!/usr/bin/env bash
# check_docker_paths.sh — Dockerfile COPY/ADD source pre-flight (blocking gate).
# Fails loudly on missing build-context sources — the class that broke main
# CI (`COPY project/` with no project/ on the data-agnostic line), caught
# here in seconds instead of after a 10-minute image build.
# Branch-aware: WORKLOAD images (configs/tooling.yaml docker.images with
# workload:true — they bake the taxi reference pipeline or FROM its base)
# are SKIPPED with notice when project/ is absent. Skipping there is
# correct; failing is not. Platform images must resolve everywhere.
# NO HARDCODED VALUES: image table (dockerfile/context/workload) resolves
# from configs/tooling.yaml, whose contexts must match ci.yml's builds.
set -euo pipefail
cd "$(dirname "$0")/.."

TOOLING="configs/tooling.yaml"
# Workload presence = the project HPO spec (not merely a project/ dir, which
# stray caches can resurrect). Same rule as ci.yml's Detect step.
HAVE_PROJECT=0
[[ -f project/config/experiments/mlflow.yaml ]] && HAVE_PROJECT=1

python3 - "$TOOLING" "$HAVE_PROJECT" <<'EOF'
import re
import sys
from pathlib import Path

import yaml

tooling_path, have_project = sys.argv[1], sys.argv[2] == "1"
images = yaml.safe_load(Path(tooling_path).read_text(encoding="utf-8"))["docker"]["images"]

copy_rx = re.compile(r"^(COPY|ADD)\s+(.*)$")
failed: list[str] = []
skipped: list[str] = []
checked = 0
for img in images:
    dockerfile, context, workload = img["dockerfile"], img["context"], bool(img.get("workload"))
    if workload and not have_project:
        skipped.append(dockerfile)
        continue
    text = Path(dockerfile).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = copy_rx.match(line)
        if not m:
            continue
        args = m.group(2).split()
        if any(a.startswith("--from=") for a in args):
            continue  # stage-to-stage copy: no build-context source
        sources = [a for a in args if not a.startswith("--")]
        for src in sources[:-1]:
            if re.match(r"https?://", src):
                continue  # remote ADD: no local source
            candidate = Path(context) / src.rstrip("/")
            if not candidate.exists():
                failed.append(f"{dockerfile}:{lineno}: {src!r} missing under context {context!r}")
    checked += 1

for d in skipped:
    print(f"docker-paths: SKIP {d} (workload image, no project/ on this line)")
if failed:
    print("docker-paths: FAIL — missing build-context sources:")
    print("\n".join(f"  {f}" for f in failed))
    raise SystemExit(1)
print(f"docker-paths: PASS ({checked} Dockerfiles, {len(skipped)} workload-skipped)")
EOF
