#!/usr/bin/env bash
# E2E determinism comparator — makes SKLEARN_PIPELINES.md's
# "End-to-end verification criteria" structural instead of prose.
#
# Usage:
#   scripts/check_e2e_determinism.sh <artifacts_dir_1> <artifacts_dir_2>
#   scripts/check_e2e_determinism.sh --run
#
# Core mode walks both artifact trees and compares every JSON pair after
# canonical normalization (parsed and re-dumped with sorted keys). Only the
# volatile-by-design fields named in the doc are excluded from equality:
#   trace.created_at  artifact_path  promote  reason  warnings
#   comparison.metrics.<metric>.{champion,delta,delta_pct}
# (delta/delta_pct are champion-derived state — identical invocations against
# a stale champion differ in them by design; verified live via --run.)
# Exit 0 + "DETERMINISM OK" when only whitelisted fields differ; exit 1 with
# one "<file>: <field path>" line per offender otherwise (missing counterpart
# files reported as "<file>: missing counterpart").
#
# --run executes the documented chain twice
# (ds-pipeline full --dataset test --experiment baseline --analysis test)
# against an ephemeral MLflow server on a scratch port/db, then compares the
# two artifact roots with the core comparator. Must be run from the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# $1=root -> relative file paths, sorted (paths with newlines unsupported)
tree_files() {
  (cd "$1" && find . -type f | sed 's|^\./||' | sort)
}

# $1=a $2=b -> prints differing leaf paths, whitelist excluded (empty = equal)
json_diff() {
  python3 - "$1" "$2" <<'PY'
import json
import re
import sys

# Volatile-by-design fields (SKLEARN_PIPELINES.md, "End-to-end verification
# criteria"): timestamps, MLflow-assigned URIs, champion-registry state.
# delta/delta_pct join champion: they are derived from the mutable champion,
# so identical invocations against different registry state differ in them.
EXACT = {"trace.created_at", "artifact_path", "promote", "reason", "warnings"}
PATTERN = re.compile(r"comparison\.metrics\.[^.]*\.(champion|delta|delta_pct)")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from leaves(value, path)
    else:
        yield prefix


def value_at(obj, path):
    node = obj
    for part in path.split("."):
        node = node[part]
    return node


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        first = json.load(fh)
    with open(sys.argv[2], encoding="utf-8") as fh:
        second = json.load(fh)
    paths_first, paths_second = set(leaves(first)), set(leaves(second))
    for path in sorted(paths_first | paths_second):
        if path in EXACT or PATTERN.fullmatch(path):
            continue
        if path not in paths_first or path not in paths_second:
            print(path)
        elif canon(value_at(first, path)) != canon(value_at(second, path)):
            print(path)


main()
PY
}

# $1=a $2=b $3=rel -> 0 when equal; prints "<rel>: <field>" per offender
compare_file() {
  local a="$1" b="$2" rel="$3" field matched=0
  if [[ "$rel" != *.json ]]; then
    if cmp -s "$a" "$b"; then
      return 0
    fi
    echo "$rel: <non-JSON content differs>"
    return 1
  fi
  while IFS= read -r field; do
    echo "$rel: $field"
    matched=1
  done < <(json_diff "$a" "$b")
  [[ $matched -eq 0 ]]
}

# $1=dir1 $2=dir2 -> 0 when deterministic, 1 when anything differs
compare_trees() {
  local dir1="$1" dir2="$2"
  local list1 list2 rel failed=0
  list1="$(mktemp)"
  list2="$(mktemp)"
  tree_files "$dir1" >"$list1"
  tree_files "$dir2" >"$list2"
  while IFS= read -r rel; do
    echo "$rel: missing counterpart (present only in $dir1)"
    failed=1
  done < <(comm -23 "$list1" "$list2")
  while IFS= read -r rel; do
    echo "$rel: missing counterpart (present only in $dir2)"
    failed=1
  done < <(comm -13 "$list1" "$list2")
  while IFS= read -r rel; do
    if ! compare_file "$dir1/$rel" "$dir2/$rel" "$rel"; then
      failed=1
    fi
  done < <(comm -12 "$list1" "$list2")
  rm -f "$list1" "$list2"
  if [[ $failed -eq 0 ]]; then
    echo "DETERMINISM OK"
    return 0
  fi
  return 1
}

# -> prints a free TCP port on 127.0.0.1
pick_free_port() {
  python3 - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

# $1=port — poll the server /health endpoint until it answers
wait_for_server() {
  local port="$1" tries=0
  until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [[ $tries -ge 60 ]]; then
      echo "MLflow server did not become ready on port $port" >&2
      return 1
    fi
    sleep 1
  done
}

SERVER_PID=""

# Executes the documented chain twice on an ephemeral MLflow server and
# compares the two artifact roots. Scratch copies only — never writes into
# the repo tree (configs are copied, data/demo are scratch-side).
run_e2e() {
  local scratch port run1 run2
  scratch="$(mktemp -d)"
  port="$(pick_free_port)"
  uv run --project "$REPO_ROOT" mlflow server \
    --backend-store-uri "sqlite:///$scratch/.mlflow.db" \
    --artifacts-destination "file://$scratch/mlruns" \
    --host 127.0.0.1 --port "$port" >"$scratch/mlflow.log" 2>&1 &
  SERVER_PID=$!
  trap 'if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
  wait_for_server "$port"
  cp -r "$REPO_ROOT/configs" "$scratch/configs"
  sed -i "s|^mlflow_tracking_uri: .*|mlflow_tracking_uri: http://127.0.0.1:$port|" \
    "$scratch/configs/environment/development.yaml"
  ln -s "$REPO_ROOT/demo" "$scratch/demo"
  mkdir -p "$scratch/data"
  run1="$scratch/run1"
  run2="$scratch/run2"
  (
    cd "$scratch"
    BROADWAY_CONFIGS_DIR="$scratch/configs" \
      uv run --project "$REPO_ROOT" ds-pipeline full \
      --dataset test --experiment baseline --analysis test
    mv artifacts "$run1"
    BROADWAY_CONFIGS_DIR="$scratch/configs" \
      uv run --project "$REPO_ROOT" ds-pipeline full \
      --dataset test --experiment baseline --analysis test
    mv artifacts "$run2"
  )
  kill "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  compare_trees "$run1" "$run2"
}

if [[ "${1:-}" == "--run" ]]; then
  run_e2e
elif [[ $# -eq 2 && -d "$1" && -d "$2" ]]; then
  compare_trees "$1" "$2"
else
  echo "usage: $0 <artifacts_dir_1> <artifacts_dir_2> | --run" >&2
  exit 2
fi
