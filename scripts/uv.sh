#!/usr/bin/env bash
# Host-local uv cache selection. Never creates a repository-local cache.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: bash scripts/uv.sh <uv arguments...>" >&2
  exit 2
fi

cache_is_writable() {
  mkdir -p "$1" 2>/dev/null && [[ -d $1 && -w $1 ]]
}

if [[ -n ${UV_CACHE_DIR:-} ]]; then
  cache_root="$UV_CACHE_DIR"
  if ! cache_is_writable "$cache_root"; then
    echo "UV CACHE ERROR: requested UV_CACHE_DIR is not writable: $cache_root" >&2
    exit 1
  fi
else
  preferred_root="${XDG_CACHE_HOME:-$HOME/.cache}/uv"
  if cache_is_writable "$preferred_root"; then
    cache_root="$preferred_root"
  else
    cache_root="${TMPDIR:-/tmp}/broadway-uv-cache"
    if ! cache_is_writable "$cache_root"; then
      echo "UV CACHE ERROR: neither preferred cache ($preferred_root) nor fallback ($cache_root) is writable" >&2
      exit 1
    fi
    echo "UV CACHE NOTICE: preferred cache unavailable; using host-local fallback: $cache_root" >&2
  fi
fi

export UV_CACHE_DIR="$cache_root"
exec uv "$@"
