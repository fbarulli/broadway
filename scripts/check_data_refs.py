"""Verify the build/deploy layer references only paths declared in the SSOT.

project/config/layout.yaml (loaded via project/paths.py) is the SINGLE SOURCE
OF TRUTH for what the images COPY and what CI builds. This gate closes the
"docker keeps breaking" class of bug where a Dockerfile COPY/ADD source or a
ci.yml dockerfile/manifest reference points at a path that was deleted or
never declared — removing a dataset or moving the project layout must land in
layout.yaml, and any reference that escapes it fails here.

Rules:
  * a Dockerfile COPY/ADD source must sit inside a declared build.copy_dirs
    directory or equal a declared build.copy_files entry, AND be git-committed
    (the build context is a fresh checkout);
  * every ci.yml ``-f <dockerfile>`` reference must be a declared
    build.dockerfiles entry, and every ci.yml ``k8s/*.yaml`` reference a
    declared build.manifests entry;
  * every declared build surface must itself exist on disk;
  * a config ``parquet:``/``path:``/``file:`` data path must exist or be a
    gitignored generated artifact.

Usage: python scripts/check_data_refs.py   # exit 0 green / 1 red
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from project.paths import load_project_paths

REPO = Path(__file__).resolve().parents[1]

# Data-asset suffixes worth closing. Source/config files (.py/.sh/.yaml) are
# already owned by ruff + shell checks; these are the ones nothing else pins.
DATA_SUFFIXES = {".parquet", ".csv", ".tsv", ".arrow", ".feather", ".pkl", ".json"}

# Repo subtrees that never belong to this checkout's build context.
_SKIP_PARTS = {".git", "deepseek-harness", "node_modules", ".venv", ".uv-cache"}


def _git_ignored(rel: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", "--", rel], cwd=REPO, check=False,
    ).returncode == 0


def _exists(rel: str) -> bool:
    return (REPO / rel).exists()


def _committed(rel: str) -> bool:
    """A Docker COPY source must be git-tracked: the build context is a fresh
    checkout, so a gitignored/untracked file is absent at COPY time."""
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=REPO, check=False, capture_output=True,
    ).returncode == 0


def _in_copy_dir(rel: str, dirs: tuple[str, ...]) -> bool:
    """True when ``rel`` is one of the declared COPY directories or beneath it."""
    return any(rel == d or rel.startswith(f"{d}/") for d in dirs)


def _dockerfile_refs() -> list[tuple[str, str]]:
    """(source, dockerfile_parent) for every COPY/ADD source token."""
    refs: list[tuple[str, str]] = []
    for df in sorted(REPO.rglob("Dockerfile*")):
        if _SKIP_PARTS & set(df.parts):
            continue
        parent = str(df.parent.relative_to(REPO))
        for line in df.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*(?:COPY|ADD)\s+(.+)$", line)
            if not m:
                continue
            for tok in m.group(1).split():
                # multi-stage (--from=), image-absolute (/uv, /app), env ($),
                # and bare-dot destination (. / ./ .. / ../) tokens
                if tok.startswith(("--", "/", "$", "~")) or tok in {".", "./", "..", "../"}:
                    continue
                refs.append((tok.strip("\"'"), parent))
    return refs


def _ci_lines() -> list[str]:
    """Non-comment lines of .github/workflows/ci.yml."""
    ci = REPO / ".github" / "workflows" / "ci.yml"
    return [
        line for line in ci.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _ci_dockerfile_refs() -> list[str]:
    """Every ``docker build -f <dockerfile>`` reference in ci.yml."""
    refs: list[str] = []
    for line in _ci_lines():
        refs.extend(re.findall(r"-f\s+(\S+)", line))
    return refs


def _ci_manifest_refs() -> list[str]:
    """Every repo-relative ``k8s/*.yaml`` reference in ci.yml."""
    refs: list[str] = []
    for line in _ci_lines():
        refs.extend(re.findall(r"\b(k8s/[A-Za-z0-9_.-]+\.ya?ml)\b", line))
    return refs


def _yaml_data_refs() -> list[str]:
    """Data-file paths declared under ``parquet:`` / ``path:`` / ``file:`` keys."""
    refs: list[str] = []
    roots = [REPO / "project", REPO / "configs"]
    for root in roots:
        for yaml in sorted(root.rglob("*.yaml")):
            if _SKIP_PARTS & set(yaml.parts):
                continue
            for line in yaml.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^\s*(?:parquet|path|file)\s*:\s*(\S+)", line)
                if not m:
                    continue
                val = m.group(1).strip("\"'")
                if Path(val).suffix.lower() in DATA_SUFFIXES:
                    refs.append(val)
    return refs


def main() -> int:
    paths = load_project_paths()
    missing: list[str] = []

    # Dockerfile COPY/ADD sources must be declared in the SSOT build surfaces
    # AND committed. A source is repo-rooted or relative to its Dockerfile dir.
    for tok, parent in _dockerfile_refs():
        candidates = [tok] if parent == "." else [tok, f"{parent}/{tok}"]
        declared = [
            c for c in candidates
            if c in paths.build_copy_files or _in_copy_dir(c, paths.build_copy_dirs)
        ]
        if not declared:
            missing.append(f"undeclared COPY/ADD source: {tok} ({parent})")
        elif not any(_committed(c) for c in declared):
            missing.append(f"dangling COPY/ADD source: {tok} ({parent})")

    # ci.yml dockerfile/manifest references must be declared in the SSOT.
    missing += [
        f"undeclared ci.yml dockerfile: {ref}"
        for ref in _ci_dockerfile_refs()
        if ref not in paths.build_dockerfiles
    ]
    missing += [
        f"undeclared ci.yml manifest: {ref}"
        for ref in _ci_manifest_refs()
        if ref not in paths.build_manifests
    ]

    # Every declared build surface must itself resolve (SSOT self-consistency).
    declared_surfaces = (
        *paths.build_dockerfiles, *paths.build_manifests,
        *paths.build_copy_files, *paths.build_copy_dirs, *paths.build_contexts,
    )
    missing += [
        f"declared build surface missing: {ref}"
        for ref in declared_surfaces
        if not _exists(ref)
    ]

    # Config data paths may legitimately be gitignored generated artifacts.
    missing += [
        f"dangling data reference: {ref}"
        for ref in _yaml_data_refs()
        if not _exists(ref) and not _git_ignored(ref)
    ]

    if missing:
        for rel in sorted(set(missing)):
            print(f"dangling data reference: {rel}", file=sys.stderr)
        print(f"{len(set(missing))} dangling data reference(s) — referenced but "
              f"not declared in the SSOT or not present", file=sys.stderr)
        return 1
    print("data-refs OK: COPY/ADD sources + ci.yml references match the SSOT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
