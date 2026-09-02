"""Verify every local DATA-file path referenced by Dockerfiles and configs exists.

Catches the class of "docker keeps breaking" bug where a Dockerfile/COPY or a
config ``parquet:`` key points at a data asset that was deleted or never
committed (e.g. k8s/optuna/Dockerfile.worker COPYing
project/experiments/results/univariate/sample_evidence/sample_evidence.parquet
after the taxi results were removed).

Rule: a referenced local path must either exist on disk OR be a gitignored
generated artifact. Anything else is a dangling reference and fails the check.
This is the data-asset analogue of the gate registry's closure test — the
knowledge graph is NOT required to answer "does every referenced file exist?".

Usage: python scripts/check_data_refs.py   # exit 0 green / 1 red
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

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


def _normalize(ref: str) -> str | None:
    """Resolve a config/Dockerfile path token to a repo-relative path.

    Handles the three conventions the repo uses:
      * repo-relative ``project/...``            -> unchanged
      * in-image absolute ``/app/project/...``   -> strip ``/app/``
      * project-relative ``experiments/...`` or  -> prefix ``project/``
        ``config/...`` (configs resolve against ``project/``)
    Returns None for tokens that are not local file references.
    """
    ref = ref.strip().strip("\"'")
    ref = ref.removeprefix("/app/")
    if ref.startswith("/"):
        return None  # image/absolute path with no local source
    if ref.startswith(("experiments/", "config/", "results/")):
        ref = f"project/{ref}"
    if ref.startswith(("$", "~", "--")):
        return None
    return ref


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
                # multi-stage (--from=), image-absolute (/uv, /app), and env ($) tokens
                if tok.startswith(("--", "/", "$", "~")):
                    continue
                refs.append((tok, parent))
    return refs


def _committed(rel: str) -> bool:
    """A Docker COPY source must be git-tracked: the build context is a fresh
    checkout, so a gitignored/untracked file is absent at COPY time."""
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=REPO, check=False, capture_output=True,
    ).returncode == 0


def _yaml_parquet_refs() -> list[str]:
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
    # Docker COPY sources must be committed (build context = git checkout);
    # a source may be repo-rooted or relative to its Dockerfile's directory.
    missing: list[str] = []
    seen: set[str] = set()
    for tok, parent in _dockerfile_refs():
        norm = _normalize(tok)
        if norm is None:
            continue
        candidates = [norm] if parent == "." else [norm, f"{parent}/{norm}"]
        if not any(_committed(c) for c in candidates) and norm not in seen:
            missing.append(norm)
            seen.add(norm)
    yaml_refs = {r for r in (_normalize(x) for x in _yaml_parquet_refs()) if r}
    # YAML data paths may legitimately be gitignored generated artifacts.
    missing += sorted(r for r in yaml_refs if not _exists(r) and not _git_ignored(r))
    if missing:
        for rel in missing:
            print(f"dangling data reference: {rel}", file=sys.stderr)
        print(f"{len(missing)} dangling data reference(s) — referenced but neither "
              f"committed nor present", file=sys.stderr)
        return 1
    print("data-refs OK: COPY/ADD sources committed, config paths resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
