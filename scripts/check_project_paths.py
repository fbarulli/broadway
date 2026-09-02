"""Fail closed when project experiment paths escape the project contract."""

from __future__ import annotations

import re
from pathlib import Path

from project.paths import load_project_paths

REPO = Path(__file__).resolve().parents[1]
CHECKED_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _legacy_pattern(prefixes: tuple[str, ...], paths: tuple[str, ...]) -> re.Pattern[str]:
    """Build the legacy-reference matcher from the SSOT retired declarations.

    A prefix is a location the layout retired (old experiments/config roots);
    a path is an exact retired config file. Root-scoped references are only
    legacy when they are NOT under ``project/`` (the current layout) or
    ``config/`` (the current ``project/config`` root).
    """
    parts: list[str] = []
    for prefix in prefixes:
        if prefix.startswith("/"):
            parts.append(re.escape(prefix))
        else:
            parts.append(f"(?<!project/)(?<!config/){re.escape(prefix)}")
    for path in paths:
        parts.append(f"(?<!project/){re.escape(path)}")
    return re.compile("|".join(parts))


def runtime_files(root: Path) -> list[Path]:
    """Return working-tree runtime/config files that can contain filesystem paths."""
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if EXCLUDED_DIRS.intersection(path.parts) or not path.is_file():
            continue
        if relative in {Path("scripts/check_project_paths.py"), Path("scripts/main_day_sync.sh")}:
            continue
        if relative.parts[:2] == ("project", "config"):
            continue
        if relative.name != ".gitignore" and relative.parts[0] not in {"src", "project", "scripts", "k8s", ".github"}:
            continue
        if path.suffix in CHECKED_SUFFIXES or path.name.startswith("Dockerfile") or path.name == ".gitignore":
            files.append(path)
    return files


def legacy_path_references(root: Path, paths: list[Path]) -> list[str]:
    """Return tracked runtime references to retired root-scoped locations."""
    layout = load_project_paths()
    pattern = _legacy_pattern(layout.retired_prefixes, layout.retired_paths)
    return [str(path.relative_to(root)) for path in paths if pattern.search(path.read_text(encoding="utf-8"))]


def main() -> None:
    """Check layout targets and reject retired root-scoped project paths."""
    paths = load_project_paths()
    missing = [
        str(path)
        for path in (paths.config, paths.experiments, paths.experiment_configs)
        if not path.is_dir()
    ]
    legacy = legacy_path_references(REPO, runtime_files(REPO))
    retired_configs = [path for path in paths.retired_paths if (REPO / path).exists()]
    retired_roots = [path for path in paths.retired_roots if (REPO / path).exists()]
    if missing or legacy or retired_configs or retired_roots:
        raise SystemExit(
            f"PROJECT PATH ERROR: missing={missing}; legacy={legacy}; "
            f"retired_configs={retired_configs}; retired_roots={retired_roots}"
        )
    print(f"PROJECT PATHS OK: experiments={paths.experiments}; configs={paths.experiment_configs}")


if __name__ == "__main__":
    main()
