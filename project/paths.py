"""Project-owned filesystem layout loaded from the project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from broadway.utils import require_keys

PROJECT_ROOT = Path(__file__).resolve().parent

# Project-relative path keys that resolve under the project root. Everything
# else in layout.yaml (``build:``/``retired:`` blocks) is repo-root-relative
# and exposed as string tuples, not resolved Paths.
_PATH_KEYS = ("config", "experiments", "experiment_configs", "observations")


def _strings(block: dict, key: str) -> tuple[str, ...]:
    """Return a layout block's string list as a tuple (empty when absent)."""
    if not isinstance(block, dict):
        return ()
    value = block.get(key)
    return tuple(value) if isinstance(value, list) else ()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    experiments: Path
    experiment_configs: Path
    observations: Path
    # Build/deploy surfaces (repo-root-relative). scripts/check_data_refs.py
    # gates every Dockerfile COPY/ADD source and ci.yml reference against them.
    build_copy_dirs: tuple[str, ...] = ()
    build_copy_files: tuple[str, ...] = ()
    build_contexts: tuple[str, ...] = ()
    build_dockerfiles: tuple[str, ...] = ()
    build_manifests: tuple[str, ...] = ()
    # Retired root-scoped locations (repo-root-relative). check_project_paths.py
    # derives its legacy-reference gate and leftover-file checks from these.
    retired_roots: tuple[str, ...] = ()
    retired_prefixes: tuple[str, ...] = ()
    retired_paths: tuple[str, ...] = ()

    @property
    def results(self) -> Path:
        return self.experiments / "results"


def load_project_paths(root: Path = PROJECT_ROOT) -> ProjectPaths:
    """Load the project-relative layout and reject paths escaping its root."""
    config_path = root / "config" / "layout.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require_keys(config, list(_PATH_KEYS), config_path.name)
    paths = {name: (root / config[name]).resolve() for name in _PATH_KEYS}
    resolved_root = root.resolve()
    if any(not path.is_relative_to(resolved_root) for path in paths.values()):
        raise ValueError(f"{config_path}: layout paths must remain under {resolved_root}")
    build = config.get("build") or {}
    retired = config.get("retired") or {}
    return ProjectPaths(
        resolved_root,
        build_copy_dirs=_strings(build, "copy_dirs"),
        build_copy_files=_strings(build, "copy_files"),
        build_contexts=_strings(build, "build_contexts"),
        build_dockerfiles=_strings(build, "dockerfiles"),
        build_manifests=_strings(build, "manifests"),
        retired_roots=_strings(retired, "roots"),
        retired_prefixes=_strings(retired, "prefixes"),
        retired_paths=_strings(retired, "paths"),
        **paths,
    )
