"""Project-owned filesystem layout loaded from the project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from broadway.utils import require_keys

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    experiments: Path
    experiment_configs: Path
    observations: Path

    @property
    def results(self) -> Path:
        return self.experiments / "results"


def load_project_paths(root: Path = PROJECT_ROOT) -> ProjectPaths:
    """Load the project-relative layout and reject paths escaping its root."""
    config_path = root / "config" / "layout.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require_keys(
        config,
        ["config", "experiments", "experiment_configs", "observations"],
        config_path.name,
    )
    paths = {name: (root / config[name]).resolve() for name in config}
    resolved_root = root.resolve()
    if any(not path.is_relative_to(resolved_root) for path in paths.values()):
        raise ValueError(f"{config_path}: layout paths must remain under {resolved_root}")
    return ProjectPaths(resolved_root, **paths)
