"""Project composition boundary for the taxi development layer."""

from __future__ import annotations

import os

from project.paths import load_project_paths


def activate_config_overlay() -> None:
    """Select this project's config tree without overriding a caller choice."""
    os.environ.setdefault("BROADWAY_CONFIG_OVERLAY_DIR", str(load_project_paths().config))


activate_config_overlay()
