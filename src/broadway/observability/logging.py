"""Canonical process logging setup (LOGGING.md §1-3).

Only process entry points call ``setup_logging``; library modules obtain
``logging.getLogger(__name__)`` and never configure handlers or levels.
"""

from __future__ import annotations

import logging
from pathlib import Path

from broadway.config.schema import EnvironmentConfig
from broadway.observability.context import current_run_id
from broadway.paths import repo_root

FORMAT = "%(asctime)s [%(levelname)s] run=%(run_id)s %(name)s: %(message)s"

_configured = False


class RunIdFilter(logging.Filter):
    """Stamp every record with the current run id (LOGGING.md §5)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = current_run_id()
        return True


def _log_path(environment: EnvironmentConfig) -> Path:
    if environment.log_dir is None:
        raise ValueError("environment.log_dir is required for logging setup")
    if environment.log_file is None:
        raise ValueError("environment.log_file is required for logging setup")
    log_dir = Path(environment.log_dir)
    root = log_dir if log_dir.is_absolute() else repo_root() / log_dir
    return root / environment.log_file


def setup_logging(environment: EnvironmentConfig) -> None:
    """Configure the root logger once: one console handler, one file handler."""
    global _configured
    if _configured:
        return
    log_path = _log_path(environment)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(FORMAT)
    run_id_filter = RunIdFilter()
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path)
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        handler.addFilter(run_id_filter)
    root = logging.getLogger()
    root.setLevel(environment.log_level)
    root.addHandler(console)
    root.addHandler(file_handler)
    _configured = True
