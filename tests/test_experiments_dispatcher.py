"""Non-mutating command-surface checks for the root experiments dispatcher."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPECTED_COMMANDS = {"ols", "diagnostics", "qq_legend", "verify"}


def _help_result() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "project/experiments.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_help_lists_exact_dispatcher_commands() -> None:
    """The help path exposes only the dispatcher commands without execution."""
    result = _help_result()
    assert result.returncode == 0
    match = re.search(r"\{(?P<commands>[^}]+)\}", result.stdout)
    assert match is not None
    assert set(match.group("commands").split(",")) == EXPECTED_COMMANDS


def test_dispatcher_uses_project_configured_data_and_results_paths() -> None:
    source = (REPO / "project/experiments.py").read_text(encoding="utf-8")
    assert 'ROOT / "data"' not in source
    assert 'PATHS.results / "ols"' in source
    assert 'PATHS.results / "diagnostics"' in source
    assert 'PATHS.results / "qq_legend"' in source


def test_project_cli_activates_the_project_config_overlay() -> None:
    environment = os.environ.copy()
    environment.pop("BROADWAY_CONFIG_OVERLAY_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import project.cli; from broadway.config.loader import load_config; "
                "print(load_config('etl', dataset='taxi', experiment='taxi').dataset.name)"
            ),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "taxi"
