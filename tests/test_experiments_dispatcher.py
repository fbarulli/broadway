"""Project CLI activates the project config overlay."""

from __future__ import annotations

import os
import subprocess
import sys

from broadway.paths import repo_root

REPO = repo_root()


def test_project_cli_activates_the_project_config_overlay() -> None:
    environment = os.environ.copy()
    environment.pop("BROADWAY_CONFIG_OVERLAY_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import project.cli; from broadway.config.loader import load_config; "
                "print(load_config('etl', dataset='olist', experiment='olist').dataset.name)"
            ),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "olist"
