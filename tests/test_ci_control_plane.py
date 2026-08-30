"""Failure behavior for the local CI control plane."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vulture_finding_makes_static_ci_red(tmp_path: Path) -> None:
    """A direct Vulture finding must fail static local CI loudly."""
    real_uv = shutil.which("uv")
    assert real_uv is not None
    shim = tmp_path / "uv"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1 $2" == "run vulture" ]]; then\n'
        '  printf "%s\\n" "src/example.py:1: unused function\\n"\n'
        "  exit 3\n"
        "fi\n"
        f'exec "{real_uv}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env = os.environ | {"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", "scripts/run_local_ci.sh", "--static"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "FAIL vulture" in output
    assert "src/example.py:1: unused function" in output
    assert "LOCAL-CI RED" in output
