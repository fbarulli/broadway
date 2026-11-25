"""scripts/verify_live.py — evidence-recorder public behavior (LIVE_VERIFICATION.md §3).

Subprocess-level only: the script is exercised exactly as a caller runs it.
tmp_path is the output dir, so the repo ``artifacts/`` custody guard stays
untouched. No network, no real dataset (a non-parquet path is honest provenance
for a smoke command).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from broadway.paths import repo_root

REPO_ROOT = repo_root()
SCRIPT = REPO_ROOT / "scripts" / "verify_live.py"


def _run(tmp_path: Path, command: str) -> subprocess.CompletedProcess[str]:
    """Invoke the recorder against tmp_path evidence output."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--step",
            "smoke",
            "--command",
            command,
            "--dataset",
            str(tmp_path / "sample.csv"),
            "--output-dir",
            str(tmp_path / "evidence"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _records(tmp_path: Path) -> list[dict[str, object]]:
    """Load every evidence record written under tmp_path."""
    files = sorted((tmp_path / "evidence").glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def test_verify_live_records_evidence_and_preserves_exit_code(tmp_path: Path) -> None:
    passed = _run(tmp_path, f'{sys.executable} -c "print(\'ok\')"')
    assert passed.returncode == 0
    assert Path(passed.stdout.strip()).is_file()
    success = _records(tmp_path)[0]
    assert success["exit_code"] == 0
    assert success["run_id"]
    assert success["rows"] is None
    assert "ok" in success["stdout_tail"]

    failed = _run(tmp_path, f'{sys.executable} -c "import sys; sys.exit(3)"')
    assert failed.returncode == 3
    assert sorted(record["exit_code"] for record in _records(tmp_path)) == [0, 3]
