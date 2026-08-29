"""Platform test hygiene — project-layer coupling is FORBIDDEN in the
platform suite.

Standing rule (agents/ledger/HANDOFF.md): every platform test runs on **generated data
only** and never references project-level data or configs. This guard
fails loudly the moment a platform test regresses:

* imports the project layer (``project.*``),
* reads a REAL data file — a ``read_parquet``/``read_csv`` whose path is not
  tmp-generated (``tmp_path``/``tmp_``/monkeypatched),
* references project configs (``configs/project/`` or ``project/config/``).

Dataset-demo tests live under ``project/tests/`` and are exempt (they test
the dataset layer with generated data). If you need a fixture, generate
it — never load project data. This file itself is exempt (it names the
patterns).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

# Project config references — always forbidden in platform tests.
_FORBIDDEN_CONFIG = [
    re.compile(r'configs/project/'),
    re.compile(r'project/config/'),
    re.compile(r'dataset="project"'),
    re.compile(r'experiment="project"'),
]

# Real-data reads: read_parquet/read_csv where the path is a literal or
# clearly not tmp-generated. A call like read_parquet(tmp_path / ...) or
# read_parquet(tmp_dataset) is fine (generated data).
_READ = re.compile(r"read_parquet\(([^)]*)\)|read_csv\(([^)]*)\)")
_PROJECT_PATH = re.compile(r'data/(raw|processed)/|"data/processed/|"data/raw/')

# project.* imports — the project demo layer has no place in platform tests.
_IMPORT = re.compile(r"^\s*(from project|import project)\b", re.MULTILINE)


def _platform_test_files() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


def _snippet(text: str, start: int) -> str:
    return text.splitlines()[text.count("\n", 0, start)].strip()


@pytest.mark.parametrize("path", _platform_test_files(), ids=lambda p: p.name)
def test_platform_test_has_no_project_coupling(path: Path) -> None:
    if path.name in {"test_platform_hygiene.py", "test_project_paths.py"}:
        return
    text = path.read_text()
    hits: list[str] = []

    for pattern in _FORBIDDEN_CONFIG:
        for m in pattern.finditer(text):
            hits.append(f"  line {text.count(chr(10), 0, m.start()) + 1}: {_snippet(text, m.start())}")

    for m in _IMPORT.finditer(text):
        hits.append(f"  line {text.count(chr(10), 0, m.start()) + 1}: {_snippet(text, m.start())}")

    for m in _READ.finditer(text):
        arg = (m.group(1) or m.group(2) or "").strip()
        if not arg or _PROJECT_PATH.search(arg):
            hits.append(f"  line {text.count(chr(10), 0, m.start()) + 1}: {_snippet(text, m.start())}")

    assert not hits, (
        f"{path.name} references project-level data/configs — "
        f"platform tests must use generated data only:\n" + "\n".join(hits)
    )


def _storytelling_modules() -> list[Path]:
    return sorted(
        p
        for root in (
            REPO_ROOT / "src/broadway/timeline",
            REPO_ROOT / "src/broadway/reports",
        )
        for p in root.glob("**/*.py")
        if "__pycache__" not in p.parts
    )


@pytest.mark.parametrize("path", _storytelling_modules(), ids=lambda p: p.name)
def test_storytelling_module_never_imports_broadway_evaluate(path: Path) -> None:
    text = path.read_text()
    assert "broadway.evaluate" not in text and "from broadway import evaluate" not in text, (
        f"{path.relative_to(REPO_ROOT)} imports broadway.evaluate — the "
        "storytelling layer (walkthrough/timeline/reporting) must not consume "
        "the CV path; evaluate/module.py is the sole production caller"
    )
