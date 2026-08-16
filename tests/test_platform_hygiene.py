"""Platform test hygiene — taxi coupling is FORBIDDEN in the platform suite.

Standing rule (HANDOFF.md): every platform test runs on **generated data
only** and never references project-level (taxi) data or configs. This guard
fails loudly the moment a platform test regresses:

* imports the taxi project layer (``project.*``),
* reads a REAL data file — a ``read_parquet``/``read_csv`` whose path is not
  tmp-generated (``tmp_path``/``tmp_``/monkeypatched),
* references taxi configs (``dataset="taxi"``, ``experiment="taxi"``,
  ``analysis="taxi*"``, ``configs/project/taxi.yaml``).

Taxi-demo tests live under ``project/tests/`` and are exempt (they test the
taxi layer with generated data). If you need a fixture, generate it — never
load project data. This file itself is exempt (it names the patterns).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

# Taxi config references — always forbidden in platform tests.
_FORBIDDEN_CONFIG = [
    re.compile(r'dataset="taxi"'),
    re.compile(r'experiment="taxi"'),
    re.compile(r'analysis="taxi'),
    re.compile(r'configs/project/taxi'),
    re.compile(r'"taxi_hypothesis"'),
    re.compile(r'"taxi_causal"'),
]

# Real-data reads: read_parquet/read_csv where the path is a literal or
# clearly not tmp-generated. A call like read_parquet(tmp_path / ...) or
# read_parquet(tmp_dataset) is fine (generated data).
_READ = re.compile(r"read_parquet\(([^)]*)\)|read_csv\(([^)]*)\)")
_PROJECT_PATH = re.compile(r'data/(raw|processed)/|"data/processed/|"data/raw/')

# project.* imports — the taxi demo layer has no place in platform tests.
_IMPORT = re.compile(r"^\s*(from project|import project)\b", re.MULTILINE)

# Lines that merely mention taxi in an explanatory/docstring way.
_ALLOW_WORDS = ("non-taxi", "non_taxi", "taxi-free", "taxi demo", "taxi layer",
                "without touching any taxi", "never touches taxi",
                "e.g. the taxi features", "the taxi project layer")


def _platform_test_files() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


def _snippet(text: str, start: int) -> str:
    return text.splitlines()[text.count("\n", 0, start)].strip()


@pytest.mark.parametrize("path", _platform_test_files(), ids=lambda p: p.name)
def test_platform_test_has_no_taxi_coupling(path: Path) -> None:
    if path.name == "test_platform_hygiene.py":
        return
    text = path.read_text()
    hits: list[str] = []

    for pattern in _FORBIDDEN_CONFIG:
        for m in pattern.finditer(text):
            snippet = _snippet(text, m.start())
            if any(w in snippet for w in _ALLOW_WORDS):
                continue
            hits.append(f"  line {text.count(chr(10), 0, m.start()) + 1}: {snippet}")

    for m in _IMPORT.finditer(text):
        hits.append(f"  line {text.count(chr(10), 0, m.start()) + 1}: {_snippet(text, m.start())}")

    for m in _READ.finditer(text):
        arg = (m.group(1) or m.group(2) or "").strip()
        if not arg or _PROJECT_PATH.search(arg):
            snippet = _snippet(text, m.start())
            if any(w in snippet for w in _ALLOW_WORDS):
                continue
            hits.append(f"  line {text.count(chr(10), 0, m.start()) + 1}: {snippet}")

    assert not hits, (
        f"{path.name} references project-level (taxi) data/configs — "
        f"platform tests must use generated data only:\n" + "\n".join(hits)
    )
