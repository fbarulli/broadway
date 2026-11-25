"""Pins for the dependency registry checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from broadway.paths import repo_root

REPO = repo_root()


def _load_checker():
    path = REPO / "scripts" / "check_dependencies.py"
    name = "dependency_checker_test"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(name: str) -> dict[str, str]:
    return {
        "name": name,
        "owner": "platform",
        "scope": "platform",
        "purpose": f"{name} test purpose",
        "adapter": "",
        "license": "MIT",
        "added": "2026-11-25",
    }


def test_check_registry_passes_when_complete_and_flags_a_missing_row(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    pyproject = {"project": {"name": "demo", "dependencies": ["alpha>=1.0", "beta[extra]>=2"]}}
    source_root = tmp_path / "src" / "broadway"
    source_root.mkdir(parents=True)

    complete = {"dependencies": [_row("alpha"), _row("beta")]}
    assert checker.check_registry(pyproject, complete, source_root) == []

    incomplete = {"dependencies": [_row("alpha")]}
    assert checker.check_registry(pyproject, incomplete, source_root) == [
        "missing registry row: beta"
    ]
