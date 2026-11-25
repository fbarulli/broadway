"""Check the dependency registry against pyproject.toml and platform imports.

Blocking findings: an unregistered direct dependency, an orphan registry row, a
duplicate row, or a ``project``-scoped dependency imported under ``src/broadway``.
Missing licenses and empty adapters are reported as notes and never fail.

This script is the single owner of dependency checks (CODE_STANDARD.md §4).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

Scope = Literal["platform", "project", "dev", "optional"]

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_NAME_SEPARATOR = re.compile(r"[-_.]+")
_DEFAULT_PYPROJECT = Path("pyproject.toml")
_DEFAULT_REGISTRY = Path("configs/dependencies.yaml")
_DEFAULT_SOURCE_ROOT = Path("src/broadway")


def canonical_name(name: str) -> str:
    """Return the PEP 503-normalized distribution name."""
    return _NAME_SEPARATOR.sub("-", name.strip().lower())


def requirement_name(requirement: str) -> str:
    """Return the distribution name of a PEP 508 requirement string."""
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(f"unparsable requirement: {requirement!r}")
    return canonical_name(match.group(1))


class RegistryRow(BaseModel):
    """One direct-dependency registry row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    owner: str
    scope: Scope
    purpose: str
    adapter: str = ""
    license: str = ""
    registered: date


class DependencyRegistry(BaseModel):
    """The parsed registry document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dependencies: list[RegistryRow]


def direct_dependency_names(pyproject: dict[str, Any]) -> set[str]:
    """Return every direct requirement name declared by pyproject."""
    project = pyproject["project"]
    requirements = [*project.get("dependencies", [])]
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    declared = {requirement_name(item) for item in requirements}
    return declared - {canonical_name(project["name"])}


def _distribution_modules() -> dict[str, set[str]]:
    """Map canonical distribution name to the top-level modules it provides."""
    modules: dict[str, set[str]] = {}
    for module, distributions in metadata.packages_distributions().items():
        for distribution in distributions:
            modules.setdefault(canonical_name(distribution), set()).add(module)
    return modules


def _modules_for(row: RegistryRow, modules: dict[str, set[str]]) -> set[str]:
    """Return the top-level import names a registry row may appear as."""
    name = canonical_name(row.name)
    return modules.get(name) or {name.replace("-", "_")}


def _project_scoped_modules(
    rows: list[RegistryRow], modules: dict[str, set[str]]
) -> dict[str, str]:
    """Map an importable module name to its project-scoped dependency."""
    scoped: dict[str, str] = {}
    for row in rows:
        if row.scope != "project":
            continue
        for module in _modules_for(row, modules):
            scoped.setdefault(module, row.name)
    return scoped


def _imported_modules(node: ast.AST) -> list[str]:
    """Return the top-level modules a single import node references."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _source_imports(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield every (file, top-level module) imported under root."""
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for module in _imported_modules(node):
                yield path, module


def _has_license(row: RegistryRow) -> bool:
    """Return whether a row records a usable SPDX id."""
    value = row.license.strip()
    return bool(value) and value.upper() != "NOASSERTION"


def _needs_adapter(row: RegistryRow) -> bool:
    """Return whether a non-dev row still lacks an owned adapter seam."""
    return row.scope != "dev" and not row.adapter.strip()


class DependencyChecker:
    """Coverage and import-boundary checks over one registry snapshot."""

    def __init__(
        self, pyproject: dict[str, Any], rows: list[RegistryRow], source_root: Path
    ) -> None:
        modules = _distribution_modules()
        self._declared = direct_dependency_names(pyproject)
        self._rows = rows
        self._source_root = source_root
        self._project_modules = _project_scoped_modules(rows, modules)

    @classmethod
    def from_configs(
        cls, pyproject: dict[str, Any], registry: dict[str, Any], source_root: Path
    ) -> DependencyChecker:
        """Build a checker from parsed pyproject and registry documents."""
        rows = DependencyRegistry.model_validate(registry).dependencies
        return cls(pyproject, rows, source_root)

    def violations(self) -> list[str]:
        """Return the blocking findings; an empty list means clean."""
        return [*self._coverage_violations(), *self._boundary_violations()]

    @classmethod
    def notes(cls, registry: dict[str, Any]) -> list[str]:
        """Return non-blocking provenance gaps for the report channel."""
        rows = DependencyRegistry.model_validate(registry).dependencies
        return [
            *(f"missing license: {row.name}" for row in rows if not _has_license(row)),
            *(f"empty adapter: {row.name}" for row in rows if _needs_adapter(row)),
        ]

    def _coverage_violations(self) -> list[str]:
        """Return missing, orphan, and duplicate registry rows."""
        registered = Counter(canonical_name(row.name) for row in self._rows)
        missing = sorted(self._declared - registered.keys())
        orphans = sorted(registered.keys() - self._declared)
        duplicates = sorted(name for name, count in registered.items() if count > 1)
        return [
            *(f"missing registry row: {name}" for name in missing),
            *(f"orphan registry row: {name}" for name in orphans),
            *(f"duplicate registry row: {name}" for name in duplicates),
        ]

    def _boundary_violations(self) -> list[str]:
        """Return platform imports of a project-scoped dependency."""
        violations: list[str] = []
        reported: set[tuple[Path, str]] = set()
        for path, module in _source_imports(self._source_root):
            dependency = self._project_modules.get(module)
            key = (path, module)
            if dependency is None or key in reported:
                continue
            reported.add(key)
            relative = path.relative_to(self._source_root)
            violations.append(f"import boundary: {relative} imports project-scoped '{dependency}'")
        return violations


def check_registry(
    pyproject: dict[str, Any],
    registry: dict[str, Any],
    source_root: Path | None = None,
) -> list[str]:
    """Return registry violations; an empty list means the registry is clean."""
    root = source_root if source_root is not None else _DEFAULT_SOURCE_ROOT
    return DependencyChecker.from_configs(pyproject, registry, root).violations()


def _parser() -> argparse.ArgumentParser:
    """Return the CLI parser with repository-relative defaults."""
    parser = argparse.ArgumentParser(description="Check the dependency registry.")
    parser.add_argument("--pyproject", type=Path, default=_DEFAULT_PYPROJECT)
    parser.add_argument("--registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--source-root", type=Path, default=_DEFAULT_SOURCE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker; 0 when clean, 1 when violations are listed."""
    args = _parser().parse_args(argv)
    pyproject = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    violations = check_registry(pyproject, registry, args.source_root)
    notes = DependencyChecker.notes(registry)
    for note in notes:
        print(f"NOTE: {note}")
    if violations:
        for violation in violations:
            print(f"VIOLATION: {violation}")
        print(f"DEPENDENCIES FAILED: {len(violations)} violation(s), {len(notes)} note(s)")
        return 1
    declared = len(direct_dependency_names(pyproject))
    print(f"DEPENDENCIES OK: {declared} direct dependencies registered, {len(notes)} note(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
