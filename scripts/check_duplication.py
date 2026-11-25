#!/usr/bin/env python3
"""Near-duplicate function detector for CONSOLIDATION.md §4 (D2, advisory).

Every function/method with at least ``min_statements`` statements is reduced to
a normalized AST signature (identifiers, constants, and docstrings erased; node
types and structure kept) and grouped. A signature spanning two or more files is
a cross-file duplicate candidate; a function name defined under two different
owners is a pointer to verify.

All thresholds, roots, exclusions, and the report path resolve from the
duplication config, whose default location derives from this script's own name
(``scripts/check_<name>.py`` -> ``configs/<name>.yaml``) — no value is copied
into code. A configured root absent from this checkout (e.g. ``project/`` on
the data-agnostic main line) is skipped with a notice; only a config with no
existing root fails. Advisory mode always exits 0; blocking mode exits 1 on any
group.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from broadway.paths import repo_root

LOGGER = logging.getLogger("duplication-checker")

REPO_ROOT = repo_root()
CONFIG_DIR = "configs"
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
Signature = tuple[Any, ...]


class DuplicationConfig(BaseModel):
    """Validated duplication config — the only file boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advisory: bool
    min_statements: int = Field(gt=0)
    scan_roots: tuple[str, ...] = Field(min_length=1)
    exclude: tuple[str, ...]
    report: str

    @classmethod
    def from_yaml(cls, path: Path) -> DuplicationConfig:
        """Load the YAML document and validate it at the boundary."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


@dataclass(frozen=True)
class FunctionSite:
    """One scanned function/method with its structural signature."""

    file: str
    owner: str
    name: str
    qualname: str
    line: int
    statements: int
    signature: str


@dataclass(frozen=True)
class DuplicateGroup:
    """One normalized signature shared across two or more files."""

    signature_id: str
    statements: int
    sites: tuple[FunctionSite, ...]


@dataclass(frozen=True)
class NameCollision:
    """One function name defined under two or more owners."""

    name: str
    owners: tuple[str, ...]
    sites: tuple[FunctionSite, ...]


@dataclass(frozen=True)
class DuplicationReport:
    """Everything one scan found."""

    files_scanned: int
    functions_scanned: int
    groups: tuple[DuplicateGroup, ...]
    collisions: tuple[NameCollision, ...]


class SignatureNormalizer:
    """Reduce AST nodes to name-free, constant-free structural tuples."""

    _SKIP_FIELDS = frozenset({"ctx", "type_comment"})

    @classmethod
    def signature(cls, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Canonical signature of one function: argument shape plus body."""
        structure = (cls._arguments(node.args), cls._body(node.body))
        return json.dumps(structure)

    @classmethod
    def statements(cls, body: list[ast.stmt]) -> list[ast.stmt]:
        """Function body without its leading docstring."""
        if body and cls._is_docstring(body[0]):
            return body[1:]
        return body

    @staticmethod
    def _is_docstring(statement: ast.stmt) -> bool:
        """Whether a statement is a bare string literal (docstring)."""
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )

    @classmethod
    def _arguments(cls, args: ast.arguments) -> tuple[int, int, int, int, int, int]:
        """Argument structure with names and annotation/constant values erased."""
        return (
            len(args.posonlyargs),
            len(args.args),
            len(args.kwonlyargs),
            int(args.vararg is not None),
            int(args.kwarg is not None),
            len(args.defaults) + sum(default is not None for default in args.kw_defaults),
        )

    @classmethod
    def _body(cls, body: list[ast.stmt]) -> tuple[Signature, ...]:
        """Normalized non-docstring statement block."""
        return tuple(cls._node(statement) for statement in cls.statements(body))

    @classmethod
    def _node(cls, node: ast.AST) -> Signature:
        """Normalize one node, erasing identifiers, constants, and docstrings."""
        if isinstance(node, FUNCTION_NODES):
            return (
                type(node).__name__,
                cls._arguments(node.args),
                cls._body(node.body),
            )
        if isinstance(node, ast.ClassDef):
            return (
                "ClassDef",
                tuple(cls._node(base) for base in node.bases),
                cls._body(node.body),
            )
        if isinstance(node, ast.Name):
            return ("Name",)
        if isinstance(node, ast.Attribute):
            return ("Attribute", cls._node(node.value))
        if isinstance(node, (ast.Constant, ast.arg)):
            return (type(node).__name__,)
        fields = tuple(
            (field, cls._value(value))
            for field, value in ast.iter_fields(node)
            if field not in cls._SKIP_FIELDS
        )
        return (type(node).__name__, fields)

    @classmethod
    def _value(cls, value: Any) -> Any:
        """Normalize an AST field value (node, list, identifier string, scalar)."""
        if isinstance(value, ast.AST):
            return cls._node(value)
        if isinstance(value, list):
            return tuple(cls._value(item) for item in value)
        if isinstance(value, str):
            return "<identifier>"
        return value


def owner_of(rel: str) -> str:
    """Canonical owner directory of a scanned file (CONSOLIDATION.md §2).

    Repo-relative files use the canonical owner mapping; an absolute path (a
    scan root outside the repo) uses its containing directory as the owner.
    """
    path = PurePosixPath(rel)
    if path.is_absolute():
        return path.parent.as_posix()
    parts = path.parts
    if parts[0] == "src":
        return "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


class DuplicationScanner:
    """Discover files, parse functions, and group their signatures."""

    def __init__(self, config: DuplicationConfig, root: Path) -> None:
        self._config = config
        self._root = root

    @classmethod
    def build(cls, config: DuplicationConfig, root: Path) -> DuplicationScanner:
        """Factory for the scanner."""
        return cls(config, root)

    def scan(self) -> DuplicationReport:
        """Scan every configured root and return the full report."""
        files = self._source_files()
        sites = [site for rel in files for site in self._file_sites(rel)]
        return DuplicationReport(
            files_scanned=len(files),
            functions_scanned=len(sites),
            groups=tuple(self._groups(sites)),
            collisions=tuple(self._collisions(sites)),
        )

    def _source_files(self) -> list[str]:
        """Resolve existing roots to sorted, non-excluded repo-relative python paths."""
        found: set[str] = set()
        scanned = 0
        for raw in self._config.scan_roots:
            root = Path(raw) if Path(raw).is_absolute() else self._root / raw
            if not root.is_dir():
                print(f"duplication: skip missing scan root: {root}")
                continue
            scanned += 1
            found.update(self._rel(path) for path in root.rglob("*.py"))
        if scanned == 0:
            raise FileNotFoundError(f"no duplication scan root exists: {self._config.scan_roots}")
        return sorted(rel for rel in found if not self._excluded(rel))

    def _rel(self, path: Path) -> str:
        """Repo-relative posix path when inside the root, absolute otherwise."""
        if path.is_relative_to(self._root):
            return path.relative_to(self._root).as_posix()
        return path.as_posix()

    def _excluded(self, rel: str) -> bool:
        """Whether a repo-relative path matches any configured exclude glob."""
        return any(PurePosixPath(rel).match(pattern) for pattern in self._config.exclude)

    def _file_sites(self, rel: str) -> list[FunctionSite]:
        """Parse one file and collect its qualifying function sites."""
        path = self._root / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (OSError, SyntaxError):
            LOGGER.exception("skipping unparseable file: %s", rel)
            return []
        return self._collect(tree, rel, owner_of(rel), ())

    def _collect(
        self, node: ast.AST, rel: str, owner: str, scope: tuple[str, ...]
    ) -> list[FunctionSite]:
        """Recursively collect function/method sites with their qualified names."""
        sites: list[FunctionSite] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FUNCTION_NODES):
                site = self._function_site(child, rel, owner, scope)
                sites.extend([site] if site is not None else [])
                sites.extend(self._collect(child, rel, owner, (*scope, child.name)))
            elif isinstance(child, ast.ClassDef):
                sites.extend(self._collect(child, rel, owner, (*scope, child.name)))
            else:
                sites.extend(self._collect(child, rel, owner, scope))
        return sites

    def _function_site(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, rel: str, owner: str, scope: tuple[str, ...]
    ) -> FunctionSite | None:
        """Build one site, or None when it is below the statement threshold."""
        statements = len(SignatureNormalizer.statements(node.body))
        if statements < self._config.min_statements:
            return None
        return FunctionSite(
            file=rel,
            owner=owner,
            name=node.name,
            qualname=".".join((*scope, node.name)),
            line=node.lineno,
            statements=statements,
            signature=SignatureNormalizer.signature(node),
        )

    @staticmethod
    def _groups(sites: list[FunctionSite]) -> list[DuplicateGroup]:
        """Group identical signatures that span different files."""
        by_signature: dict[str, list[FunctionSite]] = {}
        for site in sites:
            by_signature.setdefault(site.signature, []).append(site)
        groups: list[DuplicateGroup] = []
        for signature, members in by_signature.items():
            if len({site.file for site in members}) < 2:
                continue
            ordered = tuple(sorted(members, key=lambda site: (site.file, site.line)))
            groups.append(
                DuplicateGroup(
                    signature_id=hashlib.sha256(signature.encode()).hexdigest()[:12],
                    statements=ordered[0].statements,
                    sites=ordered,
                )
            )
        return sorted(groups, key=lambda group: (-group.statements, group.sites[0].file))

    @staticmethod
    def _collisions(sites: list[FunctionSite]) -> list[NameCollision]:
        """Group non-dunder function names that appear under several owners."""
        by_name: dict[str, list[FunctionSite]] = {}
        for site in sites:
            if not (site.name.startswith("__") and site.name.endswith("__")):
                by_name.setdefault(site.name, []).append(site)
        collisions: list[NameCollision] = []
        for name, members in by_name.items():
            owners = tuple(sorted({site.owner for site in members}))
            if len(owners) < 2:
                continue
            ordered = tuple(sorted(members, key=lambda site: (site.file, site.line)))
            collisions.append(NameCollision(name, owners, ordered))
        return sorted(collisions, key=lambda collision: collision.name)


class DuplicationReporter:
    """Write the JSON artifact and print the human summary."""

    def __init__(self, config: DuplicationConfig, root: Path) -> None:
        self._config = config
        self._root = root

    def write(self, report: DuplicationReport) -> Path:
        """Write the JSON report and return its resolved path."""
        path = Path(self._config.report)
        path = path if path.is_absolute() else self._root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._payload(report), indent=2) + "\n", encoding="utf-8")
        return path

    def announce(self, report: DuplicationReport, path: Path) -> None:
        """Print counts plus every group and collision (nothing suppressed)."""
        print(
            f"duplication: {report.files_scanned} files, {report.functions_scanned} functions "
            f">= {self._config.min_statements} statements"
        )
        self._announce_groups(report.groups)
        self._announce_collisions(report.collisions)
        print(f"report: {path}")
        print("mode: ADVISORY (exit 0)" if self._config.advisory else "mode: BLOCKING")

    @staticmethod
    def _announce_groups(groups: tuple[DuplicateGroup, ...]) -> None:
        """Print every cross-file duplicate group."""
        print(f"duplicate groups: {len(groups)} (identical normalized AST, different files)")
        for group in groups:
            print(f"- [{group.signature_id}] {group.statements} statements")
            for site in group.sites:
                print(f"    {site.file}:{site.line} {site.qualname} ({site.statements})")

    @staticmethod
    def _announce_collisions(collisions: tuple[NameCollision, ...]) -> None:
        """Print every cross-owner function-name collision."""
        print(f"name collisions: {len(collisions)} (same function name, different owners)")
        for collision in collisions:
            print(f"- {collision.name}: {', '.join(collision.owners)}")
            for site in collision.sites:
                print(f"    {site.file}:{site.line} {site.qualname}")

    def _payload(self, report: DuplicationReport) -> dict[str, Any]:
        """JSON payload for the artifact."""
        return {
            "advisory": self._config.advisory,
            "min_statements": self._config.min_statements,
            "scan_roots": list(self._config.scan_roots),
            "files_scanned": report.files_scanned,
            "functions_scanned": report.functions_scanned,
            "duplicate_groups": [
                {
                    "signature_id": group.signature_id,
                    "statements": group.statements,
                    "sites": [self._site_payload(site) for site in group.sites],
                }
                for group in report.groups
            ],
            "name_collisions": [
                {
                    "function": collision.name,
                    "owners": list(collision.owners),
                    "sites": [self._site_payload(site) for site in collision.sites],
                }
                for collision in report.collisions
            ],
        }

    @staticmethod
    def _site_payload(site: FunctionSite) -> dict[str, Any]:
        """JSON shape for one site: file:line, function name, statement count."""
        return {
            "file": site.file,
            "line": site.line,
            "function": site.qualname,
            "statements": site.statements,
        }


def default_config_path() -> Path:
    """Config path derived from this script's name (scripts/check_x.py -> configs/x.yaml)."""
    stem = Path(__file__).stem.removeprefix("check_")
    return REPO_ROOT / CONFIG_DIR / f"{stem}.yaml"


def main(argv: list[str] | None = None) -> int:
    """Run the detector; advisory exits 0, blocking exits 1 on any group."""
    parser = argparse.ArgumentParser(description="Cross-file near-duplicate function detector.")
    parser.add_argument("--config", type=Path, default=None, help="duplication config path")
    args = parser.parse_args(argv)
    config_path = args.config or default_config_path()
    config = DuplicationConfig.from_yaml(config_path)
    report = DuplicationScanner.build(config, REPO_ROOT).scan()
    reporter = DuplicationReporter(config, REPO_ROOT)
    path = reporter.write(report)
    reporter.announce(report, path)
    return 0 if config.advisory or not report.groups else 1


if __name__ == "__main__":
    raise SystemExit(main())
