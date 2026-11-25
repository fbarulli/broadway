#!/usr/bin/env python3
"""Report-only census of direct column mutations outside the plan owners.

Phase 1 of the column refactor makes ``features/plan.py`` (and the legacy
``features/pipeline.py``) the only places that may add/drop model-surface
columns. This scanner lists the remaining direct mutation sites so the
burn-down has a counted baseline. It never fails: it exits 0 with the list.

Python hits come from an AST pass (``.drop(columns=...)``, subscript
assignment on a name tracked as a DataFrame, ``.assign(...)``, ``.fillna(...)``).
The DataFrame tracker is a lightweight dataflow over annotations and
``read_*/copy/DataFrame`` producers; every hit is still printed for human
review. YAML hits are column-list keys (``include:``, ``exclude:``, ``value_columns:``,
``id_columns:``); feature-engineering configs are scanned alongside experiment
configs. Scope spans ``src/broadway/**`` and ``project/**`` Python.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PLAN_OWNER = Path("src/broadway/features/plan.py")
PIPELINE_OWNER = Path("src/broadway/features/pipeline.py")
PY_ROOTS = (Path("src/broadway"), Path("project"))
YAML_ROOTS = (Path("configs"), Path("project") / "config")
YAML_COLUMN_KEY = re.compile(r"^\s*(include|exclude|value_columns|id_columns)\s*:")

_FRAME_PRODUCERS = frozenset(
    {
        "DataFrame",
        "concat",
        "copy",
        "drop",
        "merge",
        "read_csv",
        "read_parquet",
        "reset_index",
        "select_dtypes",
    }
)


PATTERNS = (
    "drop_columns",
    "subscript_assignment",
    "assign",
    "fillna",
    "yaml_include_exclude",
)


@dataclass(frozen=True)
class Hit:
    """One reported mutation site."""

    pattern: str
    file: str
    line: int
    snippet: str


def _root_name(node: ast.AST) -> str | None:
    """Base identifier of a Name/Attribute/Subscript chain."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_frame_annotation(node: ast.AST | None) -> bool:
    """Whether an annotation names a DataFrame exactly (not a container of them)."""
    if isinstance(node, ast.Name):
        return node.id == "DataFrame"
    if isinstance(node, ast.Attribute):
        return node.attr == "DataFrame"
    return False


def _call_name(node: ast.AST) -> str | None:
    """Called name: ``f`` from ``f(...)`` or ``m`` from ``obj.m(...)``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class FrameMutationVisitor(ast.NodeVisitor):
    """Collect direct mutation sites and track names that resolve to frames."""

    def __init__(self, path: Path, display: str) -> None:
        self._display = display
        self._lines = path.read_text(encoding="utf-8").splitlines()
        self._frames: set[str] = set()
        self.hits: list[Hit] = []

    def collect(self, tree: ast.AST) -> list[Hit]:
        self.visit(tree)
        return self.hits

    def _hit(self, pattern: str, line: int) -> None:
        snippet = self._lines[line - 1].strip() if line <= len(self._lines) else ""
        self.hits.append(Hit(pattern, self._display, line, snippet))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Track frame names inside one function scope (no cross-function bleed)."""
        previous = self._frames
        self._frames = set()
        self._remember_annotated(node)
        for child in node.body:
            self.visit(child)
        self._frames = previous

    def _remember_annotated(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for arg in args:
            if _is_frame_annotation(arg.annotation):
                self._frames.add(arg.arg)

    def visit_Assign(self, node: ast.Assign) -> None:
        produces_frame = self._produces_frame(node.value)
        for target in node.targets:
            self._track_target(target, produces_frame)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        produces_frame = _is_frame_annotation(node.annotation) or self._produces_frame(
            node.value
        )
        self._track_target(node.target, produces_frame)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._track_target(node.target, False)
        self.generic_visit(node)

    def _track_target(self, target: ast.AST, produces_frame: bool) -> None:
        if isinstance(target, ast.Name):
            if produces_frame:
                self._frames.add(target.id)
            return
        if isinstance(target, ast.Subscript) and _root_name(target.value) in self._frames:
            self._hit("subscript_assignment", target.lineno)

    def _produces_frame(self, value: ast.AST | None) -> bool:
        if isinstance(value, ast.Call):
            if _call_name(value.func) in _FRAME_PRODUCERS:
                return True
            return _root_name(value.func) in self._frames
        if isinstance(value, (ast.Subscript, ast.Name, ast.Attribute)):
            return _root_name(value) in self._frames
        return False

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name == "drop" and any(keyword.arg == "columns" for keyword in node.keywords):
            self._hit("drop_columns", node.lineno)
        elif name == "assign":
            self._hit("assign", node.lineno)
        elif name == "fillna":
            self._hit("fillna", node.lineno)
        self.generic_visit(node)


class ColumnCensus:
    """Scan the configured roots and render the counted report."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._owners = {PLAN_OWNER, PIPELINE_OWNER}

    def scan(self) -> list[Hit]:
        hits = [*self._python_hits(), *self._yaml_hits()]
        return sorted(hits, key=lambda hit: (hit.pattern, hit.file, hit.line))

    def _python_hits(self) -> list[Hit]:
        hits: list[Hit] = []
        for root in PY_ROOTS:
            for path in sorted((self._root / root).rglob("*.py")):
                relative = path.relative_to(self._root)
                if "__pycache__" in path.parts or relative in self._owners:
                    continue
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except (OSError, SyntaxError):
                    continue
                hits.extend(
                    FrameMutationVisitor(path, relative.as_posix()).collect(tree)
                )
        return hits

    def _yaml_hits(self) -> list[Hit]:
        hits: list[Hit] = []
        for root in YAML_ROOTS:
            base = self._root / root
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.yaml")):
                if not (
                    any("experiment" in part for part in path.parts)
                    or path.name == "feature_engineering.yaml"
                ):
                    continue
                try:
                    yaml.safe_load(path.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError):
                    continue
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if YAML_COLUMN_KEY.match(line):
                        hits.append(
                            Hit(
                                "yaml_include_exclude",
                                path.relative_to(self._root).as_posix(),
                                line_number,
                                line.strip(),
                            )
                        )
        return hits

    def report(self, hits: list[Hit]) -> None:
        """Print every hit and the per-pattern / total counts."""
        print("column mutation census (report-only, exit 0)")
        for hit in hits:
            print(f"- [{hit.pattern}] {hit.file}:{hit.line}: {hit.snippet}")
        print()
        for pattern in PATTERNS:
            count = sum(1 for hit in hits if hit.pattern == pattern)
            print(f"{pattern}: {count}")
        print(f"total hits: {len(hits)} (burn-down baseline)")


def main() -> int:
    """Run the report against the current working tree."""
    root = Path(__file__).resolve().parent.parent
    census = ColumnCensus(root)
    census.report(census.scan())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
