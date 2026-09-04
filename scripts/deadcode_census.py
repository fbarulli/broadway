#!/usr/bin/env python3
"""Advisory dead-code census for the broadway working tree (D34 tripwire 1).

Ratified law, adopted verbatim: the census is ADVISORY — output files
findings to the backlog; NEVER a red gate; measures suspicion, not guilt.

Measured over TRACKED ``*.py`` only (``git ls-files``):

(a) module-level defs/classes under ``src/broadway`` with zero name-resolved
    references anywhere else in the tracked corpus
    (``src/ project/ experiments/ tests/ scripts/``) — AST identifier graph,
    exact-token equality only, substring matching never;
(b) console-script entrypoints from ``pyproject`` ``[project.scripts]``,
    audited but exempt-by-design (packaging invokes them; python imports do
    not);
(c) ``configs/**`` leaf keys whose name appears in no loader call-site token
    (HEURISTIC: schema field names and ``["key"]`` literals count as readers).

Usage: ``python scripts/deadcode_census.py [--out PATH]`` (default stdout).

Every item is a hypothesis for human review, never a removal instruction.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("deadcode-census")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEF_SCAN_PREFIX = "src/broadway/"
REF_ROOTS = ("src/", "project/", "experiments/", "tests/", "scripts/")
TEST_ROOTS = ("tests/", "project/tests/")
INIT_SUFFIX = "__init__.py"

LAW_VERBATIM = (
    "census is ADVISORY — output files findings to the backlog; NEVER a red "
    "gate; measures suspicion, not guilt."
)

SCORE_LABELS = (
    (5, "HIGH"),
    (4, "ELEVATED"),
    (3, "MODERATE"),
    (0, "LOW"),
)

BUCKET_TITLES = {
    "zero_ref": "A1. Zero-reference module-level defs/classes (strongest suspicion)",
    "same_module_only": (
        "A2. Same-module-only defs (never referenced from any other tracked file)"
    ),
}

LOGGER_FORMAT = "%(levelname)s %(message)s"


@dataclass(frozen=True)
class DefSite:
    """One module-level def/class declaration eligible for the census."""

    name: str
    kind: str
    rel: str
    lineno: int
    end_lineno: int


@dataclass(frozen=True)
class ModuleInfo:
    """Reference facts extracted from one tracked Python file."""

    rel: str
    ident_lines: dict[str, frozenset[int]]
    literal_lines: dict[str, frozenset[int]]
    sites: tuple[DefSite, ...]
    guard_spans: tuple[tuple[int, int], ...]
    import_pairs: frozenset[tuple[str, str]]
    imported_modules: frozenset[str]


@dataclass
class CorpusIndex:
    """Whole-corpus exact-token maps plus the candidate def sites."""

    modules: dict[str, ModuleInfo]
    ident_files: dict[str, set[str]]
    literal_files: dict[str, set[str]]
    sites: list[DefSite]
    test_modules: set[str]
    def_name_counts: Counter[str]


@dataclass(frozen=True)
class Suspect:
    """One flagged def/class carrying its suspicion rationale."""

    site: DefSite
    bucket: str
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EntrypointRow:
    """One ``[project.scripts]`` row with its reference evidence."""

    console_name: str
    target: str
    module_dotted: str
    attr: str
    module_found: bool
    attr_defined: bool
    ref_files: tuple[str, ...]


@dataclass(frozen=True)
class ConfigKeyRow:
    """One configs/** leaf key with no corpus token match (heuristic)."""

    config_file: str
    dotted: str
    token: str


def _repo_file(rel: str) -> Path:
    """Resolve a repository-relative path without permitting traversal."""
    path = Path(rel)

    if path.is_absolute():
        raise ValueError(f"absolute repository path is forbidden: {rel!r}")

    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe repository-relative path: {rel!r}")

    resolved = (REPO_ROOT / path).resolve()

    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {rel!r}") from exc

    return resolved


def _run_git(args: list[str]) -> str:
    """Run a git command rooted at the repository and return stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )

    return proc.stdout


def tracked_python_files() -> list[str]:
    """List every tracked ``*.py`` path relative to the repository root."""
    output = _run_git(["ls-files", "-z", "--", "*.py"])
    paths = [p for p in output.split("\0") if p]
    safe: list[str] = []

    for rel in paths:
        _repo_file(rel)
        safe.append(rel)

    return sorted(safe)


def head_short_sha() -> str:
    """Return the short HEAD sha stamped into the report header."""
    sha = _run_git(["rev-parse", "--short=12", "HEAD"]).strip()

    if not sha or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        raise RuntimeError(f"git returned an invalid HEAD abbreviation: {sha!r}")

    return sha


def _names_in(node: ast.AST) -> frozenset[str]:
    """Collect every identifier token (Name ids and Attribute attrs)."""
    found: set[str] = set()

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)

    return frozenset(found)


def _literals_in(node: ast.AST) -> frozenset[str]:
    """Collect every string-literal value appearing in the subtree."""
    values = (
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    )

    return frozenset(values)


def _token_lines(node: ast.AST) -> dict[str, frozenset[int]]:
    """Map identifier tokens to occurrence lines.

    Sources: ``Name`` ids, ``Attribute`` attrs, and import alias names — an
    import such as ``from m import x as y`` references both ``x`` and ``y``.
    Dotted module imports contribute their head segment.

    Exact token equality is used; substring matching is never performed.
    """
    found: list[tuple[str, int]] = []

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.append((sub.id, sub.lineno))
        elif isinstance(sub, ast.Attribute):
            found.append((sub.attr, sub.lineno))
        elif isinstance(sub, ast.Import):
            for alias in sub.names:
                found.append((alias.name.split(".")[0], sub.lineno))
                if alias.asname:
                    found.append((alias.asname, sub.lineno))
        elif isinstance(sub, ast.ImportFrom):
            for alias in sub.names:
                if alias.name != "*":
                    found.append((alias.name, sub.lineno))
                if alias.asname:
                    found.append((alias.asname, sub.lineno))

    lines: dict[str, set[int]] = {}

    for token, lineno in found:
        lines.setdefault(token, set()).add(lineno)

    return {token: frozenset(occurs) for token, occurs in lines.items()}


def _literal_lines(node: ast.AST) -> dict[str, frozenset[int]]:
    """Map every string-literal value to its occurrence lines."""
    lines: dict[str, set[int]] = {}

    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            lines.setdefault(sub.value, set()).add(sub.lineno)

    return {value: frozenset(found) for value, found in lines.items()}


def _is_main_guard(node: ast.If) -> bool:
    """Detect an ``if __name__ == "__main__"`` style top-level guard."""
    tokens = _names_in(node.test) | _literals_in(node.test)

    return "__name__" in tokens and "__main__" in tokens


def _import_facts(
    tree: ast.Module,
) -> tuple[frozenset[tuple[str, str]], frozenset[str]]:
    """Extract absolute import pairs ``(module, name|*)`` and module paths."""
    pairs: set[tuple[str, str]] = set()
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                pairs.add((alias.name, "*"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            pairs.update((node.module, alias.name) for alias in node.names)

    return frozenset(pairs), frozenset(modules)


def _read_python(rel: str) -> str:
    """Read a tracked Python file strictly as UTF-8."""
    path = _repo_file(rel)

    return path.read_text(encoding="utf-8", errors="strict")


def parse_module(rel: str) -> ModuleInfo | None:
    """AST-parse one tracked file; syntax-broken files are logged and skipped."""
    try:
        source = _read_python(rel)
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        LOGGER.warning("skipping unparseable file %s: %s", rel, exc)
        return None
    except (OSError, UnicodeError) as exc:
        LOGGER.error("skipping unreadable file %s: %s", rel, exc)
        return None

    sites = tuple(
        DefSite(
            node.name,
            "class" if isinstance(node, ast.ClassDef) else "def",
            rel,
            node.lineno,
            node.end_lineno or node.lineno,
        )
        for node in tree.body
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
    )

    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and _is_main_guard(node)
    ]

    pairs, mods = _import_facts(tree)

    return ModuleInfo(
        rel=rel,
        ident_lines=_token_lines(tree),
        literal_lines=_literal_lines(tree),
        sites=sites,
        guard_spans=tuple(
            (node.lineno, node.end_lineno or node.lineno) for node in guards
        ),
        import_pairs=pairs,
        imported_modules=mods,
    )


def build_index(files: list[str]) -> CorpusIndex:
    """Parse the corpus once and derive every exact-token map."""
    modules: dict[str, ModuleInfo] = {}

    for rel in files:
        info = parse_module(rel)

        if info is not None:
            modules[rel] = info

    ident_files: dict[str, set[str]] = {}
    literal_files: dict[str, set[str]] = {}

    for info in modules.values():
        for token in info.ident_lines:
            ident_files.setdefault(token, set()).add(info.rel)
        for literal in info.literal_lines:
            literal_files.setdefault(literal, set()).add(info.rel)

    sites = [
        site
        for rel, info in modules.items()
        if rel.startswith(DEF_SCAN_PREFIX) and not rel.endswith(INIT_SUFFIX)
        for site in info.sites
    ]

    test_modules = {
        module
        for rel, info in modules.items()
        if rel.startswith(TEST_ROOTS)
        for module in info.imported_modules
    }

    counts: Counter[str] = Counter(
        site.name
        for info in modules.values()
        for site in info.sites
    )

    LOGGER.info(
        "corpus: %d parsed files, %d candidate def/class sites",
        len(modules),
        len(sites),
    )

    return CorpusIndex(
        modules=modules,
        ident_files=ident_files,
        literal_files=literal_files,
        sites=sites,
        test_modules=test_modules,
        def_name_counts=counts,
    )


def dotted_module_of(rel: str) -> str:
    """Map a repo-relative file path to its dotted module name."""
    if not rel.startswith("src/") or not rel.endswith(".py"):
        raise ValueError(f"not a src Python path: {rel!r}")

    stripped = rel.removesuffix(".py").removeprefix("src/")

    if stripped.endswith("/__init__"):
        stripped = stripped.removesuffix("/__init__")

    return stripped.replace("/", ".")


def _external_files(
    index: CorpusIndex,
    site: DefSite,
) -> tuple[set[str], set[str]]:
    """Files referencing the name by identifier or exact string literal."""
    ident = index.ident_files.get(site.name, set()) - {site.rel}
    literal = index.literal_files.get(site.name, set()) - {site.rel}

    return ident, literal


def _covered(
    line: int,
    spans: tuple[tuple[int, int], ...],
) -> bool:
    """Whether ``line`` falls inside any inclusive ``(start, end)`` span."""
    return any(start <= line <= end for start, end in spans)


def _same_file_context(
    module: ModuleInfo,
    site: DefSite,
) -> tuple[bool, bool]:
    """Return ``(referenced_elsewhere, referenced_only_by_main_guard)``.

    Occurrences inside the declaration's own source span do not count, so
    recursion and docstring self-mentions cannot mask a real call site.
    """
    occurrences = sorted(
        module.ident_lines.get(site.name, frozenset())
        | module.literal_lines.get(site.name, frozenset())
    )

    outside = [
        line
        for line in occurrences
        if not site.lineno <= line <= site.end_lineno
    ]

    plain = any(not _covered(line, module.guard_spans) for line in outside)
    guarded = bool(outside) and not plain

    return plain, guarded


def score_suspect(
    index: CorpusIndex,
    site: DefSite,
) -> tuple[int, tuple[str, ...]]:
    """Compute the suspicion score and its per-component rationale."""
    points = 2
    reasons = ["zero cross-corpus references outside its own body"]

    if site.name not in index.literal_files:
        points += 1
        reasons.append(
            "name absent from every string literal (no dynamic-dispatch escape)"
        )

    dotted = dotted_module_of(site.rel)

    if (
        dotted in index.test_modules
        or dotted.rpartition(".")[0] in index.test_modules
    ):
        reasons.append(
            "defining module IS imported by tests/ "
            "(blind-spot point withheld)"
        )
    else:
        points += 1
        reasons.append(
            "defining module never imported from tests/ or project/tests/"
        )

    shares = index.def_name_counts[site.name] - 1

    if shares == 0:
        points += 1
        reasons.append(
            "name unique across all tracked top-level defs "
            "(resolution unambiguous)"
        )
    else:
        reasons.append(
            f"name shared with {shares} other def(s); suspicion understated"
        )

    return points, tuple(reasons)


def _bucket_for(
    index: CorpusIndex,
    site: DefSite,
) -> str | None:
    """Assign the suspicion bucket for one candidate def/class site."""
    module = index.modules[site.rel]
    plain, guarded = _same_file_context(module, site)
    ident_ext, literal_ext = _external_files(index, site)

    if not ident_ext and not literal_ext and not plain and not guarded:
        return "zero_ref"

    if not ident_ext and not literal_ext and not plain and guarded:
        return "guard_sustained"

    if not ident_ext and literal_ext and not plain and not guarded:
        return "dynamic_string_only"

    code_ext = {
        rel for rel in ident_ext if not rel.endswith(INIT_SUFFIX)
    }
    lit_ext = {
        rel for rel in literal_ext if not rel.endswith(INIT_SUFFIX)
    }

    if (
        not code_ext
        and not lit_ext
        and (ident_ext or literal_ext)
        and not plain
        and not guarded
    ):
        return "init_sustained"

    if not ident_ext and not literal_ext and plain:
        return "same_module_only"

    return None


BUCKET_ORDER = (
    "zero_ref",
    "same_module_only",
    "dynamic_string_only",
    "init_sustained",
)


def classify_sites(
    index: CorpusIndex,
    protected: frozenset[tuple[str, str]],
) -> dict[str, list[Suspect]]:
    """Bucket every candidate site; entrypoint targets are protected out."""
    buckets: dict[str, list[Suspect]] = {
        key: [] for key in (*BUCKET_ORDER, "guard_sustained")
    }

    for site in index.sites:
        if site.name.startswith("__") or (site.rel, site.name) in protected:
            continue

        bucket = _bucket_for(index, site)

        if bucket is None:
            continue

        points, reasons = score_suspect(index, site)
        buckets[bucket].append(
            Suspect(site, bucket, points, reasons)
        )

    for group in buckets.values():
        group.sort(
            key=lambda suspect: (
                -suspect.score,
                suspect.site.rel,
                suspect.site.lineno,
                suspect.site.name,
            )
        )

    LOGGER.info(
        "classified: %s",
        ", ".join(
            f"{key}={len(group)}"
            for key, group in buckets.items()
        ),
    )

    return buckets


def load_entrypoints() -> list[tuple[str, str]]:
    """Read declared console-script entrypoints from pyproject.toml."""
    pyproject = REPO_ROOT / "pyproject.toml"

    try:
        raw = tomllib.loads(
            pyproject.read_text(encoding="utf-8", errors="strict")
        )
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"cannot read valid pyproject.toml: {exc}"
        ) from exc

    project = raw.get("project")

    if not isinstance(project, dict):
        raise RuntimeError("pyproject.toml has no [project] table")

    scripts = project.get("scripts", {})

    if scripts is None:
        scripts = {}

    if not isinstance(scripts, dict):
        raise RuntimeError(
            "pyproject.toml [project.scripts] must be a table"
        )

    rows: list[tuple[str, str]] = []

    for console_name, target in scripts.items():
        if not isinstance(console_name, str) or not console_name:
            raise RuntimeError(
                "pyproject.toml contains an invalid console-script name"
            )

        if not isinstance(target, str) or not target:
            raise RuntimeError(
                f"console-script {console_name!r} has an invalid target"
            )

        rows.append((console_name, target))

    return sorted(rows)


def _module_rels(
    index: CorpusIndex,
    dotted: str,
) -> list[str]:
    """Return repo-relative files implementing a dotted module path."""
    if not dotted or dotted.startswith(".") or ".." in dotted:
        return []

    parts = dotted.split(".")

    if any(not part or not part.isidentifier() for part in parts):
        return []

    flat = "src/" + dotted.replace(".", "/") + ".py"
    package = (
        "src/"
        + dotted.replace(".", "/")
        + "/"
        + INIT_SUFFIX
    )

    return [
        rel
        for rel in (flat, package)
        if rel in index.modules
    ]


def _entrypoint_attr_defined(
    index: CorpusIndex,
    rels: list[str],
    attr: str,
) -> bool:
    """Whether any candidate module declares a top-level attr."""
    if not attr or not attr.isidentifier():
        return False

    return any(
        site.name == attr
        for rel in rels
        for site in index.modules[rel].sites
    )


def _entrypoint_ref_files(
    index: CorpusIndex,
    dotted: str,
    attr: str,
    own_rels: set[str],
) -> tuple[str, ...]:
    """Strong import-based plus weak token-cooccurrence references."""
    strong = {
        rel
        for rel, info in index.modules.items()
        if rel not in own_rels
        and any(
            mod == dotted and name in (attr, "*")
            for mod, name in info.import_pairs
        )
    }

    leaf = dotted.rpartition(".")[2]
    weak = {
        rel
        for rel, info in index.modules.items()
        if rel not in own_rels
        and rel not in strong
        and leaf in info.ident_lines
        and attr in info.ident_lines
    }

    return tuple(sorted(strong) + sorted(weak))


def entrypoint_rows(
    index: CorpusIndex,
) -> tuple[list[EntrypointRow], frozenset[tuple[str, str]]]:
    """Audit every console-script entrypoint; return rows and protected sites."""
    rows: list[EntrypointRow] = []
    protected: set[tuple[str, str]] = set()

    for console_name, target in load_entrypoints():
        dotted, separator, attr = target.partition(":")
        if not separator or not dotted or not attr:
            rows.append(
                EntrypointRow(
                    console_name=console_name,
                    target=target,
                    module_dotted=dotted,
                    attr=attr,
                    module_found=False,
                    attr_defined=False,
                    ref_files=(),
                )
            )
            continue

        rels = _module_rels(index, dotted)
        refs = _entrypoint_ref_files(
            index,
            dotted,
            attr,
            set(rels),
        )

        rows.append(
            EntrypointRow(
                console_name=console_name,
                target=target,
                module_dotted=dotted,
                attr=attr,
                module_found=bool(rels),
                attr_defined=_entrypoint_attr_defined(
                    index,
                    rels,
                    attr,
                ),
                ref_files=refs,
            )
        )

        protected.update((rel, attr) for rel in rels)

    return rows, frozenset(protected)


def tracked_config_files() -> list[str]:
    """List every tracked YAML file under configs/."""
    output = _run_git(["ls-files", "-z", "--", "configs"])
    files = [
        path
        for path in output.split("\0")
        if path and path.endswith((".yaml", ".yml"))
    ]

    for rel in files:
        if not rel.startswith("configs/"):
            raise RuntimeError(
                f"git returned unexpected config path: {rel!r}"
            )
        _repo_file(rel)

    return sorted(files)


def leaf_paths(
    value: Any,
    prefix: str = "",
) -> Iterator[str]:
    """Yield dotted leaf paths of a parsed YAML document (lists transparent)."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, (str, int, float, bool)):
                LOGGER.warning(
                    "ignoring YAML mapping key with unsupported type %r",
                    type(key).__name__,
                )
                continue

            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield from leaf_paths(child, path)
    elif isinstance(value, list):
        for child in value:
            yield from leaf_paths(child, prefix)
    elif prefix:
        yield prefix


def _load_yaml_config(rel: str) -> Any | None:
    """Load one tracked YAML config without executing arbitrary constructors."""
    path = _repo_file(rel)

    try:
        return yaml.safe_load(
            path.read_text(encoding="utf-8", errors="strict")
        )
    except yaml.YAMLError as exc:
        LOGGER.warning("skipping unreadable config %s: %s", rel, exc)
        return None
    except (OSError, UnicodeError) as exc:
        LOGGER.warning("skipping unreadable config %s: %s", rel, exc)
        return None


def config_key_rows(index: CorpusIndex) -> list[ConfigKeyRow]:
    """Flag configs/** leaf keys never named by any tracked Python token."""
    rows: list[ConfigKeyRow] = []

    for config_rel in tracked_config_files():
        parsed = _load_yaml_config(config_rel)

        if parsed is None:
            continue

        for dotted in leaf_paths(parsed):
            token = dotted.rsplit(".", 1)[-1]

            if (
                token not in index.ident_files
                and token not in index.literal_files
            ):
                rows.append(
                    ConfigKeyRow(
                        config_file=config_rel,
                        dotted=dotted,
                        token=token,
                    )
                )

    LOGGER.info(
        "config keys without corpus token match: %d",
        len(rows),
    )

    return rows


def score_label(score: int) -> str:
    """Translate a numeric suspicion score into its band label."""
    for threshold, label in SCORE_LABELS:
        if score >= threshold:
            return label

    return "LOW"


def _suspect_line(
    suspect: Suspect,
    with_proof: bool,
) -> str:
    """Render one suspect as a markdown bullet with evidence."""
    site = suspect.site

    line = (
        f"- **{site.kind} `{site.name}`** — "
        f"`{site.rel}:{site.lineno}` — "
        f"suspicion **{suspect.score}/5 "
        f"({score_label(suspect.score)})**"
    )

    if with_proof:
        line += (
            f" — proof: `git grep -n -w '{site.name}' -- '*.py'` "
            "→ expect only def-site hits"
        )

    detail = "; ".join(suspect.reasons)

    return f"{line}\n  - rationale: {detail}"


def _bucket_section(
    buckets: dict[str, list[Suspect]],
    key: str,
    proof: bool,
) -> list[str]:
    """Render one bucket as a titled markdown section."""
    group = buckets[key]

    lines = [
        f"## {BUCKET_TITLES[key]}",
        "",
    ]

    if not group:
        lines.extend(("_none found_", ""))
        return lines

    lines.extend(
        _suspect_line(suspect, with_proof=proof)
        for suspect in group
    )
    lines.append("")

    return lines


def _entrypoint_section(
    rows: list[EntrypointRow],
) -> list[str]:
    """Render the entrypoint audit; every row is exempt-by-design."""
    lines = [
        "## B. Console-script entrypoints (EXEMPT-BY-DESIGN)",
        "",
        "Packaging console-scripts invoke these targets at install time;",
        "absence of in-tree python imports is EXPECTED, not evidence of death.",
        "",
    ]

    if not rows:
        lines.extend(("_none declared_", ""))

    for row in rows:
        state = (
            "target-found"
            if row.module_found and row.attr_defined
            else "TARGET-MISSING"
        )
        refs = (
            f" ({', '.join(row.ref_files)})"
            if row.ref_files
            else ""
        )
        lines.append(
            f"- `{row.console_name}` → `{row.target}` — {state}; "
            f"in-tree references: {len(row.ref_files)}{refs}"
        )

    lines.append("")

    return lines


def _config_section(
    rows: list[ConfigKeyRow],
) -> list[str]:
    """Render the heuristic config-key census."""
    lines = [
        "## C. configs/** leaf keys with no loader-side token match (HEURISTIC)",
        "",
        "Best-effort key-name match over identifier and string-literal tokens;",
        "a hit anywhere (schema field, `cfg['key']`, docs string) suppresses the",
        "row, so false NEGATIVES are likely and every row needs human review.",
        "",
    ]

    lines.extend(
        (
            f"- `{row.config_file}` → `{row.dotted}` — "
            f"token `{row.token}` unmatched — "
            f"proof: `git grep -n -w '{row.token}' -- '*.py'` "
            "(HEURISTIC)"
        )
        for row in rows
    )

    if not rows:
        lines.append("_none found_")

    lines.append("")

    return lines


def _exclusions_section(
    buckets: dict[str, list[Suspect]],
) -> list[str]:
    """Render the known-intentional exclusion classes."""
    lines = [
        "## Known-intentional exclusions (derived categories, not a symbol allowlist)",
        "",
        "- `__init__.py`-resident defs: excluded by policy (package re-export",
        "  surface); count excluded: see methodology.",
        "- Entrypoint targets: section B, exempt-by-design.",
    ]

    guards = buckets["guard_sustained"]
    lines.append(
        "- `__main__`-guard-sustained defs (in-file runnable demos; the qq-demo "
        "`__main__` ratification packet class):"
        + ("" if guards else " none found")
    )
    lines.extend(
        f"  - {_suspect_line(s, with_proof=False)[2:]}"
        for s in guards
    )

    dynamic = buckets["dynamic_string_only"]
    lines.append(
        "- dynamic-string-only referenced defs (exact string-literal mention is "
        "their only cross-file evidence — weak, review manually):"
        + ("" if dynamic else " none found")
    )
    lines.extend(
        f"  - {_suspect_line(s, with_proof=False)[2:]}"
        for s in dynamic
    )

    init_only = buckets["init_sustained"]
    lines.append(
        "- `__init__` re-export-sustained defs (referenced ONLY from packages'"
        "  `__init__.py`):"
        + ("" if init_only else " none found")
    )
    lines.extend(
        f"  - {_suspect_line(s, with_proof=False)[2:]}"
        for s in init_only
    )

    lines.append("")

    return lines


def _methodology_lines(
    index: CorpusIndex,
    init_def_count: int,
) -> list[str]:
    """Render methodology, corpus stats, and honest limitations."""
    roots = ", ".join(REF_ROOTS)

    return [
        "## Methodology & limitations",
        "",
        f"- Corpus: {len(index.modules)} tracked `*.py` files under {roots}",
        "  (exact-token AST graph: Name ids, Attribute attrs, string literals;",
        "  substring matching NEVER used).",
        f"- Candidate surface: {len(index.sites)} module-level defs/classes in",
        f"  `{DEF_SCAN_PREFIX}` ({init_def_count} `__init__.py`-resident defs excluded",
        "  by policy).",
        "- Class/method/nested-def level death is OUT OF SCOPE (module-level only).",
        "- Same-name defs across modules blur attribution; such suspects carry a",
        "  collision note instead of the uniqueness point.",
        "- Unparseable or unreadable files are logged to stderr and dropped from",
        "  BOTH sides of the graph (disclosed bias toward under-reporting).",
        "- Suspicion bands: HIGH ≥5, ELEVATED 4, MODERATE 3, LOW ≤2 (max 5:",
        "  zero-refs 2 + no-string-hit 1 + test-blind module 1 + unique name 1).",
        "",
    ]


def render_report(
    index: CorpusIndex,
    buckets: dict[str, list[Suspect]],
    ep_rows: list[EntrypointRow],
    cfg_rows: list[ConfigKeyRow],
    init_def_count: int,
) -> str:
    """Assemble the full advisory markdown report."""
    lines = [
        "# Dead-Code Census — ADVISORY (D34 tripwire 1)",
        "",
        f"> Ratified law (verbatim): {LAW_VERBATIM}",
        "",
        (
            f"- Generated (UTC): "
            f"{datetime.now(tz=UTC).isoformat(timespec='seconds')}"
        ),
        f"- HEAD: `{head_short_sha()}`",
        "- Disposition: file to backlog; NEVER a gate; every item below is",
        "  suspicion-not-guilt until a human rules.",
        "",
    ]

    lines += _methodology_lines(
        index,
        init_def_count,
    )

    lines += _bucket_section(
        buckets,
        "zero_ref",
        proof=True,
    )

    lines += _bucket_section(
        buckets,
        "same_module_only",
        proof=False,
    )

    lines += _entrypoint_section(ep_rows)

    lines += _config_section(cfg_rows)

    lines += _exclusions_section(buckets)

    lines.append(
        "_End of census. Advisory only — route findings through the ledger._"
    )

    return "\n".join(lines) + "\n"


def _init_def_count(index: CorpusIndex) -> int:
    """Count defs living in __init__.py files (policy-excluded surface)."""
    return sum(
        len(info.sites)
        for rel, info in index.modules.items()
        if rel.startswith(DEF_SCAN_PREFIX)
        and rel.endswith(INIT_SUFFIX)
    )


def _write_atomic(path: Path, content: str) -> None:
    """Atomically replace an output file while keeping it inside its parent."""
    if path.exists() and path.is_dir():
        raise RuntimeError(f"output path is a directory: {path}")

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            errors="strict",
            newline="\n",
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the census and emit the markdown report to stdout or --out."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGER_FORMAT,
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="ADVISORY dead-code census (D34 tripwire 1)."
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="markdown output path (default stdout)",
    )

    args = parser.parse_args(argv)

    try:
        repo_root = Path(
            _run_git(["rev-parse", "--show-toplevel"]).strip()
        ).resolve()

        if repo_root != REPO_ROOT:
            raise RuntimeError(
                f"script repository root mismatch: expected {REPO_ROOT}, "
                f"git reports {repo_root}"
            )

        files = tracked_python_files()
        index = build_index(files)

        ep_rows, protected = entrypoint_rows(index)
        buckets = classify_sites(index, protected)
        cfg_rows = config_key_rows(index)

        report = render_report(
            index,
            buckets,
            ep_rows,
            cfg_rows,
            _init_def_count(index),
        )

        if args.out is None:
            sys.stdout.write(report)
        else:
            output = args.out

            if not output.is_absolute():
                output = REPO_ROOT / output

            output = output.resolve()

            try:
                output.relative_to(REPO_ROOT)
            except ValueError as exc:
                raise RuntimeError(
                    f"--out must remain inside repository root: {args.out}"
                ) from exc

            _write_atomic(output, report)
            LOGGER.info("report written: %s", output)

        return 0

    except KeyboardInterrupt:
        LOGGER.error("interrupted")
        return 130

    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
