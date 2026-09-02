"""Reconcile the graphify code graph against gates.yaml owners (surface gate).

graphify (``graphify extract . --code-only --no-cluster``) indexes every
callable in the governed code-bearing surfaces (``src/`` + ``project/``,
non-test) into ``graphify-out/graph.json``. This gate closes the "what is not
mapped" gap that used to require manual grep: it proves that every governed
file carrying callable symbols is OWNED by at least one gate, and warns about
callables that are not individually named.

Contract (ratified as the graphify surface gate, GATE-INFRA-149):
  * HARD FAIL — a governed file (under ``src/`` or ``project/``, excluding
    ``project/tests/``) that carries callable symbols but is referenced by NO
    gate's ``owner:`` path is UNMAPPED. Registration is derived from the
    ``owner:`` field ONLY (never inputs/outputs/transforms) — the registry
    convention is that a gate OWNS the entry point it names, while the
    FILE-level tripwire (probe-e) already covers inputs/outputs mention.
    An owner path matches a file exactly, as a directory prefix, as a
    ``X/__init__.py`` package marker (covers the whole ``X/`` package), or as
    a glob.
  * WARN (never a failure) — a callable inside an OWNED file whose name is not
    individually named by that file's owners. The registry names entry points,
    not every helper, so this is informational.
  * KNOWN_UNMAPPED is the dated baseline of files that are genuinely unowned
    at HEAD (recorded by STATE-20260902-010): each is a candidate for a future
    GATE-SURF/GATE-INFRA registration. A governed file with callables that is
    neither owner-referenced nor in KNOWN_UNMAPPED fails the gate — so any NEW
    unowned surface trips it, while the known baseline stays green.

Regenerate the graph (deterministic, byte-identical, no LLM):
    graphify extract . --code-only --no-cluster
Scope is pinned by ``.graphifyignore`` (src/ + project/, non-test). The
committed ``graphify-out/graph.json`` is the diffable baseline.

Usage: python scripts/check_graphify_surfaces.py [--graph PATH] [--gates PATH]
Exit 0 green / 1 red.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_GRAPH = REPO / "graphify-out" / "graph.json"
DEFAULT_GATES = REPO / "agents" / "ledger" / "gates.yaml"

# Governed code-bearing surfaces: a file under one of these roots that carries
# callable symbols must be OWNED (owner:-referenced) by at least one gate.
GOVERNED_ROOTS = ("src/", "project/")

# Under a governed root, tests self-describe through validated_by pins and are
# not OWNED surfaces; they are out of scope for owner reconciliation.
EXCLUDED_PREFIXES = ("project/tests/",)

# Callable label shapes emitted by graphify's AST extractor: ``name()`` for
# functions/methods (methods carry a leading ``.``), ``Name``/``_Name`` for
# classes. File-level nodes (``foo.py``, ``pkg/__init__.py``) and the
# pyproject ``type: package`` node never match these.
FUNC_LABEL = re.compile(r"^\.*_?[A-Za-z_][A-Za-z0-9_]*\(\)$")
CLASS_LABEL = re.compile(r"^_?[A-Z][A-Za-z0-9_]*$")

# Identifier tokens in an owner's post-colon text (function/class names and the
# prose that names them); used only for the WARN tier.
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Known-unmapped baseline (STATE-20260902-010): governed files that carry
# callables but are referenced by NO gate owner at HEAD 1bed31e. Each entry is
# a future GATE-SURF/GATE-INFRA registration candidate. A file NOT in this
# baseline and NOT owner-referenced fails the gate. Grouped by package.
KNOWN_UNMAPPED: tuple[str, ...] = (
    # project/ composition + plumbing
    "project/__init__.py",
    "project/paths.py",
    # analysis
    "src/broadway/analysis/contracts.py",
    # baseline (module.py IS owned)
    "src/broadway/baseline/causal.py",
    "src/broadway/baseline/contracts.py",
    "src/broadway/baseline/hypothesis.py",
    "src/broadway/baseline/improvement.py",
    "src/broadway/baseline/prediction.py",
    # causal (module.py, hte.py, sequential.py ARE owned)
    "src/broadway/causal/analysis.py",
    "src/broadway/causal/assignment.py",
    "src/broadway/causal/contracts.py",
    "src/broadway/causal/design.py",
    "src/broadway/causal/multiple.py",
    # cleaning
    "src/broadway/cleaning/models.py",
    # config
    "src/broadway/config/viz.py",
    # contracts (pandera.py IS owned)
    "src/broadway/contracts/checks.py",
    "src/broadway/contracts/module.py",
    "src/broadway/contracts/selectors.py",
    # data
    "src/broadway/data/splitter.py",
    # discover
    "src/broadway/discover/profile.py",
    # evaluate (module.py, metrics.py, promotion.py ARE owned)
    "src/broadway/evaluate/comparison.py",
    "src/broadway/evaluate/contracts.py",
    "src/broadway/evaluate/explain.py",
    "src/broadway/evaluate/feature_selection.py",
    "src/broadway/evaluate/validation.py",
    # features (builders.py, generic.py, module.py, pipeline.py ARE owned)
    "src/broadway/features/contracts.py",
    "src/broadway/features/schema.py",
    "src/broadway/features/transformers.py",
    # formatting / viz / timing / trace / pipeline (root helpers)
    "src/broadway/formatting.py",
    "src/broadway/lineage/graph.py",
    "src/broadway/lineage/ids.py",
    "src/broadway/lineage/mermaid.py",
    "src/broadway/lineage/sample.py",
    "src/broadway/onboard/models.py",
    "src/broadway/pipeline.py",
    "src/broadway/samples/models.py",
    # stats (anova.py, assumptions.py, describe.py, groups.py, guards.py,
    # module.py, post_hoc.py, robust.py ARE owned)
    "src/broadway/stats/baseline.py",
    "src/broadway/stats/diagnostic_models.py",
    "src/broadway/stats/diagnostics.py",
    "src/broadway/stats/effect_size.py",
    "src/broadway/stats/plan.py",
    "src/broadway/stats/regression.py",
    "src/broadway/stats/time_series.py",
    # timeline (decide.py, models.py, module.py, runners.py, sequence.py,
    # walkthrough.py ARE owned)
    "src/broadway/timeline/evidence.py",
    "src/broadway/timeline/suggest.py",
    "src/broadway/timing.py",
    "src/broadway/trace.py",
    # training (module.py, trainer.py, mlflow_utils.py ARE owned)
    "src/broadway/training/contracts.py",
    "src/broadway/training/optuna_worker.py",
    "src/broadway/training/models/base.py",
    "src/broadway/training/models/lightgbm.py",
    "src/broadway/training/models/linear.py",
    "src/broadway/training/models/pyfunc_wrapper.py",
    "src/broadway/training/models/random_forest.py",
    "src/broadway/training/models/registry.py",
    "src/broadway/training/models/xgboost.py",
    "src/broadway/viz.py",
)


def _is_governed(rel: str) -> bool:
    """A repo-relative file under a governed code-bearing surface (non-test)."""
    if not rel.startswith(GOVERNED_ROOTS):
        return False
    return not any(rel.startswith(p) for p in EXCLUDED_PREFIXES)


def callable_name(label: str) -> str | None:
    """Bare symbol name for a callable label, else None (file/library node)."""
    if FUNC_LABEL.match(label):
        name = label[:-2].lstrip(".")
        return name.rsplit(".", 1)[-1] if "." in name else name
    if CLASS_LABEL.match(label):
        return label
    return None


def extract_callables(nodes: list[dict]) -> dict[str, set[str]]:
    """{repo-relative file: {callable names}} for governed code files only."""
    callables: dict[str, set[str]] = {}
    for node in nodes:
        if node.get("file_type") != "code":
            continue
        src = str(node.get("source_file") or "")
        if not src.endswith(".py") or not node.get("source_location"):
            continue
        if not _is_governed(src):
            continue
        name = callable_name(str(node.get("label") or ""))
        if name is None:
            continue
        callables.setdefault(src, set()).add(name)
    return callables


def _owner_segments(owner: str) -> list[str]:
    """Split an owner string on ``->`` and `` + <path>`` boundaries."""
    segments: list[str] = []
    for part in re.split(r"\s+->\s+", owner):
        segments.extend(re.split(r"\s+\+\s+(?=[\w.]+/)", part))
    return segments


def owner_paths(owner: str) -> list[str]:
    """Repo-relative path tokens referenced by an owner string."""
    paths: list[str] = []
    for chunk in _owner_segments(owner):
        head = re.match(r"[^\s:]+", chunk)
        if not head:
            continue
        token = head.group(0)
        if "/" in token or re.search(r"\.[A-Za-z0-9]+$", token):
            paths.append(token)
    return paths


def _normalize(path: str) -> str:
    path = path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def owner_symbols(owner: str) -> set[str]:
    """Identifier tokens after the first path:line prefix (WARN tier only)."""
    _, sep, rest = owner.partition(":")
    if not sep:
        return set()
    rest = re.sub(r"^\d+(?:-\d+)?\s*", "", rest)
    return set(IDENT.findall(rest))


def build_owner_coverage(
    gates: list[dict],
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    """(owned_files, owned_dirs, symbols_by_file) from gate owner fields.

    ``owned_files`` holds exact file paths; ``owned_dirs`` holds directory
    prefixes (including ``X/`` for an ``X/__init__.py`` package owner).
    ``symbols_by_file`` maps an exactly-owned file to the symbol tokens its
    owners name, for the WARN tier.
    """
    files: set[str] = set()
    dirs: set[str] = set()
    symbols: dict[str, set[str]] = {}
    for gate in gates:
        owner = str(gate.get("owner") or "")
        for path in owner_paths(owner):
            norm = _normalize(path)
            if not norm:
                continue
            if norm.endswith(".py") or "." in norm.rsplit("/", 1)[-1]:
                files.add(norm)
                symbols.setdefault(norm, set()).update(owner_symbols(owner))
                # package marker: X/__init__.py owns the whole X/ package
                if norm.endswith("/__init__.py"):
                    dirs.add(norm.rsplit("/__init__.py", 1)[0])
            else:
                dirs.add(norm)
    return files, dirs, symbols


def _is_covered(rel: str, files: set[str], dirs: set[str]) -> bool:
    if rel in files:
        return True
    for d in dirs:
        if rel.startswith(d + "/"):
            return True
    for pat in files | dirs:
        if any(ch in pat for ch in "*?[") and fnmatch.fnmatch(rel, pat):
            return True
    return False


def reconcile(
    nodes: list[dict],
    gates: list[dict],
    *,
    allowlist: tuple[str, ...] = KNOWN_UNMAPPED,
) -> tuple[dict[str, set[str]], list[str], list[str], dict[str, set[str]]]:
    """Reconcile graph nodes against gates.yaml owners.

    Returns ``(callables, mapped, unmapped_fail, unnamed)`` where:
      * ``callables`` — governed file -> callable names;
      * ``mapped`` — governed files with callables that ARE owner-referenced
        (sorted);
      * ``unmapped_fail`` — governed files with callables referenced by no
        owner AND not in ``allowlist`` (sorted) — these fail the gate;
      * ``unnamed`` — for each mapped file, the callables not individually
        named by that file's owners (WARN tier, non-failing).
    """
    callables = extract_callables(nodes)
    files, dirs, symbols = build_owner_coverage(gates)
    allow = set(allowlist)
    mapped: list[str] = []
    unmapped_fail: list[str] = []
    unmapped_baseline: list[str] = []
    for rel in sorted(callables):
        if _is_covered(rel, files, dirs):
            mapped.append(rel)
        elif rel in allow:
            unmapped_baseline.append(rel)
        else:
            unmapped_fail.append(rel)

    unnamed: dict[str, set[str]] = {}
    for rel in mapped:
        named = symbols.get(rel, set())
        missing = callables[rel] - named
        if missing:
            unnamed[rel] = missing
    return callables, mapped, unmapped_fail, unnamed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH,
                        help="path to graphify-out/graph.json (default: %(default)s)")
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES,
                        help="path to agents/ledger/gates.yaml (default: %(default)s)")
    args = parser.parse_args(argv)

    import json

    try:
        import yaml  # project dependency (same as render_gates.py / probe-e)
    except ModuleNotFoundError:
        print("graphify surfaces: PyYAML required — run via scripts/uv.sh", file=sys.stderr)
        return 1

    if not args.graph.is_file():
        print(
            f"graphify surfaces: missing graph {args.graph} — regenerate with "
            "`graphify extract . --code-only --no-cluster`",
            file=sys.stderr,
        )
        return 1
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        print(f"graphify surfaces: malformed graph (no nodes list) in {args.graph}", file=sys.stderr)
        return 1
    if not args.gates.is_file():
        print(f"graphify surfaces: missing registry {args.gates}", file=sys.stderr)
        return 1
    gates = yaml.safe_load(args.gates.read_text(encoding="utf-8"))["gates"]

    callables, mapped, unmapped_fail, unnamed = reconcile(nodes, gates)
    total_files = len(callables)
    total_callables = sum(len(v) for v in callables.values())
    unmapped_total = total_files - len(mapped)

    print(
        f"graphify surfaces: {total_files} governed file(s), "
        f"{total_callables} callable(s) · {len(mapped)} mapped · "
        f"{unmapped_total} unmapped (baseline) · {len(unmapped_fail)} NEW"
    )

    # WARN tier: functions in OWNED files not individually named. Non-failing.
    warn_files = len(unnamed)
    warn_total = sum(len(v) for v in unnamed.values())
    if warn_files:
        print(f"WARN: {warn_total} callable(s) in {warn_files} owned file(s) "
              f"not individually named in an owner (entry points only, non-failing)")
        for rel in sorted(unnamed):
            names = sorted(unnamed[rel])
            shown = ", ".join(names[:10])
            more = f" +{len(names) - 10}" if len(names) > 10 else ""
            print(f"  {rel}: {shown}{more}")

    # Full unmapped baseline (the key deliverable — what is not mapped at HEAD).
    baseline = sorted(rel for rel in callables
                      if rel not in set(mapped) and rel not in set(unmapped_fail))
    if baseline:
        print(f"UNMAPPED baseline ({len(baseline)} file(s), KNOWN_UNMAPPED):")
        for rel in baseline:
            names = sorted(callables[rel])
            print(f"  {rel}: {', '.join(names[:8])}"
                  + (f" +{len(names) - 8}" if len(names) > 8 else ""))

    if unmapped_fail:
        print(
            f"graphify surfaces RED: {len(unmapped_fail)} governed file(s) with "
            f"callables and no gate owner — add a gates.yaml owner row or a "
            f"KNOWN_UNMAPPED baseline entry"
        )
        for rel in unmapped_fail:
            names = sorted(callables[rel])
            print(f"  {rel}: {', '.join(names)}")
        return 1

    print("graphify surfaces OK: every governed callable file is owner-mapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
