"""Canvas map specifications — the standardized way to create diagrams.

Diagrams (``diagrams/*.tldr``) are DATA. The code that builds them lives
HERE: every map is produced by a builder below from live project surfaces
(experiment series, result files, graphify traversals) — never hand-written
JSON. Workflow: builder -> spec dict -> ``emit_tldr`` (skill generator) ->
``diagrams/<name>.tldr`` -> dashboard ``/canvas/<name>`` for annotation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGRAMS_DIR = REPO_ROOT / "diagrams"
SKILL_GENERATOR = (
    Path.home() / ".claude" / "skills" / "tldraw-json-to-tldr"
    / "scripts" / "gen-tldr.mjs"
)


def new_spec(title: str) -> dict[str, object]:
    """Blank canvas spec: title + empty nodes/edges."""
    return {"title": title, "nodes": [], "edges": []}


def add_node(spec: dict[str, object], node_id: str, label: str, group: str) -> None:
    """Append one node (mutates the spec)."""
    nodes = spec["nodes"]
    assert isinstance(nodes, list)
    nodes.append({"id": node_id, "label": label, "group": group})


def add_edge(spec: dict[str, object], source: str, target: str, label: str) -> None:
    """Append one directed edge (mutates the spec)."""
    edges = spec["edges"]
    assert isinstance(edges, list)
    edges.append({"from": source, "to": target, "label": label})


def experiment_series_spec(series: dict[str, list[str]]) -> dict[str, object]:
    """Two-track map: experiment steps -> the result files they produce.

    ``series`` maps a series/track name to its ordered step stems; steps
    producing into ``results/<stem>/`` get an edge to the results node.
    """
    spec = new_spec("Results producers")
    add_node(spec, "dashboard", "dashboard :8000 (results + maps)", "serve")
    for track, stems in series.items():
        track_id = f"track-{track}"
        add_node(spec, track_id, f"{track} ({len(stems)} steps)", "series")
        for stem in stems:
            step_id = f"{track}/{stem}"
            add_node(spec, step_id, stem, "series")
            add_edge(spec, track_id, step_id, "runs")
            add_edge(spec, step_id, "dashboard", "plots")
    return spec


def blast_radius_spec(symbol: str, callers: list[tuple[str, str]]) -> dict[str, object]:
    """Fan-out map: ``symbol`` center + one node per caller (file:line)."""
    spec = new_spec(f"{symbol} blast radius")
    add_node(spec, "hub", symbol, "contract")
    for name, location in callers:
        node_id = f"caller-{name}"
        add_node(spec, node_id, f"{name} ({location})", "consumers")
        add_edge(spec, "hub", node_id, "calls")
    add_edge(spec, "hub", "dashboard", "gates")
    add_node(spec, "dashboard", "gates + probes", "evidence")
    return spec


def write_spec(spec: dict[str, object], path: Path) -> None:
    """Write a spec dict as JSON."""
    path.write_text(json.dumps(spec, indent=1), encoding="utf-8")


def emit_tldr(spec_path: Path, out_path: Path) -> None:
    """Render a spec file to .tldr via the tldraw skill generator."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is required to render .tldr files")
    if not SKILL_GENERATOR.is_file():
        raise RuntimeError(f"tldraw skill generator not found: {SKILL_GENERATOR}")
    subprocess.run(
        [node, str(SKILL_GENERATOR), "--in", str(spec_path), "--out", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
