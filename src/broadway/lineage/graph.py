from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from broadway.lineage.ids import node_id
from broadway.lineage.models import (
    DatasetRef,
    DatasetSlice,
    DecisionRecord,
    LineageEdge,
    LineageGraph,
    LineageNode,
    LineageRecord,
)

KIND_LABELS = {
    "dataset": "Dataset",
    "profile": "Profile",
    "etl": "ETL",
    "ingest": "Ingest",
    "join": "JoinAudit",
    "lookup_value": "LookupValueAudit",
    "analysis": "AnalysisContract",
    "baseline": "Baseline",
    "stats": "Stats",
    "describe": "Describe",
    "causal": "Causal",
    "training": "Training",
    "evaluation": "Evaluation",
    "slice": "Slice",
    "decision": "Decision",
    "features": "FeatureSpec",
}


def _read_yaml_dir(configs_dir: Path, subdir: str) -> dict[str, Any]:
    directory = configs_dir / subdir
    if not directory.is_dir():
        return {}
    result: dict[str, Any] = {}
    for path in sorted(directory.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            result[path.stem] = data
    return result


def _dedupe_edges(edges: list[LineageEdge]) -> list[LineageEdge]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[LineageEdge] = []
    for edge in sorted(edges, key=lambda e: (e.source, e.target, e.relation)):
        key = (edge.source, edge.target, edge.relation)
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def build_graph(configs_dir: Path, lineage_dir: Path) -> LineageGraph:
    nodes: dict[str, LineageNode] = {}
    edges: list[LineageEdge] = []

    for stem, data in _read_yaml_dir(configs_dir, "dataset").items():
        ref = DatasetRef(name=data["name"], path=data["path"], row_count=data.get("row_count"))
        node = LineageNode(
            id=node_id("dataset", ref.name),
            kind="dataset",
            label=ref.name,
            artifact=ref.path,
            status="produced",
        )
        nodes[node.id] = node

    for stem, data in _read_yaml_dir(configs_dir, "analysis").items():
        name = data["name"]
        path = configs_dir / "analysis" / f"{stem}.yaml"
        node = LineageNode(
            id=node_id("analysis", name),
            kind="analysis",
            label=name,
            artifact=str(path),
            status="produced",
        )
        nodes[node.id] = node

    for stem, data in _read_yaml_dir(configs_dir, "slice").items():
        slice_ = DatasetSlice.model_validate(data)
        path = configs_dir / "slice" / f"{stem}.yaml"
        node = LineageNode(
            id=node_id("slice", slice_.name),
            kind="slice",
            label=slice_.name,
            artifact=str(path),
            status="produced",
        )
        nodes[node.id] = node
        edges.append(
            LineageEdge(source=node.id, target=node_id("dataset", slice_.dataset), relation="filters")
        )

    records_path = lineage_dir / "records"
    if records_path.is_dir():
        for path in sorted(records_path.glob("*.json")):
            rec = LineageRecord.model_validate_json(path.read_text(encoding="utf-8"))
            status = "produced" if Path(rec.artifact).exists() else "ran_but_output_missing"
            node = LineageNode(
                id=rec.node_id,
                kind=rec.kind,
                label=rec.node_id,
                artifact=rec.artifact,
                status=status,
                sample_name=rec.sample_name,
                sample_role=rec.sample_role,
            )
            nodes[node.id] = node
            for parent in rec.parents:
                edges.append(LineageEdge(source=parent, target=rec.node_id, relation="produces"))

    decisions_path = lineage_dir / "decisions"
    if decisions_path.is_dir():
        for path in sorted(decisions_path.glob("*.json")):
            decision = DecisionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            node = LineageNode(
                id=node_id("decision", decision.id),
                kind="decision",
                label=decision.id,
                status="produced",
            )
            nodes[node.id] = node
            for parent in decision.parents:
                edges.append(LineageEdge(source=parent, target=node.id, relation="raises"))

    for edge in edges:
        if edge.source not in nodes:
            nodes[edge.source] = LineageNode(
                id=edge.source,
                kind=edge.source.split(":")[0],
                label=edge.source,
                artifact=None,
                status="referenced_not_found",
            )

    return LineageGraph(
        nodes=sorted(nodes.values(), key=lambda n: n.id),
        edges=_dedupe_edges(edges),
    )


class LineageScopeError(ValueError):
    pass


DATA_PREP = {"ingest", "join", "etl", "lookup_value", "profile"}


def _forward_closure(edges: list[LineageEdge], start: str) -> set[str]:
    selected = {start}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.source in selected and edge.target not in selected and edge.relation == "produces":
                selected.add(edge.target)
                changed = True
    return selected


def scope_graph(
    graph: LineageGraph, analysis: str | None = None, dataset: str | None = None
) -> LineageGraph:
    nodes = {n.id: n for n in graph.nodes}
    edges = graph.edges

    if analysis is not None:
        analysis_id = node_id("analysis", analysis)
        if analysis_id not in nodes:
            raise LineageScopeError(f"analysis {analysis!r} not found")
        selected = _forward_closure(edges, analysis_id)
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge.target in selected and edge.source not in selected:
                    selected.add(edge.source)
                    changed = True
        for edge in edges:
            if edge.relation == "raises" and edge.source in selected:
                selected.add(edge.target)
    else:
        dataset_id = node_id("dataset", dataset)
        if dataset_id not in nodes:
            raise LineageScopeError(f"dataset {dataset!r} not found")
        selected = {dataset_id}
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if (
                    edge.source in selected
                    and edge.relation == "produces"
                    and edge.target not in selected
                    and edge.target.split(":")[0] in DATA_PREP
                ):
                    selected.add(edge.target)
                    changed = True
        for edge in edges:
            if edge.relation == "filters" and edge.target == dataset_id:
                selected.add(edge.source)
        for edge in edges:
            if edge.relation == "raises" and edge.source in selected:
                selected.add(edge.target)

    if dataset is not None:
        if node_id("dataset", dataset) not in selected:
            raise LineageScopeError(
                f"analysis {analysis!r} is not derived from dataset {dataset!r}"
            )

    return LineageGraph(
        nodes=[n for n in graph.nodes if n.id in selected],
        edges=[e for e in edges if e.source in selected and e.target in selected],
    )


def load_decisions(lineage_dir: Path) -> list[DecisionRecord]:
    directory = lineage_dir / "decisions"
    if not directory.is_dir():
        return []
    decisions: list[DecisionRecord] = []
    for path in sorted(directory.glob("*.json")):
        decisions.append(DecisionRecord.model_validate_json(path.read_text(encoding="utf-8")))
    return decisions
