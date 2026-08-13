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
            )
            nodes[node.id] = node
            for parent in rec.parents:
                edges.append(LineageEdge(source=parent, target=rec.node_id, relation="produced_by"))

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

    return LineageGraph(
        nodes=sorted(nodes.values(), key=lambda n: n.id),
        edges=_dedupe_edges(edges),
    )


def load_decisions(lineage_dir: Path) -> list[DecisionRecord]:
    directory = lineage_dir / "decisions"
    if not directory.is_dir():
        return []
    decisions: list[DecisionRecord] = []
    for path in sorted(directory.glob("*.json")):
        decisions.append(DecisionRecord.model_validate_json(path.read_text(encoding="utf-8")))
    return decisions
