from __future__ import annotations

from broadway.lineage.mermaid import to_mermaid
from broadway.lineage.models import LineageEdge, LineageGraph, LineageNode


def test_to_mermaid_renders_nodes_and_edges() -> None:
    graph = LineageGraph(
        nodes=[
            LineageNode(id="dataset:test", kind="dataset", label="test", status="produced"),
            LineageNode(
                id="baseline:test", kind="baseline", label="baseline:test", status="produced"
            ),
        ],
        edges=[LineageEdge(source="dataset:test", target="baseline:test", relation="produced_by")],
    )
    output = to_mermaid(graph)
    assert output.startswith("flowchart LR")
    assert 'dataset_test["test"]' in output
    assert "dataset_test -->|produced_by| baseline_test" in output
