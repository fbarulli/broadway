from __future__ import annotations

from broadway.lineage.mermaid import to_mermaid
from broadway.lineage.models import LineageEdge, LineageGraph, LineageNode


def test_to_mermaid_renders_nodes_and_edges() -> None:
    graph = LineageGraph(
        nodes=[
            LineageNode(id="dataset:taxi", kind="dataset", label="taxi", status="produced"),
            LineageNode(
                id="baseline:taxi", kind="baseline", label="baseline:taxi", status="produced"
            ),
        ],
        edges=[LineageEdge(source="dataset:taxi", target="baseline:taxi", relation="produced_by")],
    )
    output = to_mermaid(graph)
    assert output.startswith("flowchart LR")
    assert 'dataset_taxi["taxi"]' in output
    assert "dataset_taxi -->|produced_by| baseline_taxi" in output


def test_to_mermaid_renders_sample_suffix() -> None:
    graph = LineageGraph(
        nodes=[
            LineageNode(id="dataset:taxi", kind="dataset", label="taxi", status="produced"),
            LineageNode(
                id="describe:taxi",
                kind="describe",
                label="describe:taxi",
                status="produced",
                sample_name="taxi_diagnostic",
                sample_role="diagnostic",
            ),
        ],
        edges=[],
    )
    output = to_mermaid(graph)
    assert "describe_taxi[\"describe:taxi (sample=taxi_diagnostic, role=diagnostic)\"]" in output
    assert 'dataset_taxi["taxi"]' in output
