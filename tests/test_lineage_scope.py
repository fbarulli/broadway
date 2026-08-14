from __future__ import annotations

from broadway.lineage.graph import LineageScopeError, scope_graph
from broadway.lineage.models import LineageEdge, LineageGraph, LineageNode


def _node(node_id: str) -> LineageNode:
    return LineageNode(
        id=node_id,
        kind=node_id.split(":")[0],
        label=node_id,
        artifact=None,
        status="produced",
    )


def _graph() -> LineageGraph:
    nodes = [
        _node("dataset:taxi"),
        _node("ingest:taxi"),
        _node("etl:taxi"),
        _node("describe:taxi_hypothesis"),
        _node("stats:taxi_hypothesis"),
        _node("analysis:taxi_hypothesis"),
        _node("analysis:taxi"),
        _node("training:taxi"),
        _node("slice:airport"),
        _node("decision:keep"),
    ]
    edges = [
        LineageEdge(source="dataset:taxi", target="ingest:taxi", relation="produces"),
        LineageEdge(source="ingest:taxi", target="etl:taxi", relation="produces"),
        LineageEdge(source="etl:taxi", target="describe:taxi_hypothesis", relation="produces"),
        LineageEdge(source="etl:taxi", target="stats:taxi_hypothesis", relation="produces"),
        LineageEdge(
            source="analysis:taxi_hypothesis", target="describe:taxi_hypothesis", relation="produces"
        ),
        LineageEdge(
            source="analysis:taxi_hypothesis", target="stats:taxi_hypothesis", relation="produces"
        ),
        LineageEdge(source="analysis:taxi", target="training:taxi", relation="produces"),
        LineageEdge(source="etl:taxi", target="training:taxi", relation="produces"),
        LineageEdge(source="slice:airport", target="dataset:taxi", relation="filters"),
        LineageEdge(source="slice:airport", target="decision:keep", relation="raises"),
    ]
    return LineageGraph(nodes=nodes, edges=edges)


def test_scope_graph_by_analysis_and_dataset() -> None:
    scoped = scope_graph(_graph(), analysis="taxi_hypothesis", dataset="taxi")
    node_ids = {n.id for n in scoped.nodes}

    for expected in {
        "dataset:taxi",
        "etl:taxi",
        "describe:taxi_hypothesis",
        "stats:taxi_hypothesis",
        "analysis:taxi_hypothesis",
        "slice:airport",
        "decision:keep",
    }:
        assert expected in node_ids

    assert "training:taxi" not in node_ids
    assert "analysis:taxi" not in node_ids


def test_scope_graph_by_dataset_only() -> None:
    scoped = scope_graph(_graph(), dataset="taxi")
    node_ids = {n.id for n in scoped.nodes}

    for expected in {"dataset:taxi", "ingest:taxi", "etl:taxi", "slice:airport", "decision:keep"}:
        assert expected in node_ids

    for absent in {
        "analysis:taxi",
        "analysis:taxi_hypothesis",
        "describe:taxi_hypothesis",
        "stats:taxi_hypothesis",
        "training:taxi",
    }:
        assert absent not in node_ids


def test_scope_graph_wrong_dataset_raises() -> None:
    try:
        scope_graph(_graph(), analysis="taxi_hypothesis", dataset="taxi_other")
    except LineageScopeError:
        pass
    else:
        raise AssertionError("expected LineageScopeError")
