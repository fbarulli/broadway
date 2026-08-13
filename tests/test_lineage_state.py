from __future__ import annotations

from broadway.lineage.models import DecisionRecord, LineageGraph, LineageNode
from broadway.lineage.state import current_state


def _graph(kinds: set[str]) -> LineageGraph:
    return LineageGraph(
        nodes=[LineageNode(id=k, kind=k, label=k, status="produced") for k in sorted(kinds)],
        edges=[],
    )


def _decisions() -> list[DecisionRecord]:
    return [
        DecisionRecord(id="a", question="q", outcome="o", reason=["r"], status="open", parents=[]),
        DecisionRecord(
            id="b", question="q", outcome="o", reason=["r"], status="resolved", parents=[]
        ),
    ]


def test_prediction_state() -> None:
    state = current_state(_graph({"profile", "baseline"}), "prediction", "g", _decisions())
    assert state.stage == "baseline"
    assert state.not_yet_produced == ["training", "evaluation"]
    assert state.open_decisions == ["a"]
    assert state.resolved_decisions == ["b"]


def test_hypothesis_state() -> None:
    state = current_state(_graph({"profile", "baseline"}), "hypothesis", "g", [])
    assert state.stage == "baseline"
    assert state.not_yet_produced == ["stats"]


def test_non_lineage_kinds_never_listed() -> None:
    kinds = {"profile", "baseline", "etl", "eda", "contracts", "features"}
    state = current_state(_graph(kinds), "prediction", "g", [])
    for kind in ("etl", "eda", "contracts", "features"):
        assert kind not in state.not_yet_produced
