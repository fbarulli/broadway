from __future__ import annotations

from broadway.lineage.models import DecisionRecord, LineageGraph, RunState

LINEAGE_STEPS = {
    "prediction": ["profile", "etl", "baseline", "features", "training", "evaluation"],
    "hypothesis": ["profile", "etl", "baseline", "stats"],
    "causal": ["profile", "etl", "baseline", "causal"],
}


def current_state(
    graph: LineageGraph, mode: str, goal: str, decisions: list[DecisionRecord]
) -> RunState:
    present = {node.kind for node in graph.nodes}
    ordered = LINEAGE_STEPS.get(mode, [])
    produced = [k for k in ordered if k in present]
    stage = produced[-1] if produced else None
    not_yet = [k for k in ordered if k not in present]
    open_decisions = sorted(d.id for d in decisions if d.status == "open")
    resolved_decisions = sorted(d.id for d in decisions if d.status == "resolved")
    return RunState(
        goal=goal,
        stage=stage,
        open_decisions=open_decisions,
        resolved_decisions=resolved_decisions,
        not_yet_produced=not_yet,
    )
