from __future__ import annotations

from broadway.reports.mermaid import (
    render_decision_flowchart,
    render_timeline_gantt,
    sanitize_id,
    sanitize_label,
)
from broadway.timeline.models import AnalysisDecision, AnalysisStep, StepStatus
from broadway.timeline.sequence import WalkthroughSequence, WalkthroughStepConfig

HOSTILE = 'A<B>C[D]E{F}G|H#I"J\'K:L;M,N`O'


def _seq() -> WalkthroughSequence:
    return WalkthroughSequence(
        steps=[
            WalkthroughStepConfig(
                id="describe_groups", label="Describe groups", order=1,
                question="Q?", kind="evidence",
            ),
            WalkthroughStepConfig(
                id="decide_omnibus", label="Choose principal method", order=2,
                question="Q?", kind="decision",
            ),
            WalkthroughStepConfig(
                id="omnibus", label="Principal analysis", order=3,
                question="Q?", kind="analysis",
            ),
        ]
    )


def _step(**overrides) -> AnalysisStep:
    base = {
        "analysis": "test",
        "step_id": "describe_groups",
        "order": 1,
        "question": "Q?",
        "status": StepStatus.COMPLETED,
        "method": "describe",
        "source": "canonical",
        "sample_name": None,
        "evidence_refs": [],
        "result_summary": {},
        "ramification": "ok",
        "decision_required": False,
        "performed_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return AnalysisStep(**base)


def _decision() -> AnalysisDecision:
    return AnalysisDecision(
        analysis="test",
        id="omnibus",
        kind="omnibus",
        question="Q?",
        method="welch",
        reason=["non-normal"],
        status="resolved",
        parents=["describe_groups"],
        decided_at="2026-01-01T00:00:00Z",
    )


def test_gantt_sections_in_run_order_with_statuses() -> None:
    md = render_timeline_gantt("taxi", _seq(), [_step()], [])
    assert md.startswith("gantt")
    assert "dateFormat YYYY-MM-DD" in md
    first = md.index("Describe groups")
    second = md.index("Choose principal method")
    third = md.index("Principal analysis")
    assert first < second < third
    assert "completed" in md
    assert "blocked" in md


def test_flowchart_chains_steps_and_marks_decision() -> None:
    md = render_decision_flowchart("taxi", _seq(), [_step()], [_decision()])
    assert md.startswith("flowchart LR")
    assert 'describe_groups["Describe groups - completed"]' in md
    assert 'decide_omnibus{"Choose principal method - completed - welch"}' in md
    assert "describe_groups --> decide_omnibus" in md
    assert "decide_omnibus --> omnibus" in md
    assert "describe_groups -.-> decide_omnibus" in md


def test_hostile_labels_render_safe() -> None:
    assert sanitize_label(HOSTILE) == "ABCDEFGHIJKLMNO"
    assert sanitize_id("9 lives: <odd>") == "s_9_lives_odd"
    seq = WalkthroughSequence(
        steps=[
            WalkthroughStepConfig(
                id=f"weird:{HOSTILE}", label=f"Bad {HOSTILE}", order=1,
                question="Q?", kind="evidence",
            ),
        ]
    )
    gantt = render_timeline_gantt(f"taxi {HOSTILE}", seq, [], [])
    flow = render_decision_flowchart(f"taxi {HOSTILE}", seq, [], [])
    for doc in (gantt, flow):
        assert HOSTILE not in doc
        assert "<B>" not in doc
        assert "Bad ABCDEFGHIJKLMNO" in doc
    assert "weird" in flow
