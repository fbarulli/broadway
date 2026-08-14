from __future__ import annotations

import broadway.timeline.module as module
from broadway.reports.timeline import render_timeline
from broadway.timeline.models import (
    Alternative,
    AnalysisDecision,
    AnalysisStep,
    FigureRef,
    StepStatus,
    Suggestion,
)
from broadway.timeline.sequence import (
    WalkthroughSequence,
    WalkthroughStepConfig,
    load_walkthrough_sequence,
)


def _step(**overrides) -> AnalysisStep:
    base = {
        "analysis": "taxi",
        "step_id": "describe_groups",
        "order": 1,
        "question": "Do the groups contain enough observations?",
        "status": StepStatus.COMPLETED,
        "method": "describe",
        "source": "canonical",
        "sample_name": None,
        "evidence_refs": ["profile.json"],
        "result_summary": {"n": 10},
        "ramification": "groups are comparable",
        "decision_required": False,
        "performed_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return AnalysisStep(**base)


def _decision(**overrides) -> AnalysisDecision:
    base = {
        "analysis": "taxi",
        "id": "omnibus",
        "kind": "omnibus",
        "question": "Which principal method should answer the question?",
        "method": "welch",
        "reason": ["non-normal"],
        "status": "resolved",
        "parents": ["normality"],
        "decided_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return AnalysisDecision(**base)


def _seq(steps: list[dict]) -> WalkthroughSequence:
    configs = []
    for s in steps:
        s = dict(s)
        s.setdefault("label", s.get("id", "step").replace("_", " ").title())
        configs.append(WalkthroughStepConfig(**s))
    return WalkthroughSequence(steps=configs)


def test_step_roundtrip() -> None:
    step = _step()
    assert AnalysisStep.model_validate_json(step.model_dump_json()) == step


def test_figure_ref_roundtrip() -> None:
    fig = FigureRef(path="figures/describe_boxplot.png", caption="How to read: ...")
    assert FigureRef.model_validate_json(fig.model_dump_json()) == fig


def test_step_without_figures_defaults_to_empty() -> None:
    step = _step()
    assert step.figures == []
    data = step.model_dump()
    data.pop("figures", None)
    parsed = AnalysisStep.model_validate(data)
    assert parsed.figures == []


def test_step_figures_roundtrip() -> None:
    step = _step(
        figures=[
            FigureRef(
                path="figures/describe_boxplot.png",
                caption="How to read: each box spans the interquartile range.",
            )
        ]
    )
    assert AnalysisStep.model_validate_json(step.model_dump_json()).figures == step.figures


def test_decision_roundtrip() -> None:
    decision = _decision()
    assert AnalysisDecision.model_validate_json(decision.model_dump_json()) == decision


def test_suggestion_roundtrip() -> None:
    suggestion = Suggestion(
        step_id="omnibus",
        headline="use welch",
        rationale=["non-normal"],
        command="run welch",
        alternatives=[
            Alternative(
                label="kruskal",
                command="run kruskal",
                intent="alternative",
                rationale="rank-based",
            )
        ],
    )
    assert Suggestion.model_validate_json(suggestion.model_dump_json()) == suggestion


def test_load_walkthrough_sequence() -> None:
    seq = load_walkthrough_sequence()
    assert [s.order for s in seq.steps] == list(range(1, 9))
    assert [s.id for s in seq.steps] == [
        "describe_groups",
        "normality",
        "variance",
        "decide_omnibus",
        "omnibus",
        "decide_posthoc",
        "posthoc",
        "conclusion",
    ]
    assert [s.kind for s in seq.steps] == [
        "evidence",
        "evidence",
        "evidence",
        "decision",
        "analysis",
        "decision",
        "analysis",
        "analysis",
    ]
    assert [s.label for s in seq.steps] == [
        "Describe groups",
        "Normality diagnostics",
        "Variance homogeneity",
        "Choose principal method",
        "Principal analysis",
        "Choose post-hoc method",
        "Post-hoc comparisons",
        "Conclusion",
    ]


def test_step_status_has_note_and_failed() -> None:
    assert StepStatus.NOTE.value == "note"
    assert StepStatus.FAILED.value == "failed"
    assert StepStatus.WARNING.value == "warning"
    assert StepStatus.COMPLETED.value == "completed"


def test_persistence_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "TIMELINE_DIR", tmp_path / "timeline")
    module.save_step(_step(step_id="normality", order=2))
    module.save_step(_step(step_id="describe_groups", order=1))

    assert module.load_step("taxi", "normality") == _step(step_id="normality", order=2)
    assert module.load_step("taxi", "missing") is None

    loaded = module.load_steps("taxi")
    assert [s.step_id for s in loaded] == ["describe_groups", "normality"]


def test_decision_persistence_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "TIMELINE_DIR", tmp_path / "timeline")
    module.save_decision(_decision())
    assert module.load_decision("taxi", "omnibus") == _decision()
    assert module.load_decision("taxi", "missing") is None
    assert [d.id for d in module.load_decisions("taxi")] == ["omnibus"]


def test_render_completed_step() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    md = render_timeline("taxi", seq, [_step()], [])
    assert "| completed |" in md


def test_render_warning_step() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    md = render_timeline("taxi", seq, [_step(status=StepStatus.WARNING)], [])
    assert "| warning |" in md


def test_render_note_step() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    md = render_timeline("taxi", seq, [_step(status=StepStatus.NOTE)], [])
    assert "| completed with note |" in md


def test_render_failed_step() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    md = render_timeline("taxi", seq, [_step(status=StepStatus.FAILED)], [])
    assert "| failed |" in md


def test_render_decision_resolved() -> None:
    seq = _seq([{"id": "decide_omnibus", "order": 1, "question": "Q?", "kind": "decision"}])
    md = render_timeline("taxi", seq, [], [_decision(id="omnibus", kind="omnibus", method="welch")])
    assert "| completed |" in md


def test_render_decision_required() -> None:
    seq = _seq([{"id": "decide_omnibus", "order": 1, "question": "Q?", "kind": "decision"}])
    md = render_timeline("taxi", seq, [], [])
    assert "| awaiting decision |" in md


def test_render_blocked_after_unresolved_decision() -> None:
    seq = _seq(
        [
            {"id": "decide_omnibus", "order": 1, "question": "Q?", "kind": "decision"},
            {"id": "omnibus", "order": 2, "question": "Q?", "kind": "analysis"},
        ]
    )
    md = render_timeline("taxi", seq, [], [])
    assert "| blocked |" in md


def test_render_not_started_evidence() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    md = render_timeline("taxi", seq, [], [])
    assert "| not started |" in md


def test_render_empty_no_crash() -> None:
    seq = load_walkthrough_sequence()
    md = render_timeline("taxi", seq, [], [])
    assert "| not started |" in md
    assert "| blocked |" in md
    assert "awaiting decision" not in md


def test_render_decide_posthoc_blocked_when_omnibus_incomplete() -> None:
    seq = _seq(
        [
            {"id": "omnibus", "order": 1, "question": "Q?", "kind": "analysis"},
            {"id": "decide_posthoc", "order": 2, "question": "Q?", "kind": "decision"},
        ]
    )
    md = render_timeline("taxi", seq, [], [])
    assert "| 2 | Decide Posthoc | Q? | blocked |" in md
    assert "awaiting decision" not in md


def test_render_decide_posthoc_required_when_omnibus_completed() -> None:
    seq = _seq(
        [
            {"id": "omnibus", "order": 1, "question": "Q?", "kind": "analysis"},
            {"id": "decide_posthoc", "order": 2, "question": "Q?", "kind": "decision"},
        ]
    )
    omnibus_step = _step(step_id="omnibus", order=1)
    md = render_timeline("taxi", seq, [omnibus_step], [])
    assert "| 2 | Decide Posthoc | Q? | awaiting decision |" in md


def test_render_decide_omnibus_required_only_when_prereqs_completed() -> None:
    seq = _seq(
        [
            {"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"},
            {"id": "normality", "order": 2, "question": "Q?", "kind": "evidence"},
            {"id": "variance", "order": 3, "question": "Q?", "kind": "evidence"},
            {"id": "decide_omnibus", "order": 4, "question": "Q?", "kind": "decision"},
        ]
    )
    completed = [
        _step(step_id="describe_groups", order=1),
        _step(step_id="normality", order=2),
        _step(step_id="variance", order=3),
    ]
    md = render_timeline("taxi", seq, completed, [])
    assert "| 4 | Decide Omnibus | Q? | awaiting decision |" in md

    partial = [_step(step_id="describe_groups", order=1)]
    md = render_timeline("taxi", seq, partial, [])
    assert "| 4 | Decide Omnibus | Q? | blocked |" in md
    assert "awaiting decision" not in md


def test_render_includes_details_for_completed() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    md = render_timeline("taxi", seq, [_step()], [])
    assert "## Describe Groups" in md
    assert "ramification: groups are comparable" in md
    assert "evidence_refs" not in md
    assert "n: 10" in md


def test_render_timeline_figure_link_has_no_parent_prefix() -> None:
    seq = _seq([{"id": "describe_groups", "order": 1, "question": "Q?", "kind": "evidence"}])
    caption = "How to read: each box spans the interquartile range."
    step = _step(
        figures=[FigureRef(path="figures/describe_boxplot.png", caption=caption)]
    )
    md = render_timeline("taxi", seq, [step], [])
    assert f"![{caption}](figures/describe_boxplot.png)" in md
    assert "](../figures/describe_boxplot.png)" not in md
    assert "(../figures/describe_boxplot.png)" not in md


def test_render_timeline_machine_json_refs_are_plain_text() -> None:
    seq = _seq([{"id": "normality", "order": 1, "question": "Q?", "kind": "evidence"}])
    step = _step(
        step_id="normality",
        evidence_refs=["normality.json", "figures/normality_A.png"],
        figures=[FigureRef(path="figures/normality_A.png", caption="How to read: points hug the diagonal.")],
    )
    md = render_timeline("taxi", seq, [step], [])
    assert "evidence_refs" not in md
    assert "normality.json" not in md
    assert "[normality_A.png]" not in md
    assert "![How to read: points hug the diagonal.](figures/normality_A.png)" in md


def test_render_timeline_labels_omnibus_statistic() -> None:
    seq = _seq([{"id": "omnibus", "order": 1, "question": "Q?", "kind": "analysis"}])
    step = _step(
        step_id="omnibus",
        method="welch",
        result_summary={
            "method": "welch",
            "statistic": 7000.0,
            "p_value": 0.0004,
            "passed": True,
            "eta_squared": 0.9,
            "omega_squared": 0.1,
        },
    )
    md = render_timeline("taxi", seq, [step], [])
    assert "F: 7e+03" in md
    assert "omega²" in md
    assert "the more conservative estimate" in md


def test_render_timeline_conclusion_shows_effect_size() -> None:
    seq = _seq([{"id": "conclusion", "order": 1, "question": "Q?", "kind": "analysis"}])
    step = _step(
        step_id="conclusion",
        method="conclusion",
        result_summary={
            "verdict": "group means differ",
            "principal_method": "welch",
            "p_value": 0.0004,
            "significant_pairs": 2,
            "eta_squared": 0.9,
            "omega_squared": 0.1,
        },
    )
    md = render_timeline("taxi", seq, [step], [])
    assert "omega²" in md
    assert "the more conservative estimate" in md
