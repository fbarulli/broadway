from __future__ import annotations

from pathlib import Path

import pytest

from broadway.config import loader
from broadway.reports.index import render_dashboard
from broadway.timeline import runners
from broadway.timeline.models import AnalysisDecision, AnalysisStep, StepStatus
from broadway.timeline.sequence import (
    load_walkthrough_config,
    load_walkthrough_sequence,
)
from broadway.timeline.suggest import suggest_after, suggest_next


def _cfg() -> object:
    return load_walkthrough_config()


def _step(step_id: str, status: StepStatus, result_summary: dict, order: int = 1) -> AnalysisStep:
    return AnalysisStep(
        analysis="test",
        step_id=step_id,
        order=order,
        question="q?",
        status=status,
        method="describe",
        source="canonical",
        sample_name=None,
        evidence_refs=["evidence.json"],
        result_summary=result_summary,
        ramification="ramification",
        decision_required=False,
        performed_at="2026-01-01T00:00:00Z",
    )


def _decision(kind: str) -> AnalysisDecision:
    return AnalysisDecision(
        analysis="test",
        id=kind,
        kind=kind,
        question="q?",
        method="welch" if kind == "omnibus" else "games_howell",
        reason=["reason"],
        status="resolved",
        parents=[],
        decided_at="2026-01-01T00:00:00Z",
    )


def test_load_walkthrough_config_defaults() -> None:
    cfg = load_walkthrough_config()
    assert cfg.skew_threshold == 2.0
    assert cfg.kurtosis_threshold == 7.0
    assert cfg.shapiro_alpha == 0.05
    assert cfg.shapiro_seed == 0
    assert cfg.imbalance_ratio_threshold == 1.5
    assert cfg.significance_alpha == 0.05
    assert cfg.max_qq_groups == 12


def test_load_walkthrough_config_picks_up_custom_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    (configs_dir / "step").mkdir(parents=True)
    (configs_dir / "step" / "walkthrough.yaml").write_text(
        "skew_threshold: 5.0\n"
        "kurtosis_threshold: 7.0\n"
        "shapiro_alpha: 0.05\n"
        "shapiro_seed: 3\n"
        "imbalance_ratio_threshold: 1.5\n"
        "significance_alpha: 0.05\n"
        "max_qq_groups: 12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    assert load_walkthrough_config().skew_threshold == 5.0
    assert load_walkthrough_config().shapiro_seed == 3
    assert runners._thresholds().skew_threshold == 5.0


def test_suggest_variance_warning_favors_welch() -> None:
    step = _step(
        "variance",
        StepStatus.WARNING,
        {"statistic": 12.3, "p_value": 0.001},
        order=3,
    )
    suggestion = suggest_after("variance", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "favors considering Welch's ANOVA" in suggestion.headline
    assert "use Welch" not in suggestion.headline
    assert "use Welch" not in suggestion.command


def test_suggest_variance_warning_command_not_prefilled() -> None:
    step = _step(
        "variance",
        StepStatus.WARNING,
        {"statistic": 12.3, "p_value": 0.001},
        order=3,
    )
    suggestion = suggest_after("variance", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "--method <method>" in suggestion.command
    assert "--method welch" not in suggestion.command
    for alt in suggestion.alternatives:
        assert "--method welch" not in alt.command
        assert "--method kruskal" not in alt.command


def test_suggest_variance_ok_command_not_prefilled() -> None:
    step = _step(
        "variance",
        StepStatus.COMPLETED,
        {"statistic": 0.5, "p_value": 0.9},
        order=3,
    )
    suggestion = suggest_after("variance", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "--method <method>" in suggestion.command
    assert "--method anova" not in suggestion.command


def test_suggest_omnibus_passed_command_not_prefilled() -> None:
    step = _step(
        "omnibus",
        StepStatus.COMPLETED,
        {"method": "welch", "p_value": 0.001, "passed": True},
        order=5,
    )
    suggestion = suggest_after("omnibus", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "--kind posthoc --method <method>" in suggestion.command
    assert "games_howell" not in suggestion.command


def test_suggest_decide_posthoc_command_not_prefilled() -> None:
    suggestion = suggest_after("decide_posthoc", [], [], _cfg(), "test")
    assert suggestion is not None
    assert "--kind posthoc --method <method>" in suggestion.command
    assert "games_howell" not in suggestion.command


def test_suggest_describe_note_is_flagged() -> None:
    step = _step(
        "describe_groups",
        StepStatus.NOTE,
        {"imbalance_ratio": 3.0, "absent_groups": 0},
        order=1,
    )
    suggestion = suggest_after("describe_groups", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "imbalanced" in suggestion.headline


def test_suggest_normality_note_is_flagged() -> None:
    step = _step(
        "normality",
        StepStatus.NOTE,
        {"downtown": {"skew": 3.0, "kurtosis": 8.0, "shapiro_p": 0.001}},
        order=2,
    )
    suggestion = suggest_after("normality", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "flagged" in suggestion.headline


def test_suggest_omnibus_passed_requests_posthoc() -> None:
    step = _step(
        "omnibus",
        StepStatus.COMPLETED,
        {"method": "welch", "p_value": 0.001, "passed": True},
        order=5,
    )
    suggestion = suggest_after("omnibus", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert "--kind posthoc" in suggestion.command
    assert suggestion.headline == "Omnibus result is significant"


def test_suggest_omnibus_kruskal_rationale_includes_epsilon_squared() -> None:
    step = _step(
        "omnibus",
        StepStatus.COMPLETED,
        {"method": "kruskal", "p_value": 0.001, "passed": True, "epsilon_squared": 0.1149},
        order=5,
    )
    suggestion = suggest_after("omnibus", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert any("rank-based ε²" in line for line in suggestion.rationale)


def test_suggest_omnibus_welch_rationale_includes_eta_omega() -> None:
    step = _step(
        "omnibus",
        StepStatus.COMPLETED,
        {"method": "welch", "p_value": 0.001, "passed": True, "eta_squared": 0.5, "omega_squared": 0.4},
        order=5,
    )
    suggestion = suggest_after("omnibus", [step], [], _cfg(), "test")
    assert suggestion is not None
    assert any("eta²" in line and "omega²" in line for line in suggestion.rationale)


def test_suggest_next_frontier_before_anything() -> None:
    suggestion = suggest_next(load_walkthrough_sequence(), [], [], _cfg(), "test")
    assert suggestion is not None
    assert suggestion.step_id == "describe_groups"


def test_suggest_next_none_when_all_resolved() -> None:
    steps = [
        _step("describe_groups", StepStatus.COMPLETED, {"imbalance_ratio": 1.0, "absent_groups": 0}, 1),
        _step("normality", StepStatus.COMPLETED, {}, 2),
        _step("variance", StepStatus.COMPLETED, {"statistic": 0.5, "p_value": 0.9}, 3),
        _step("omnibus", StepStatus.COMPLETED, {"passed": True}, 5),
        _step("posthoc", StepStatus.COMPLETED, {"significant_pairs": 1}, 7),
        _step("conclusion", StepStatus.COMPLETED, {"verdict": "done"}, 8),
    ]
    decisions = [_decision("omnibus"), _decision("posthoc")]
    assert suggest_next(load_walkthrough_sequence(), steps, decisions, _cfg(), "test") is None


def test_suggest_next_frontier_decision() -> None:
    steps = [
        _step("describe_groups", StepStatus.COMPLETED, {"imbalance_ratio": 1.0, "absent_groups": 0}, 1),
        _step("normality", StepStatus.COMPLETED, {}, 2),
        _step("variance", StepStatus.COMPLETED, {"statistic": 0.5, "p_value": 0.9}, 3),
    ]
    suggestion = suggest_next(load_walkthrough_sequence(), steps, [], _cfg(), "test")
    assert suggestion is not None
    assert suggestion.step_id == "decide_omnibus"
    assert suggestion.headline == "A principal-method decision is required"


def test_render_dashboard_sections() -> None:
    steps = [_step("describe_groups", StepStatus.COMPLETED, {"imbalance_ratio": 1.0, "absent_groups": 0}, 1)]
    suggestion = suggest_next(load_walkthrough_sequence(), steps, [], _cfg(), "test")
    md = render_dashboard("test", "goal", load_walkthrough_sequence(), steps, [], suggestion)
    assert "## Progress" in md
    assert "## Next action" in md
    assert "## Navigation" in md
    assert "1 of 8 stages completed" in md
    assert "IN PROGRESS" in md


def test_render_dashboard_complete() -> None:
    steps = [
        _step("describe_groups", StepStatus.COMPLETED, {}, 1),
        _step("normality", StepStatus.COMPLETED, {}, 2),
        _step("variance", StepStatus.COMPLETED, {}, 3),
        _step("omnibus", StepStatus.COMPLETED, {}, 5),
        _step("posthoc", StepStatus.COMPLETED, {}, 7),
        _step("conclusion", StepStatus.COMPLETED, {}, 8),
    ]
    decisions = [_decision("omnibus"), _decision("posthoc")]
    md = render_dashboard("test", "goal", load_walkthrough_sequence(), steps, decisions, None)
    assert "8 of 8 stages completed" in md
    assert "COMPLETE" in md
    assert "none — analysis complete" in md
