from __future__ import annotations

from pathlib import Path

import pytest

from broadway.reports import paths
from broadway.reports.results import (
    humanize_float,
    humanize_pvalue,
    render_results,
    write_results,
)
from broadway.timeline.models import AnalysisDecision, AnalysisStep, StepStatus
from broadway.timeline.sequence import load_walkthrough_sequence


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
        "evidence_refs": ["describe.json"],
        "result_summary": {
            "total_n": 123456,
            "imbalance_ratio": 1.23456789,
            "absent_groups": 0,
            "n_total": 123456,
            "n_used": 123455,
            "n_excluded": 1,
            "exclusion_reason": "null target",
        },
        "ramification": "group sizes are adequate.",
        "decision_required": False,
        "performed_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return AnalysisStep(**base)


def _full_steps() -> list[AnalysisStep]:
    return [
        _step(step_id="describe_groups", order=1),
        _step(
            step_id="normality",
            order=2,
            method="check_normality",
            result_summary={
                "Manhattan": {"skew": 0.123456, "kurtosis": 0.1, "shapiro_p": 0.0004}
            },
        ),
        _step(
            step_id="variance",
            order=3,
            method="levene",
            result_summary={"statistic": 1.23456789, "p_value": 0.4},
        ),
        _step(
            step_id="omnibus",
            order=5,
            method="welch",
            result_summary={
                "method": "welch",
                "statistic": 12.345678901,
                "p_value": 0.0004,
                "passed": True,
                "eta_squared": 0.123456,
                "omega_squared": 0.098765,
            },
        ),
    ]


def test_humanize_pvalue_floors_at_small() -> None:
    assert humanize_pvalue(0.0004) == "< 0.001"
    assert humanize_pvalue(0.001) == "0.001"
    assert humanize_pvalue(0.123456) == "0.123"


def test_humanize_float_three_sig_figs() -> None:
    assert humanize_float(1.23456789) == "1.23"
    assert humanize_float(12.345678901) == "12.3"


def test_render_results_index_and_pages() -> None:
    seq = load_walkthrough_sequence()
    pages = render_results("taxi", seq, _full_steps(), [])
    assert set(pages) == {
        "index.md",
        "describe_groups.md",
        "normality.md",
        "variance.md",
        "omnibus.md",
    }

    idx = pages["index.md"]
    assert "awaiting decision" in idx
    assert "[Describe groups](describe_groups.md)" in idx
    assert "[Principal analysis](omnibus.md)" in idx
    assert "Choose principal method" in idx

    describe_page = pages["describe_groups.md"]
    assert "# Describe groups" in describe_page
    assert "## Question" in describe_page
    assert "## What was run" in describe_page
    assert "## What it found" in describe_page
    assert "## Why it matters" in describe_page
    assert "## Attrition" in describe_page
    assert "null target" in describe_page

    normality_page = pages["normality.md"]
    assert "< 0.001" in normality_page

    omnibus_page = pages["omnibus.md"]
    assert "## Effect size" in omnibus_page
    assert "eta²" in omnibus_page
    assert "omega²" in omnibus_page


def test_render_results_kruskal_not_computed() -> None:
    seq = load_walkthrough_sequence()
    steps = [
        _step(
            step_id="omnibus",
            order=5,
            method="kruskal",
            result_summary={
                "method": "kruskal",
                "statistic": 5.0,
                "p_value": 0.02,
                "passed": True,
                "effect_size": "not_computed",
            },
        )
    ]
    pages = render_results("taxi", seq, steps, [])
    assert "deliberately not computed" in pages["omnibus.md"]


def test_rendered_output_plain_text_no_glyphs_or_literals() -> None:
    seq = load_walkthrough_sequence()
    pages = render_results("taxi", seq, _full_steps(), [])
    for content in pages.values():
        assert "{" not in content
        assert "}" not in content
        assert "✓" not in content
        assert "12.345678901" not in content
        assert "1.23456789" not in content


def test_write_results_orphan_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir)
    results_dir.mkdir(parents=True)
    (results_dir / "describe_groups.md").write_text("old", encoding="utf-8")
    (results_dir / "ghost.md").write_text("orphan", encoding="utf-8")
    (results_dir / "index.md").write_text("old index", encoding="utf-8")

    seq = load_walkthrough_sequence()
    write_results("taxi", seq, [_step(step_id="describe_groups", order=1)], [])

    assert (results_dir / "index.md").exists()
    assert (results_dir / "describe_groups.md").exists()
    assert not (results_dir / "ghost.md").exists()
    assert (results_dir / "describe_groups.md").read_text() != "old"


def test_write_results_failed_step_has_no_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir)
    results_dir.mkdir(parents=True)
    (results_dir / "describe_groups.md").write_text("old", encoding="utf-8")

    seq = load_walkthrough_sequence()
    steps = [_step(step_id="describe_groups", order=1, status=StepStatus.FAILED)]
    write_results("taxi", seq, steps, [])

    assert not (results_dir / "describe_groups.md").exists()
    idx = (results_dir / "index.md").read_text()
    assert "failed" in idx


def test_render_results_decision_awaiting() -> None:
    seq = load_walkthrough_sequence()
    steps = [
        _step(step_id="describe_groups", order=1),
        _step(step_id="normality", order=2, method="check_normality", result_summary={}),
        _step(step_id="variance", order=3, method="levene", result_summary={}),
    ]
    idx = render_results("taxi", seq, steps, [])["index.md"]
    assert "| Choose principal method | awaiting decision |" in idx
