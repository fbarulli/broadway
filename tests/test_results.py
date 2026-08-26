from __future__ import annotations

from pathlib import Path

import pytest

from broadway.formatting import humanize_float as shared_humanize_float
from broadway.formatting import humanize_pvalue as shared_humanize_pvalue
from broadway.reports import paths
from broadway.reports.results import (
    humanize_float,
    humanize_pvalue,
    humanize_summary,
    posthoc_headline,
    posthoc_pair_rows,
    render_results,
    slugify,
    statistic_label,
    write_results,
)
from broadway.reports.timeline import render_timeline
from broadway.stats.diagnostic_models import DiagnosticResult
from broadway.timeline.models import AnalysisStep, StepStatus
from broadway.timeline.sequence import load_walkthrough_sequence


def _step(**overrides) -> AnalysisStep:
    base = {
        "analysis": "test",
        "step_id": "describe_groups",
        "order": 1,
        "question": "Do the groups contain enough observations?",
        "status": StepStatus.COMPLETED,
        "method": "describe",
        "source": "canonical",
        "sample_name": None,
        "evidence_refs": ["describe.json"],
        "result_summary": {
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
                "downtown": {"skew": 0.123456, "kurtosis": 0.1, "shapiro_p": 0.0004}
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


def _posthoc_step() -> AnalysisStep:
    return _step(
        step_id="posthoc",
        order=7,
        method="games_howell",
        result_summary={
            "method": "games_howell",
            "pairs": 3,
            "significant_pairs": 2,
            "significant_pair_details": [
                {
                    "a": "downtown",
                    "b": "suburbs",
                    "p_value": 0.0004,
                    "cohens_d": 1.23456789,
                    "hedges_g": 1.24,
                    "effect_size_note": "large",
                },
                {
                    "a": "downtown",
                    "b": "suburbs",
                    "p_value": 0.01,
                    "cohens_d": 0.5,
                    "hedges_g": 0.49,
                    "effect_size_note": "medium",
                },
            ],
        },
        ramification=(
            "Games-Howell found 2 significant pairwise difference(s) at alpha=0.05."
        ),
    )


def test_humanize_pvalue_floors_at_small() -> None:
    assert humanize_pvalue(0.0004) == "< 0.001"
    assert humanize_pvalue(0.001) == "0.001"
    assert humanize_pvalue(0.123456) == "0.123"


def test_humanize_float_three_sig_figs() -> None:
    assert humanize_float(1.23456789) == "1.23"
    assert humanize_float(12.345678901) == "12.3"


def test_humanize_helpers_live_in_shared_module() -> None:
    assert shared_humanize_float is humanize_float
    assert shared_humanize_pvalue is humanize_pvalue
    assert shared_humanize_float(4199.7167447099055) == "4.2e+03"
    assert shared_humanize_pvalue(0.0) == "< 0.001"


def test_slugify_humanizes_step_labels() -> None:
    assert slugify("Describe groups") == "describe-groups"
    assert slugify("Principal analysis") == "principal-analysis"
    assert slugify("Post-hoc comparisons") == "post-hoc-comparisons"
    assert slugify("Normality diagnostics") == "normality-diagnostics"
    assert slugify("Variance homogeneity") == "variance-homogeneity"
    assert slugify("Conclusion") == "conclusion"


def test_statistic_label_per_step() -> None:
    assert statistic_label("variance", {"statistic": 4.2}) == "Levene statistic"
    assert statistic_label("omnibus", {"method": "welch"}) == "F"
    assert statistic_label("omnibus", {"method": "anova"}) == "F"
    assert statistic_label("omnibus", {"method": "kruskal"}) == "H"
    assert statistic_label("omnibus", {}) == "statistic"


def test_humanize_summary_labels_statistic() -> None:
    variance_lines = humanize_summary(
        {"statistic": 4199.7167447099055, "p_value": 0.0}, "variance"
    )
    assert "Levene statistic: 4.2e+03" in variance_lines

    omnibus_lines = humanize_summary(
        {"method": "welch", "statistic": 7000.0, "p_value": 0.0001}, "omnibus"
    )
    assert "F: 7e+03" in omnibus_lines

    kruskal_lines = humanize_summary(
        {"method": "kruskal", "statistic": 12.5, "p_value": 0.01}, "omnibus"
    )
    assert "H: 12.5" in kruskal_lines


def test_render_results_conclusion_shows_effect_size_caveat() -> None:
    seq = load_walkthrough_sequence()
    conclusion = _step(
        step_id="conclusion",
        order=8,
        method="conclusion",
        result_summary={
            "verdict": "group means differ",
            "principal_method": "welch",
            "p_value": 0.0004,
            "significant_pairs": 2,
            "eta_squared": 0.123456,
            "omega_squared": 0.098765,
        },
    )
    page = render_results("test", seq, [conclusion], [])["conclusion.md"]
    assert "## Effect size" in page
    assert "omega²" in page
    assert "the more conservative estimate" in page


def test_render_results_index_and_pages() -> None:
    seq = load_walkthrough_sequence()
    pages = render_results("test", seq, _full_steps(), [])
    assert set(pages) == {
        "index.md",
        "describe-groups.md",
        "normality-diagnostics.md",
        "variance-homogeneity.md",
        "principal-analysis.md",
    }

    idx = pages["index.md"]
    assert "awaiting decision" in idx
    assert "[Describe groups](describe-groups.md)" in idx
    assert "[Principal analysis](principal-analysis.md)" in idx
    assert "Choose principal method" in idx

    describe_page = pages["describe-groups.md"]
    assert "# Describe groups" in describe_page
    assert "## Question" in describe_page
    assert "## What was run" in describe_page
    assert "## What it found" in describe_page
    assert "## Why it matters" in describe_page
    assert "## Attrition" in describe_page
    assert "null target" in describe_page

    normality_page = pages["normality-diagnostics.md"]
    assert "< 0.001" in normality_page

    omnibus_page = pages["principal-analysis.md"]
    assert "## Effect size" in omnibus_page
    assert "eta²" in omnibus_page
    assert "omega²" in omnibus_page


def test_render_results_kruskal_epsilon_squared() -> None:
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
                "epsilon_squared": 0.1149,
            },
        )
    ]
    pages = render_results("test", seq, steps, [])
    assert "rank-based ε²" in pages["principal-analysis.md"]
    assert "proportion of variance in ranks" in pages["principal-analysis.md"]


def test_render_results_kruskal_not_computed_backward_compat() -> None:
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
    pages = render_results("test", seq, steps, [])
    assert "deliberately not computed" in pages["principal-analysis.md"]


def test_rendered_output_plain_text_no_glyphs_or_literals() -> None:
    seq = load_walkthrough_sequence()
    pages = render_results("test", seq, _full_steps(), [])
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
    (results_dir / "describe-groups.md").write_text("old", encoding="utf-8")
    (results_dir / "ghost.md").write_text("orphan", encoding="utf-8")
    (results_dir / "index.md").write_text("old index", encoding="utf-8")

    seq = load_walkthrough_sequence()
    write_results("test", seq, [_step(step_id="describe_groups", order=1)], [])

    assert (results_dir / "index.md").exists()
    assert (results_dir / "describe-groups.md").exists()
    assert not (results_dir / "ghost.md").exists()
    assert (results_dir / "describe-groups.md").read_text() != "old"


def test_write_results_failed_step_has_no_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir)
    results_dir.mkdir(parents=True)
    (results_dir / "describe-groups.md").write_text("old", encoding="utf-8")

    seq = load_walkthrough_sequence()
    steps = [_step(step_id="describe_groups", order=1, status=StepStatus.FAILED)]
    write_results("test", seq, steps, [])

    assert not (results_dir / "describe-groups.md").exists()
    idx = (results_dir / "index.md").read_text()
    assert "failed" in idx


def test_render_results_decision_awaiting() -> None:
    seq = load_walkthrough_sequence()
    steps = [
        _step(step_id="describe_groups", order=1),
        _step(step_id="normality", order=2, method="check_normality", result_summary={}),
        _step(step_id="variance", order=3, method="levene", result_summary={}),
    ]
    idx = render_results("test", seq, steps, [])["index.md"]
    assert "| Choose principal method | awaiting decision |" in idx


def test_posthoc_headline_and_rows() -> None:
    summary = _posthoc_step().result_summary
    assert posthoc_headline(summary) == "2 of 3 pairs significant"
    assert posthoc_pair_rows(summary) == summary["significant_pair_details"]
    assert posthoc_pair_rows({"method": "games_howell"}) == []


def test_posthoc_step_page_renders_significant_pairs() -> None:
    seq = load_walkthrough_sequence()
    pages = render_results("test", seq, [_posthoc_step()], [])
    page = pages["post-hoc-comparisons.md"]
    assert "## Significant pairs" in page
    assert "2 of 3 pairs significant" in page
    assert "| Pair | p | Cohen's d | Hedges' g | Note |" in page
    assert "| downtown vs suburbs | < 0.001 | 1.23 | 1.24 | large |" in page
    assert "| downtown vs suburbs | 0.010 | 0.5 | 0.49 | medium |" in page
    assert "significant_pair_details" not in page
    assert "[{" not in page


def test_step_page_renders_diagnostic_evidence_section() -> None:
    seq = load_walkthrough_sequence()
    step = _step(
        diagnostic=DiagnosticResult(
            question="Is the mean relationship correctly specified?",
            evidence=[
                "residual-vs-fitted plot persisted at figures/residuals_vs_fitted.png"
            ],
            ramification="systematic residual structure suggests misspecification",
            warnings=["small sample"],
        )
    )
    page = render_results("test", seq, [step], [])["describe-groups.md"]
    assert "## Evidence" in page
    assert "- residual-vs-fitted plot persisted at figures/residuals_vs_fitted.png" in page
    assert "## Warnings" in page
    assert "- small sample" in page


def test_step_page_without_diagnostic_omits_evidence_section() -> None:
    seq = load_walkthrough_sequence()
    step = _step()
    page = render_results("test", seq, [step], [])["describe-groups.md"]
    assert "## Evidence" not in page
    assert "## Warnings" not in page


def test_posthoc_step_page_zero_significant_pairs() -> None:
    seq = load_walkthrough_sequence()
    step = _step(
        step_id="posthoc",
        order=7,
        method="games_howell",
        result_summary={
            "method": "games_howell",
            "pairs": 3,
            "significant_pairs": 0,
            "significant_pair_details": [],
        },
    )
    page = render_results("test", seq, [step], [])["post-hoc-comparisons.md"]
    assert "## Significant pairs" in page
    assert "0 of 3 pairs significant" in page
    assert "none" in page
    assert "| Pair |" not in page


def test_humanize_summary_does_not_leak_pair_list() -> None:
    lines = humanize_summary(_posthoc_step().result_summary)
    text = "\n".join(lines)
    assert "significant_pair_details" not in text
    assert "[" not in text
    assert "pairs: 3" in text
    assert "significant_pairs: 2" in text


def test_render_timeline_posthoc_bullet() -> None:
    seq = load_walkthrough_sequence()
    timeline = render_timeline("test", seq, [_posthoc_step()], [])
    assert "2 of 3 pairs significant:" in timeline
    assert "downtown vs suburbs: p < 0.001, Cohen's d 1.23, Hedges' g 1.24" in timeline
    assert "downtown vs suburbs: p 0.010, Cohen's d 0.5, Hedges' g 0.49" in timeline
