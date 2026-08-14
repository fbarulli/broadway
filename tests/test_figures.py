from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.reports.results import render_results, write_results
from broadway.reports.timeline import render_timeline
from broadway.timeline import module, runners
from broadway.timeline.models import AnalysisStep, FigureRef, StepStatus
from broadway.timeline.sequence import load_walkthrough_sequence


def _cfg():
    return load_config("stats", dataset="taxi", analysis="taxi_hypothesis")


def _figure_step(**overrides) -> AnalysisStep:
    base = {
        "analysis": "taxi",
        "step_id": "describe_groups",
        "order": 1,
        "question": "Do the groups contain enough observations?",
        "status": StepStatus.COMPLETED,
        "method": "describe",
        "source": "canonical",
        "sample_name": None,
        "evidence_refs": [
            "describe.json",
            "figures/describe.png",
        ],
        "figures": [
            FigureRef(path="figures/describe.png", caption="How to read: boxplot."),
        ],
        "result_summary": {"n": 10},
        "ramification": "group sizes are adequate.",
        "decision_required": False,
        "performed_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return AnalysisStep(**base)


def test_run_describe_sets_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "TIMELINE_DIR", tmp_path / "timeline")
    cfg = _cfg()
    df = pd.DataFrame(
        {
            "Borough": [
                "Manhattan", "Manhattan", "Manhattan",
                "Brooklyn", "Brooklyn", "Brooklyn",
                "Queens", "Queens", "Queens",
            ],
            "trip_duration_minutes": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0, 30.0, 28.0, 29.0],
        }
    )
    figures_dir = tmp_path / "reports" / "figures"
    step = runners.run_describe(
        cfg.analysis, 1, "q?", df, "Borough", "Borough",
        ["Manhattan", "Brooklyn", "Queens"], "trip_duration_minutes", "x.parquet",
        None, "canonical", tmp_path / "timeline" / "taxi_hypothesis", figures_dir,
    )
    assert [f.path for f in step.figures] == [
        "figures/describe.png",
    ]
    assert (figures_dir / "describe.png").exists()
    assert "figures/describe.png" in step.evidence_refs
    for fig in step.figures:
        assert fig.caption.startswith("How to read:")


def test_run_normality_sets_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "TIMELINE_DIR", tmp_path / "timeline")
    cfg = _cfg()
    groups = {
        "Manhattan": np.random.default_rng(0).normal(10.0, 2.0, 30),
        "Brooklyn": np.random.default_rng(1).normal(12.0, 2.0, 30),
    }
    figures_dir = tmp_path / "reports" / "figures"
    step = runners.run_normality(
        cfg.analysis, 2, "q?", groups,
        tmp_path / "timeline" / "taxi_hypothesis", figures_dir, "canonical", None,
    )
    assert [f.path for f in step.figures] == ["figures/normality_qq.png"]
    for fig in step.figures:
        assert "standardized" in fig.caption
    assert "figures/normality_qq.png" in step.evidence_refs


def test_render_timeline_figures_relative_to_reports() -> None:
    seq = load_walkthrough_sequence()
    md = render_timeline("taxi", seq, [_figure_step()], [])
    assert "![How to read: boxplot.](figures/describe.png)" in md
    assert "](../figures/" not in md


def test_render_results_figures_relative_to_results() -> None:
    seq = load_walkthrough_sequence()
    pages = render_results("taxi", seq, [_figure_step()], [])
    page = pages["describe-groups.md"]
    assert "![How to read: boxplot.](../figures/describe.png)" in page


def test_write_results_figures_relative_to_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from broadway.reports import paths

    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "reports" / "results")
    seq = load_walkthrough_sequence()
    write_results("taxi", seq, [_figure_step()], [])
    page = (tmp_path / "reports" / "results" / "describe-groups.md").read_text()
    assert "![How to read: boxplot.](../figures/describe.png)" in page
