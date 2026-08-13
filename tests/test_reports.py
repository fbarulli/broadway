from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from broadway.config import loader
from broadway.reports import describe as describe_reports
from broadway.reports import paths
from broadway.reports.index import render_index
from broadway.reports.markdown import render_result
from broadway.reports.sequence import StatsSequence, load_stats_sequence
from broadway.stats.describe import GroupStat, GroupSummary


def _summary() -> GroupSummary:
    return GroupSummary(
        group_column="Borough",
        source_group_column="pickup_borough",
        target="trip_duration_minutes",
        total_n=5,
        source_path="x.parquet",
        sample_name="diagnostic",
        sample_role="diagnostic",
        groups={
            "A": GroupStat(n=2, mean=1.5, std=0.5),
            "B": GroupStat(n=0, mean=None, std=None),
        },
        absent_groups=["B"],
        imbalance_ratio=2.0,
        proportions={"A": 1.0, "B": 0.0},
        warnings=["small sample"],
    )


def test_render_result_structure() -> None:
    md = render_result("Title", [("Alpha", "body a"), ("Beta", "body b")])
    lines = md.split("\n")
    assert lines[0] == "# Title"
    assert "## Alpha" in lines
    assert "## Beta" in lines
    assert "body a" in md
    assert "body b" in md


def test_stats_sequence_parse() -> None:
    seq = StatsSequence(steps=["describe", "anova"])
    assert seq.steps == ["describe", "anova"]


def test_stats_sequence_bad_yaml_raises() -> None:
    with pytest.raises(ValidationError):
        StatsSequence(steps="describe")
    with pytest.raises(ValidationError):
        StatsSequence(steps=["describe"], bogus=1)


def test_load_stats_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs_dir = tmp_path / "configs"
    (configs_dir / "flow").mkdir(parents=True)
    (configs_dir / "flow" / "stats_sequence.yaml").write_text(
        "steps:\n  - describe\n  - normality\n", encoding="utf-8"
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    assert load_stats_sequence().steps == ["describe", "normality"]


def test_render_describe_headings_and_figures() -> None:
    md = describe_reports.render(_summary())
    for heading in ("Question", "Sample", "Groups", "Imbalance", "Warnings", "Figures"):
        assert f"## {heading}" in md
    assert "figures/describe_boxplot.png" in md
    assert "figures/describe_group_sizes.png" in md


def test_describe_headline_one_liner() -> None:
    text = describe_reports.headline(_summary())
    assert isinstance(text, str)
    assert text
    assert "\n" not in text


def test_render_index_all_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    md = render_index("q?", tmp_path / "stats")
    assert "## Question" in md
    assert "## Results" in md
    assert "[ ]" in md
    assert "[x]" not in md
    assert "## Next test\n\ndescribe\n" in md
    assert "none yet" in md


def test_render_index_describe_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "describe.md").write_text("# describe", encoding="utf-8")
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir)
    md = render_index("q?", tmp_path / "stats")
    assert "[x]" in md
    assert "results/describe.md" in md
    assert "## Next test\n\nnormality\n" in md
