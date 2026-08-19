"""UI surface tests — synthetic only (no taxi coupling, no server).

Covers the failure classes that have bitten the experiment UI:
raw-JS leaking out of its ``<script>`` tags, evidence-table rendering,
and HTML escaping of user/file-derived content.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# experiments_ui.py is a repo-root module (not installed); make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments_ui as ui


def test_series_page_script_is_wrapped() -> None:
    """The drag-and-drop JS must appear ONLY inside ``<script>`` tags."""
    html = ui._render_series_page(
        from_stem="01_a",
        to_stem="03_c",
        stems=["01_a", "02_b", "03_c"],
        profiles={
            "01_a": ui.ScriptProfile(produced=set(), consumed=set(), external=True),
            "02_b": ui.ScriptProfile(produced=set(), consumed=set(), external=True),
            "03_c": ui.ScriptProfile(produced=set(), consumed=set(), external=True),
        },
        focus="test",
        series=["test"],
    )
    assert html.count("<script>") == 1
    assert html.count("</script>") == 1
    inside = html.split("<script>")[1].split("</script>")[0]
    assert "() =>" in inside  # the JS lives inside the tags
    outside = html.split("<script>")[0] + html.split("</script>")[1]
    assert "() =>" not in outside  # and nowhere else (the raw-leak bug)


def test_evidence_table_escapes_and_formats(monkeypatch, tmp_path) -> None:
    """Evidence CSVs render as escaped tables; count rows show integers."""
    csv_path = tmp_path / "05_step_describe.csv"
    pd.DataFrame(
        {
            "feature": ["fare_amount", "trip_distance"],
            "mean": [10.5, 3.25],
            "count": [988077.0, 988077.0],
        },
        index=["mean", "count"],
    ).to_csv(csv_path)
    monkeypatch.setattr(ui, "_results_dir", lambda focus: tmp_path)
    html = ui._render_evidence("05_step", "test")
    assert '<table class="evidence">' in html
    assert "988,077" in html  # whole floats render as integers with commas
    assert "10.50" in html


def test_evidence_table_escapes_html_in_cells(monkeypatch, tmp_path) -> None:
    """User/file-derived cell content must be HTML-escaped."""
    csv_path = tmp_path / "05_step_describe.csv"
    pd.DataFrame({"feature": ["<script>alert(1)</script>"], "mean": [1.5]},
                 index=["weird"]).to_csv(csv_path)
    monkeypatch.setattr(ui, "_results_dir", lambda focus: tmp_path)
    html = ui._render_evidence("05_step", "test")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
