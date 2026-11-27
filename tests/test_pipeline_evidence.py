"""Public behavior of pipeline steps on the experiment dashboard.

A synthetic checkout with one script series plus a published pipeline tree: the
dashboard must list ``pipeline`` after the script series, show one
``experiment | question | status | artifacts`` row per pipeline step (linked to its
step page), render that page with the decision, figure, table, report and code, and
skip a malformed record with one logged traceback instead of failing the page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from broadway.reports import experiments_dashboard as ui
from broadway.reports.decision import Decision


def _decision(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": "Does the metric clear its threshold?",
        "metric": "association_<x>",
        "value": 0.94,
        "interval_low": 0.93,
        "interval_high": 0.95,
        "reference": 0.0,
        "delta": 0.94,
        "threshold": 0.05,
        "null_threshold": 0.0089,
        "direction": "max",
        "verdict": "act",
        "action": "keep the feature",
        "owner": "eda",
        "sample_size": 50000,
        "unit": None,
        "min_sample_size": 100,
        "evidence": {
            "baseline": "no-effect shuffle of the target at this n",
            "delta": None,
            "segments": [{"segment": "weekday", "value": 0.94, "sample_size": 30000}],
            "sensitivity": {"half_sample": 0.93},
            "falsifier": "the interval falls at or below the threshold",
            "provenance": {"dataset": "synthetic", "seed": "42"},
        },
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pipeline_series_lists_steps_and_renders_step_pages(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    root = tmp_path / "experiments"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "01_step.py").write_text('"""01: A script step."""', encoding="utf-8")

    step = root / "results" / "pipeline" / "etl"
    _write(
        step / "etl_decision.json",
        Decision.model_validate(_decision()).model_dump(mode="json"),
    )
    (step / "etl_effects.png").write_bytes(b"png")
    (step / "etl_effects.csv").write_text("stat,value\ncount,1\n", encoding="utf-8")
    (step / "etl.md").write_text("# ETL report\n", encoding="utf-8")

    bad = root / "results" / "pipeline" / "bad"
    bad.mkdir(parents=True)
    (bad / "bad_decision.json").write_text("[]", encoding="utf-8")

    evaluate = root / "results" / "pipeline" / "evaluate"
    for stem in ("champion_promotion", "tuned_promotion"):
        _write(
            evaluate / f"{stem}_decision.json",
            Decision.model_validate(_decision(metric=stem)).model_dump(mode="json"),
        )

    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)

    assert ui.list_series() == ["alpha", "pipeline"]
    rows = ui.census_rows("pipeline")
    assert all(set(row) == {"experiment", "question", "status", "artifacts"} for row in rows)
    assert rows[0] == {
        "experiment": "bad",
        "question": "",
        "status": "no record",
        "artifacts": 1,
    }
    assert rows[1] == {
        "experiment": "etl",
        "question": "Does the metric clear its threshold?",
        "status": "act",
        "artifacts": 4,
    }
    assert rows[2] == {
        "experiment": "evaluate",
        "question": "Does the metric clear its threshold?",
        "status": "act",
        "artifacts": 2,
    }

    with caplog.at_level(logging.ERROR):
        dashboard = ui.index("pipeline")
        ui.index("pipeline")
        step_page = ui.experiment_page("etl", "pipeline")
        bad_page = ui.experiment_page("bad", "pipeline")
        multi_record_page = ui.experiment_page("evaluate", "pipeline")

    assert dashboard.status_code == 200
    dashboard_html = dashboard.body.decode()
    assert "<tr><th>experiment</th><th>question</th><th>status</th><th>artifacts</th></tr>" in (
        dashboard_html
    )
    assert (
        '<tr><td><a href="/experiments/etl?focus=pipeline">etl</a></td>'
        "<td>Does the metric clear its threshold?</td><td>act</td><td>4</td></tr>"
    ) in dashboard_html
    assert "pipeline evidence" not in dashboard_html

    assert step_page.status_code == 200
    page = step_page.body.decode()
    assert 'class="decision act"' in page
    assert "association_&lt;x&gt;" in page
    assert "dataset: synthetic \u00b7 seed: 42" in page
    assert '<img src="/results/pipeline/etl/etl_effects.png"' in page
    assert '<table class="evidence">' in page
    assert "<td>count</td>" in page
    assert "ETL report" in page
    assert "show code — module.py" in page
    assert '<a href="/results/pipeline/etl/etl_decision.json">etl_decision.json</a>' in page

    multi_record = multi_record_page.body.decode()
    assert "<h2>champion_promotion</h2>" in multi_record
    assert "<h2>tuned_promotion</h2>" in multi_record
    assert multi_record.count('class="decision act"') == 2

    assert bad_page.status_code == 200
    assert caplog.text.count("unreadable decision record") == 1
