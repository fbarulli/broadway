"""Public behavior of pipeline steps on the experiment dashboard.

A synthetic checkout with one script series plus a published pipeline tree: the
dashboard must list ``pipeline`` after the script series, show one
``experiment | question | status | artifacts`` row per pipeline step (linked to its
step page), render that page conclusions-first (decision strip, sidecar findings,
figures, code) without the series graph, raw CSV previews, dropdown panels or
observations, and skip a malformed record with one logged traceback instead of
failing the page. The script-series step page has the same conclusions-first
shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from broadway.reports import experiments_dashboard as ui
from broadway.reports.decision import Decision
from broadway.reports.pipeline_evidence import PIPELINE_STEPS


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
    _write(
        step / "etl_feature_decisions.json",
        {
            "summary": _decision(),
            "features": [_decision(metric="association_price")],
            "outliers": [_decision(metric="outlier_share_age")],
            "comparison": _decision(metric="shift_age"),
        },
    )

    alpha_results = root / "results" / "alpha"
    alpha_results.mkdir(parents=True)
    (alpha_results / "01_step.csv").write_text("stat,value\ncount,1\n", encoding="utf-8")
    (alpha_results / "01_step.png").write_bytes(b"png")
    _write(
        alpha_results / "01_step_decision.json",
        Decision.model_validate(_decision(metric="association_alpha")).model_dump(mode="json"),
    )
    _write(
        alpha_results / "01_step_feature_decisions.json",
        {"features": [_decision(metric="association_alpha")]},
    )

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
    assert [row["experiment"] for row in rows] == [*PIPELINE_STEPS, "bad"]
    by_step = {row["experiment"]: row for row in rows}
    assert by_step["etl"] == {
        "experiment": "etl",
        "question": "Does the metric clear its threshold?",
        "status": "act",
        "artifacts": 5,
    }
    assert by_step["evaluate"] == {
        "experiment": "evaluate",
        "question": "Does the metric clear its threshold?",
        "status": "act",
        "artifacts": 2,
    }
    assert by_step["01_clean"] == {
        "experiment": "01_clean",
        "question": "",
        "status": "not run yet",
        "artifacts": 0,
    }
    assert by_step["bad"]["status"] == "not run yet"

    with caplog.at_level(logging.ERROR):
        dashboard = ui.index("pipeline")
        ui.index("pipeline")
        step_page = ui.experiment_page("etl", "pipeline")
        bad_page = ui.experiment_page("bad", "pipeline")
        multi_record_page = ui.experiment_page("evaluate", "pipeline")
        pending_page = ui.experiment_page("01_clean", "pipeline")

    assert dashboard.status_code == 200
    dashboard_html = bytes(dashboard.body).decode()
    assert "<tr><th>experiment</th><th>question</th><th>status</th><th>artifacts</th></tr>" in (
        dashboard_html
    )
    assert (
        '<tr><td><a href="/experiments/etl?focus=pipeline">etl</a></td>'
        "<td>Does the metric clear its threshold?</td><td>act</td><td>5</td></tr>"
    ) in dashboard_html
    assert "pipeline evidence" not in dashboard_html

    assert step_page.status_code == 200
    page = bytes(step_page.body).decode()
    assert '<p class="question">Does the metric clear its threshold?</p>' in page
    assert 'class="decision act"' in page
    assert "code — module.py" in page
    # The conclusions stay: sidecar findings as tables, figures, then the code.
    assert "findings — features" in page
    assert "association_price" in page
    assert "outlier_share_age" in page
    assert "shift_age" in page
    assert '<table class="evidence">' in page
    assert '<img src="/results/pipeline/etl/etl_effects.png"' in page
    # No graph, no raw CSV preview, no dropdown panels, no artifact links/observations.
    assert "steps taken in the pipeline" not in page
    assert "tables and raw artifacts" not in page
    assert "<details" not in page
    assert "etl_effects.csv" not in page
    assert "<td>count</td>" not in page
    assert "ETL report" not in page
    assert '<a href="/results/' not in page
    assert "observations" not in page

    experiment = bytes(ui.experiment_page("01_step", "alpha").body).decode()
    assert '<p class="question">A script step.</p>' in experiment
    assert 'class="decision act"' in experiment
    assert "findings — features" in experiment
    assert "association_alpha" in experiment
    assert '<img src="/results/alpha/01_step.png"' in experiment
    assert "code — 01_step.py" in experiment
    assert "steps taken in the pipeline" not in experiment
    assert "tables and raw artifacts" not in experiment
    assert "<details" not in experiment
    assert "<td>count</td>" not in experiment
    assert '<a href="/results/' not in experiment
    assert "observations" not in experiment

    multi_record = bytes(multi_record_page.body).decode()
    assert "<h2>champion_promotion</h2>" in multi_record
    assert "<h2>tuned_promotion</h2>" in multi_record
    assert multi_record.count('class="decision act"') == 2

    assert bad_page.status_code == 200
    assert pending_page.status_code == 200
    pending_html = bytes(pending_page.body).decode()
    assert "not run yet — this step will publish its decision record" in pending_html
    assert "<table" not in pending_html
    assert "<img" not in pending_html
    assert caplog.text.count("unreadable decision record") == 1


def test_pipeline_map_lists_pending_and_fills_when_a_lane_runs(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "experiments"
    pipeline = root / "results" / "pipeline"
    for step in ("etl", "eda"):
        _write(
            pipeline / step / f"{step}_decision.json",
            Decision.model_validate(_decision(metric=step)).model_dump(mode="json"),
        )
    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)

    rows = ui.census_rows("pipeline")
    assert [row["experiment"] for row in rows] == list(PIPELINE_STEPS)
    by_step = {row["experiment"]: row for row in rows}
    assert by_step["etl"]["status"] == "act"
    assert by_step["01_clean"] == {
        "experiment": "01_clean",
        "question": "",
        "status": "not run yet",
        "artifacts": 0,
    }
    for step in ("01_clean", "06_correlation", "evaluate"):
        response = ui.experiment_page(step, "pipeline")
        assert response.status_code == 200
        body = bytes(response.body).decode()
        assert "not run yet — this step will publish" in body
        assert "<table" not in body
        assert "<img" not in body

    # A lane record fills its page and row on the next request.
    _write(
        pipeline / "01_clean" / "01_clean_decision.json",
        Decision.model_validate(_decision(metric="rows_dropped_share")).model_dump(mode="json"),
    )
    _write(
        pipeline / "01_clean" / "01_clean_feature_decisions.json",
        {"features": [_decision(metric="association_clean")]},
    )
    by_step = {row["experiment"]: row for row in ui.census_rows("pipeline")}
    assert by_step["01_clean"]["status"] == "act"
    page = ui.experiment_page("01_clean", "pipeline")
    assert page.status_code == 200
    body = bytes(page.body).decode()
    assert "findings — features" in body
    assert "association_clean" in body
