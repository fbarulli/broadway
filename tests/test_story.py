"""Public behavior of the project story page.

A synthetic checkout with two valid pipeline records, their EDA evidence and one
malformed record: the story must lead with the question, narrate each stage
conclusions-first, reference generated figures, hyperlink the decision records,
say "not run yet" for a stage with no record, escape record text, and never
raise for the malformed record (one logged traceback instead).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

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
        "owner": "synthetic",
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


def test_story_leads_with_conclusions_and_skips_bad_records(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    root = tmp_path / "experiments"
    etl = root / "results" / "pipeline" / "etl"
    _write(
        etl / "etl_decision.json",
        Decision.model_validate(
            _decision(
                question="Did <b>ingest</b> lose rows the contract explains?",
                metric="rows_dropped_unexplained_share",
                value=0.0,
                interval_low=0.0,
                interval_high=0.0,
                unit="share",
                verdict="no_action",
                sample_size=50,
                evidence={
                    "baseline": "every dropped row carries a reason",
                    "delta": None,
                    "segments": [{"segment": "unexplained", "value": 0.0, "sample_size": 50}],
                    "sensitivity": {"rows_out": 50.0},
                    "falsifier": "the interval clears the drop ceiling",
                    "provenance": {
                        "dataset": "synthetic",
                        "rows_in": "60",
                        "rows_out": "50",
                        "reasons": "null target",
                    },
                },
            )
        ).model_dump(mode="json"),
    )

    eda = root / "results" / "pipeline" / "eda"
    _write(
        eda / "eda_decision.json",
        Decision.model_validate(
            _decision(
                metric="association_rating_accuracy_score",
                value=0.25,
                interval_low=0.18,
                interval_high=0.31,
                unit="epsilon_squared",
                sample_size=50,
            )
        ).model_dump(mode="json"),
    )
    _write(
        eda / "eda_feature_decisions.json",
        {
            "features": [
                _decision(
                    metric="association_rating_accuracy_score",
                    value=0.25,
                    interval_low=0.18,
                    interval_high=0.31,
                    sample_size=50,
                ),
                _decision(
                    metric="association_personalization_score",
                    value=0.19,
                    interval_low=0.14,
                    interval_high=0.24,
                    sample_size=50,
                ),
                _decision(
                    metric="association_browsing_frequency",
                    value=0.01,
                    interval_low=0.0,
                    interval_high=0.02,
                    verdict="no_action",
                    sample_size=50,
                ),
            ],
            "outliers": [
                _decision(
                    metric="outlier_share_age",
                    value=0.03,
                    interval_low=0.02,
                    interval_high=0.04,
                    sample_size=50,
                ),
            ],
            "comparison": [
                _decision(
                    metric="standardized_shift_age",
                    value=0.0,
                    interval_low=-0.01,
                    interval_high=0.01,
                    sample_size=50,
                    verdict="no_action",
                )
            ],
        },
    )
    (eda / "eda_effects.csv").write_text(
        "feature,effect,verdict\nrating_accuracy_score,0.25,act\n", encoding="utf-8"
    )

    # Malformed record: the story must skip it and log one traceback.
    hpo = root / "results" / "pipeline" / "hpo"
    hpo.mkdir(parents=True)
    (hpo / "hpo_decision.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)
    monkeypatch.setenv("BROADWAY_STORY_DATASET", "synthetic")

    with caplog.at_level(logging.ERROR):
        client = TestClient(ui.app)
        response = client.get("/story")
        figure = client.get("/story/figure/data")
        missing_figure = client.get("/story/figure/model")

    assert response.status_code == 200
    page = response.text
    assert "The question" in page
    assert "What the data says" in page
    assert "bad review" in page
    assert "rating accuracy score" in page
    # Conclusions first, graph, then numbers and the record link.
    assert "Verdict: no_action" in page
    assert 'alt="The data figure"' in page
    assert '<img src="/story/figure/data.png"' in page
    assert '<img src="/story/figure/eda.png"' in page
    assert "verdict: act" in page
    assert '<a href="/experiments/etl?focus=pipeline">ETL record</a>' in page
    # A stage without a record says so; the malformed HPO record does too.
    assert "not run yet — the features decision record has not been published." in page
    assert "not run yet — the baseline decision record has not been published." in page
    # Dynamic record text is escaped.
    assert "&lt;b&gt;ingest&lt;/b&gt;" in page
    assert "<b>ingest</b>" not in page
    # The root dashboard keeps the story as the first surface a human sees.
    dashboard = ui.index("pipeline").body.decode()
    assert 'class="story-banner"' in dashboard
    assert '<a href="/story">Read the project story' in dashboard
    # Graph and 404s.
    assert figure.status_code == 200
    assert figure.headers["content-type"] == "image/png"
    assert figure.content.startswith(b"\x89PNG")
    assert missing_figure.status_code == 404
    assert caplog.text.count("unreadable decision record") == 1
