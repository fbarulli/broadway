"""Public behavior of the dashboard's pipeline-evidence section.

One synthetic checkout with no experiment series: the dashboard root must still
render a committed pipeline decision (and its links), while malformed records
are skipped with a logged error instead of failing the page.
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


def test_dashboard_root_renders_pipeline_records_and_skips_malformed(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    root = tmp_path / "experiments"
    pipeline = root / "results" / "pipeline"

    _write(
        pipeline / "eda" / "eda_decision.json",
        Decision.model_validate(_decision()).model_dump(mode="json"),
    )
    (pipeline / "eda" / "eda_effects.png").write_bytes(b"png")
    (pipeline / "eda" / "eda_feature_decisions.json").write_text("{}", encoding="utf-8")
    _write(
        pipeline / "features" / "features_decision.json",
        Decision.model_validate(
            _decision(
                metric="missing_share",
                value=0.01,
                interval_low=0.005,
                interval_high=0.02,
                threshold=0.9,
                null_threshold=None,
                direction="min",
                verdict="no_action",
            )
        ).model_dump(mode="json"),
    )
    no_dataset = _decision(metric="no_dataset_metric")
    no_dataset["evidence"] = {
        **no_dataset["evidence"],
        "provenance": {"seed": "42"},
    }
    _write(
        pipeline / "no_dataset" / "no_dataset_decision.json",
        Decision.model_validate(no_dataset).model_dump(mode="json"),
    )

    (pipeline / "bad_list").mkdir(parents=True)
    (pipeline / "bad_list" / "bad_list_decision.json").write_text("[]", encoding="utf-8")
    overflow = json.dumps(
        Decision.model_validate(_decision()).model_dump(mode="json")
    ).replace('"sample_size": 50000', '"sample_size": 1e999')
    (pipeline / "bad_overflow").mkdir(parents=True)
    (pipeline / "bad_overflow" / "bad_overflow_decision.json").write_text(
        overflow, encoding="utf-8"
    )
    _write(
        pipeline / "bad_interval" / "bad_interval_decision.json",
        _decision(value=0.5, interval_low=0.9, interval_high=0.1),
    )
    _write(
        pipeline / "bad_direction" / "bad_direction_decision.json",
        _decision(direction="sideways"),
    )

    monkeypatch.setattr(ui, "EXPERIMENTS_ROOT", root)
    with caplog.at_level(logging.WARNING):
        response = ui.index(focus="")
        ui.index(focus="")

    assert response.status_code == 200
    assert ui.index(focus="../..").status_code == 404
    assert ui.index(focus="alpha").status_code == 404
    body = response.body.decode()
    assert '<td>eda<br><span class="dataset">synthetic</span></td>' in body
    assert "association_&lt;x&gt;" in body
    assert "association_<x>" not in body
    assert "0.9400 [0.9300, 0.9500]" in body
    assert "> 0.0500" in body
    assert "0.0089" in body
    assert "50,000" in body
    assert ">act<" in body
    assert "/results/pipeline/eda/eda_effects.png" in body
    assert "/results/pipeline/eda/eda_feature_decisions.json" in body
    assert "/results/pipeline/eda/eda_decision.json" in body

    assert '<td>features<br><span class="dataset">synthetic</span></td>' in body
    assert "< 0.9000" in body
    assert ">no_action<" in body
    assert "<td>no_dataset</td>" in body

    for skipped in ("bad_list", "bad_overflow", "bad_interval", "bad_direction"):
        assert f"<td>{skipped}</td>" not in body
    assert caplog.text.count("unreadable decision record") == 4
