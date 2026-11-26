"""DecisionLog: MLflow projects the decision record, it never becomes the truth."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput

EXPERIMENT = "broadway-decisions"


class _Figure:
    def savefig(self, path: Path, dpi: int | None = None) -> None:
        path.write_bytes(b"png")


def _decision() -> Decision:
    rule = DecisionRule(
        metric="ks_fare_amount",
        direction="max",
        min_effect=0.01,
        min_sample_size=100,
        action="review the policy",
        owner="fare_prediction",
    )
    evidence = Evidence(
        baseline="no-effect permutation",
        segments=[SegmentEvidence(segment="weekday", value=0.011, sample_size=800)],
        sensitivity={"ks_half_sample": 0.011},
        falsifier="the interval no longer clears the null band",
        provenance={"artifact": "demo", "seed": "42"},
    )
    return Decision.evaluate(
        rule=rule,
        value=0.012,
        interval=(0.011, 0.013),
        sample_size=10_000,
        question="q",
        evidence=evidence,
        reference=0.010,
    )


def _output(results: Path, log: DecisionLog) -> StepOutput:
    output = StepOutput(results_dir=results, stem="01_step", decision=_decision(), log=log)
    output.add_table("funnel", pd.DataFrame({"stage": ["source"], "rows": [10]}))
    output.save_figure(_Figure())
    return output


def test_disabled_without_a_uri_is_a_noop_with_a_recorded_reason(tmp_path: Path) -> None:
    log = DecisionLog(tracking_uri="")

    _, figure, record = _output(tmp_path, log).write()
    before = record.read_bytes()

    assert log.enabled() is False
    assert log.reason is not None
    assert "mlflow_tracking_uri" in log.reason
    assert record.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "01_step.md",
        "01_step.png",
        "01_step_decision.json",
        "01_step_funnel.csv",
    ]
    assert (
        log.log(
            stem="01_step",
            decision=_decision(),
            figure=figure,
            tables=[],
            params={},
            record_path=record,
        )
        is None
    )

    unreadable = DecisionLog.from_config(environment="missing-environment")
    assert unreadable.enabled() is False
    assert "missing-environment" in (unreadable.reason or "")


def test_enabled_projects_the_run_and_leaves_the_record_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    log = DecisionLog(tracking_uri=(tmp_path / "mlruns").as_uri())

    _, _, record = _output(results, log).write()
    record_bytes = record.read_bytes()
    _output(results, log).write()

    # A rerun never rewrites the record and adds no local projection state.
    assert record.read_bytes() == record_bytes
    assert (
        "mlflow_run_id"
        not in json.loads(record.read_text(encoding="utf-8"))["evidence"]["provenance"]
    )
    assert not list(results.glob("*.mlflow.json"))
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert experiment is not None
    assert len(mlflow.search_runs(experiment_ids=[experiment.experiment_id])) == 2

    # One invocation projecting the same step twice reuses its run.
    monkeypatch.setenv("BROADWAY_RUN_ID", "a" * 32)
    _output(results, log).write()
    _output(results, log).write()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.broadway_run_id = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        output_format="list",
    )
    assert len(runs) == 1
    run = client.get_run(runs[0].info.run_id)
    assert run.data.tags["step"] == "01_step"
    assert run.data.tags["verdict"] == "act"
    assert run.data.tags["owner"] == "fare_prediction"
    assert run.data.tags["action"] == "review the policy"
    assert run.data.tags["direction"] == "max"
    assert run.data.params["metric"] == "ks_fare_amount"
    assert run.data.params["threshold"] == "0.01"
    assert run.data.params["sample_size"] == "10000"
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["value"] == pytest.approx(0.012)
    assert run.data.metrics["interval_low"] == pytest.approx(0.011)
    assert run.data.metrics["interval_high"] == pytest.approx(0.013)
    assert run.data.metrics["threshold"] == pytest.approx(0.01)
    assert run.data.metrics["delta"] == pytest.approx(0.002)
    artifacts = {Path(item.path).name for item in client.list_artifacts(run.info.run_id)}
    assert {"01_step.png", "01_step_funnel.csv", "01_step_decision.json"} <= artifacts
    assert record.read_bytes() == record_bytes


def test_unreachable_server_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*args: object, **kwargs: object) -> None:
        raise mlflow.exceptions.MlflowException(
            "API request failed: [Errno 111] Connection refused"
        )

    monkeypatch.setattr(mlflow, "set_experiment", _refuse)
    log = DecisionLog(tracking_uri="http://localhost:5000")

    report, figure, record = _output(tmp_path, log).write()

    assert log.reason is not None
    assert "unreachable" in log.reason
    assert all(path.exists() for path in (report, figure, record))
    assert not list(tmp_path.glob("*.mlflow.json"))
