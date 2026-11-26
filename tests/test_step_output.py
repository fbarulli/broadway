"""StepOutput + Decision: a step publishes a computed verdict, not prose."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from broadway.reports.decision import Decision, DecisionRule
from broadway.reports.step_output import StepOutput


def _rule(**overrides) -> DecisionRule:
    data = {
        "metric": "ks_fare_amount",
        "direction": "max",
        "threshold": 0.01,
        "action": "review the policy",
        "owner": "fare_prediction",
    }
    data.update(overrides)
    return DecisionRule(**data)


class _Figure:
    def savefig(self, path, dpi=None):
        path.write_bytes(b"png")


def test_verdict_act_when_the_interval_clears_the_threshold() -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.012, interval=(0.011, 0.013), sample_size=10, question="q"
    )

    assert decision.verdict == "act"


def test_verdict_investigate_when_the_interval_straddles_the_threshold() -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.009, interval=(0.008, 0.012), sample_size=10, question="q"
    )

    assert decision.verdict == "investigate"


def test_verdict_no_action_when_the_interval_is_inside_the_rule() -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.005, interval=(0.004, 0.006), sample_size=10, question="q"
    )

    assert decision.verdict == "no_action"


def test_value_must_lie_inside_its_interval() -> None:
    with pytest.raises(ValidationError):
        Decision.evaluate(
            rule=_rule(), value=0.02, interval=(0.011, 0.013), sample_size=10, question="q"
        )


def test_output_writes_the_decision_record(tmp_path) -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.012, interval=(0.011, 0.013), sample_size=10, question="q"
    )
    output = StepOutput(results_dir=tmp_path, stem="01_step", decision=decision)
    output.save_figure(_Figure())
    report, figure, decisions = output.write()

    payload = json.loads(decisions.read_text(encoding="utf-8"))
    assert payload["verdict"] == "act"
    assert payload["threshold"] == 0.01
    text = report.read_text(encoding="utf-8")
    assert "## decision — act" in text
    assert "metric: ks_fare_amount = 0.0120 [0.0110, 0.0130]" in text
    assert figure == tmp_path / "01_step.png"


def test_output_without_a_figure_is_refused(tmp_path) -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.012, interval=(0.011, 0.013), sample_size=10, question="q"
    )
    output = StepOutput(results_dir=tmp_path, stem="02_step", decision=decision)

    with pytest.raises(FileNotFoundError, match="no figure"):
        output.write()
