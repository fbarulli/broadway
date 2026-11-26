"""StepOutput + Decision: a step publishes a computed, n-calibrated verdict."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from broadway.reports.decision import Decision, DecisionRule, Evidence, SegmentEvidence
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput


def _rule(**overrides) -> DecisionRule:
    data = {
        "metric": "ks_fare_amount",
        "direction": "max",
        "min_effect": 0.01,
        "min_sample_size": 100,
        "action": "review the policy",
        "owner": "fare_prediction",
    }
    data.update(overrides)
    return DecisionRule(**data)


def _evidence() -> Evidence:
    return Evidence(
        baseline="no-effect permutation",
        segments=[SegmentEvidence(segment="weekday", value=0.011, sample_size=800)],
        sensitivity={"ks_half_sample": 0.011},
        falsifier="the interval no longer clears the null band",
        provenance={"artifact": "demo", "seed": "42"},
    )


class _Figure:
    def savefig(self, path, dpi=None):
        path.write_bytes(b"png")


def test_verdict_act_when_the_interval_clears_the_threshold() -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.012, interval=(0.011, 0.013), sample_size=10_000, question="q", evidence=_evidence()
    )

    assert decision.verdict == "act"


def test_verdict_investigate_when_the_interval_straddles_the_threshold() -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.009, interval=(0.008, 0.012), sample_size=10_000, question="q", evidence=_evidence()
    )

    assert decision.verdict == "investigate"


def test_verdict_no_action_when_the_interval_is_inside_the_rule() -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.005, interval=(0.004, 0.006), sample_size=10_000, question="q", evidence=_evidence()
    )

    assert decision.verdict == "no_action"


def test_small_samples_get_an_insufficient_verdict() -> None:
    decision = Decision.evaluate(
        rule=_rule(min_sample_size=5_000),
        value=0.012,
        interval=(0.011, 0.013),
        sample_size=800,
        question="q",
        evidence=_evidence(),
    )

    assert decision.verdict == "insufficient"
    assert "collect at least 5,000 rows" in decision.action


def test_the_null_band_raises_the_threshold_at_small_n() -> None:
    """A 0.012 KS is noise on 1k rows: the permutation null lifts the bar."""
    decision = Decision.evaluate(
        rule=_rule(min_sample_size=100),
        value=0.012,
        interval=(0.011, 0.013),
        sample_size=1_000,
        null_values=[0.02, 0.03, 0.05, 0.06],
        question="q",
        evidence=_evidence(),
    )

    assert decision.null_threshold == pytest.approx(0.0585, abs=1e-9)
    assert decision.threshold == pytest.approx(0.0585, abs=1e-9)
    assert decision.verdict == "no_action"


def test_value_must_lie_inside_its_interval() -> None:
    with pytest.raises(ValidationError):
        Decision.evaluate(
            rule=_rule(), value=0.02, interval=(0.011, 0.013), sample_size=10_000, question="q", evidence=_evidence()
        )


def test_output_writes_the_decision_record(tmp_path) -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.012, interval=(0.011, 0.013), sample_size=10_000, question="q", evidence=_evidence()
    )
    output = StepOutput(
        results_dir=tmp_path, stem="01_step", decision=decision, log=DecisionLog(tracking_uri="")
    )
    output.save_figure(_Figure())
    report, figure, decisions = output.write()

    payload = json.loads(decisions.read_text(encoding="utf-8"))
    assert payload["verdict"] == "act"
    assert payload["threshold"] == 0.01
    text = report.read_text(encoding="utf-8")
    assert "## decision — act" in text
    assert "metric: ks_fare_amount = 0.0120 [0.0110, 0.0130]" in text
    assert "### segments" in text
    assert "### falsifier" in text
    assert figure == tmp_path / "01_step.png"


def test_output_without_a_figure_is_refused(tmp_path) -> None:
    decision = Decision.evaluate(
        rule=_rule(), value=0.012, interval=(0.011, 0.013), sample_size=10_000, question="q", evidence=_evidence()
    )
    output = StepOutput(
        results_dir=tmp_path, stem="02_step", decision=decision, log=DecisionLog(tracking_uri="")
    )

    with pytest.raises(FileNotFoundError, match="no figure"):
        output.write()


def test_evidence_bundle_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            baseline="none",
            segments=[],
            sensitivity={},
            falsifier="nothing",
            provenance={},
        )
