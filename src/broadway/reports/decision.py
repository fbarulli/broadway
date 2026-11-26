"""Decision records: a measured value, its uncertainty, a rule, and a verdict.

An insight is prose. A decision is evidence: one metric, a 95% interval, the
config-declared threshold it must clear, the resulting verdict
(``act`` / ``investigate`` / ``no_action``) and the action an owner takes. The
verdict is computed from the interval against the threshold — never hand-typed —
so a step cannot claim "act" on noise or "no action" while the interval crosses
the line.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["act", "investigate", "no_action"]
Direction = Literal["max", "min"]


class DecisionRule(BaseModel):
    """The config-declared test for one step (``decisions.<step>`` in YAML).

    ``direction: max`` means the metric must stay at or below ``threshold``;
    ``direction: min`` means it must stay at or above it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(min_length=1)
    direction: Direction
    threshold: float
    action: str = Field(min_length=1)
    owner: str = Field(min_length=1)


class Decision(BaseModel):
    """One evaluated decision: value + interval vs the rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    interval_low: float
    interval_high: float
    reference: float | None = None
    delta: float | None = None
    threshold: float
    direction: Direction
    verdict: Verdict
    action: str
    owner: str
    sample_size: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered_interval(self) -> Decision:
        if self.interval_low > self.interval_high:
            raise ValueError("interval_low must not exceed interval_high")
        if not self.interval_low <= self.value <= self.interval_high:
            raise ValueError("value must lie inside its interval")
        return self

    @classmethod
    def evaluate(
        cls,
        *,
        rule: DecisionRule,
        value: float,
        interval: tuple[float, float],
        sample_size: int,
        question: str,
        reference: float | None = None,
    ) -> Decision:
        """Compute the verdict from the interval vs the rule's threshold."""
        low, high = interval
        if rule.direction == "max":
            if low > rule.threshold:
                verdict: Verdict = "act"
            elif high > rule.threshold:
                verdict = "investigate"
            else:
                verdict = "no_action"
        else:
            if high < rule.threshold:
                verdict = "act"
            elif low < rule.threshold:
                verdict = "investigate"
            else:
                verdict = "no_action"
        return cls(
            question=question,
            metric=rule.metric,
            value=value,
            interval_low=low,
            interval_high=high,
            reference=reference,
            delta=None if reference is None else value - reference,
            threshold=rule.threshold,
            direction=rule.direction,
            verdict=verdict,
            action=rule.action,
            owner=rule.owner,
            sample_size=sample_size,
        )
