"""Decision records: a measured value, its uncertainty, a calibrated rule, a verdict.

An insight is prose. A decision is evidence: one metric, a 95% interval, the
**null band at the observed sample size**, the config-declared minimum effect,
and the resulting verdict (``act`` / ``investigate`` / ``no_action`` /
``insufficient``) plus the owner's action.

Absolute thresholds do not travel across sample sizes: a KS distance of 0.01 is
noise on 1,000 rows and a real signal on 8.5M. The rule therefore clears
``max(min_effect, null_quantile of the permutation null)`` and a sample below
``min_sample_size`` yields ``insufficient`` instead of a false verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["act", "investigate", "no_action", "insufficient"]
Direction = Literal["max", "min"]


class DecisionRule(BaseModel):
    """The config-declared test for one step (``decisions.<step>`` in YAML)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(min_length=1)
    direction: Direction
    min_effect: float = Field(gt=0)
    null_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_sample_size: int = Field(default=100, gt=0)
    action: str = Field(min_length=1)
    owner: str = Field(min_length=1)


class Decision(BaseModel):
    """One evaluated decision: value + interval vs the calibrated threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    interval_low: float
    interval_high: float
    reference: float | None = None
    delta: float | None = None
    threshold: float
    null_threshold: float | None = None
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
        null_values: Sequence[float] | None = None,
        reference: float | None = None,
    ) -> Decision:
        """Compute the verdict from the interval vs the calibrated threshold.

        ``null_values`` is the metric's distribution under the no-effect
        permutation at this sample size; absent, the threshold is the declared
        ``min_effect`` alone (a small-sample run then usually lands in
        ``insufficient`` or ``investigate``, which is the honest outcome).
        """
        low, high = interval
        null_threshold = None
        threshold = rule.min_effect
        if null_values is not None and len(null_values) > 0:
            percentile = rule.null_quantile if rule.direction == "max" else 1.0 - rule.null_quantile
            null_threshold = float(np.percentile(np.asarray(null_values, dtype=float), percentile * 100))
            threshold = max(threshold, null_threshold) if rule.direction == "max" else min(
                threshold, null_threshold
            )
        if sample_size < rule.min_sample_size:
            verdict: Verdict = "insufficient"
            action = f"collect at least {rule.min_sample_size:,} rows before deciding"
        elif rule.direction == "max":
            if low > threshold:
                verdict = "act"
            elif high > threshold:
                verdict = "investigate"
            else:
                verdict = "no_action"
            action = rule.action
        else:
            if high < threshold:
                verdict = "act"
            elif low < threshold:
                verdict = "investigate"
            else:
                verdict = "no_action"
            action = rule.action
        return cls(
            question=question,
            metric=rule.metric,
            value=value,
            interval_low=low,
            interval_high=high,
            reference=reference,
            delta=None if reference is None else value - reference,
            threshold=threshold,
            null_threshold=null_threshold,
            direction=rule.direction,
            verdict=verdict,
            action=action,
            owner=rule.owner,
            sample_size=sample_size,
        )
