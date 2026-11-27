"""One measurement owner: point estimate, interval, null band, n and seed together.

A published number is only evidence when every part of it was measured on the
same rows at the same sample size with the same seed. This module is that one
place: the point estimate, the bootstrap interval and the no-effect null are
all computed from ONE deterministic draw of the frame, and the immutable
:class:`Measurement` records the ``n`` and ``seed`` that were actually used —
so a surface can no longer quote a value at one ``n``, an interval at another,
and a seed it never applied.

Resampling is clustered by default: rows of one day/route are not independent
draws, so ``bootstrap="block"`` resamples whole blocks and ``null=
"block_shuffle"`` permutes whole blocks instead of individual rows. A
:class:`Measurement` is turned into the existing
:class:`~broadway.reports.decision.Decision` through
:meth:`Measurement.to_decision`, which copies the record's value, interval,
sample size and null threshold — the verdict can then never disagree with the
measurement it claims to be based on.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from broadway.reports.decision import Decision, DecisionRule, Evidence

logger = logging.getLogger(__name__)

BootstrapMode = Literal["iid", "block"]
NullMode = Literal["shuffle", "block_shuffle"]
Statistic = Callable[[pd.DataFrame], float]

# Independent RNG streams per resampling stage: re-tuning the bootstrap budget
# cannot silently move the null band, while one seed still reproduces both.
_BOOTSTRAP_STREAM = 0
_NULL_STREAM = 1

# pandas' seeded ``DataFrame.sample`` accepts a 32-bit integer only.
_MAX_SEED = 2**32 - 1


class MeasurementSpec(BaseModel):
    """How one metric is measured: resampling mode, sample size, seed, confidence.

    ``block_column`` names the cluster label (a day, a route) and is required
    exactly when a mode resamples blocks — ``bootstrap="block"`` or
    ``null="block_shuffle"`` — and forbidden otherwise, so a mode can never
    silently ignore a declared block. ``null_column`` is the column whose
    dependence with the statistic the null destroys, and is required exactly
    when a null is declared.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap: BootstrapMode
    block_column: str | None = None
    null: NullMode | None = None
    null_column: str | None = None
    sample_size: int | None = Field(default=None, gt=0)
    iterations: int = Field(default=1000, ge=2)
    seed: int = Field(ge=0, le=_MAX_SEED)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _references_declared(self) -> MeasurementSpec:
        needs_blocks = self.bootstrap == "block" or self.null == "block_shuffle"
        if needs_blocks and not self.block_column:
            raise ValueError(
                "block_column is required when bootstrap='block' "
                "or null='block_shuffle'"
            )
        if not needs_blocks and self.block_column is not None:
            raise ValueError(
                "block_column only applies to bootstrap='block' or "
                "null='block_shuffle'"
            )
        if self.null is not None and not self.null_column:
            raise ValueError(f"null='{self.null}' requires null_column")
        if self.null is None and self.null_column is not None:
            raise ValueError("null_column only applies when null is declared")
        return self


class Measurement(BaseModel):
    """One immutable measurement: value + interval + null band at one n and seed.

    ``null_threshold`` is the ``confidence``-quantile of the no-effect null at
    the recorded sample size (the value a ``max``-direction statistic must
    clear); ``None`` means no null was measured. ``resampling`` names the
    constructions used, e.g. ``"iid+shuffle"`` or ``"block+block_shuffle"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    statistic: str = Field(min_length=1)
    value: float
    interval_low: float
    interval_high: float
    null_threshold: float | None = None
    # True when the bootstrap interval excludes the point estimate (a biased
    # resample); the decision is then never `act`.
    bootstrap_biased: bool = False
    sample_size: int = Field(gt=0)
    seed: int = Field(ge=0)
    resampling: str = Field(min_length=1)
    block_column: str | None = None
    iterations: int = Field(ge=2)
    confidence: float = Field(gt=0.0, lt=1.0)
    unit: str | None = None

    @model_validator(mode="after")
    def _ordered_and_finite(self) -> Measurement:
        numbers = (
            ("value", self.value),
            ("interval_low", self.interval_low),
            ("interval_high", self.interval_high),
            ("null_threshold", self.null_threshold),
        )
        for name, number in numbers:
            if number is not None and not math.isfinite(number):
                raise ValueError(f"{name} must be finite, got {number!r}")
        if self.interval_low > self.interval_high:
            raise ValueError("interval_low must not exceed interval_high")
        inside = self.interval_low <= self.value <= self.interval_high
        if inside == self.bootstrap_biased:
            raise ValueError(
                "bootstrap_biased must be true exactly when the point estimate "
                "lies outside its interval"
            )
        return self

    def to_decision(
        self,
        rule: DecisionRule,
        *,
        question: str,
        evidence: Evidence,
        reference: float | None = None,
    ) -> Decision:
        """Build the step's :class:`Decision` from this record's own numbers.

        The value, interval, sample size and null threshold are the record's —
        only the verdict rule (direction, minimum effect, action, owner) comes
        from ``rule``. The null band is recorded as an upper-tail threshold, so
        a ``min``-direction rule cannot consume it (that would judge a metric
        against the wrong tail) and fails loud instead.
        """
        if self.null_threshold is not None and rule.direction == "min":
            raise ValueError(
                f"measurement {self.statistic!r} records an upper-tail null "
                f"threshold ({self.null_threshold!r}); rule {rule.metric!r} is "
                "min-direction — measure a magnitude statistic instead"
            )
        null_values = (
            None if self.null_threshold is None else [self.null_threshold]
        )
        return Decision.evaluate(
            rule=rule,
            value=self.value,
            interval=(self.interval_low, self.interval_high),
            sample_size=self.sample_size,
            question=question,
            evidence=evidence,
            null_values=null_values,
            reference=reference,
            unit=self.unit,
            bootstrap_biased=self.bootstrap_biased,
        )


class MeasurementEstimator:
    """Measure a statistic on one deterministic draw of a frame.

    The draw is taken ONCE: when ``spec.sample_size`` is smaller than the frame
    it is a seeded subsample and every later computation (value, bootstrap,
    null) reuses it, so the record's ``n`` is the ``n`` that was measured.
    """

    def estimate(
        self,
        frame: pd.DataFrame,
        statistic: Statistic,
        spec: MeasurementSpec,
        *,
        name: str | None = None,
        unit: str | None = None,
    ) -> Measurement:
        """Return the one immutable :class:`Measurement` for ``frame``/``spec``.

        ``name`` labels the statistic in the record (defaults to the callable's
        ``__name__``; pass the decision rule's metric when publishing).
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"frame must be a DataFrame, got {type(frame).__name__}")
        draw = self._draw(frame, spec)
        label = name if name is not None else self._label(statistic)
        value = self._evaluate(statistic, draw, f"statistic {label!r}")
        low, high = self._interval(draw, statistic, spec)
        null_threshold = self._null_threshold(draw, statistic, spec)
        return Measurement(
            statistic=label,
            value=value,
            interval_low=low,
            interval_high=high,
            null_threshold=null_threshold,
            bootstrap_biased=not (low <= value <= high),
            sample_size=len(draw),
            seed=spec.seed,
            resampling=self._resampling(spec),
            block_column=spec.block_column,
            iterations=spec.iterations,
            confidence=spec.confidence,
            unit=unit,
        )

    def _draw(self, frame: pd.DataFrame, spec: MeasurementSpec) -> pd.DataFrame:
        """The ONE deterministic draw every part of the measurement reuses."""
        if len(frame) == 0:
            raise ValueError("measurement needs at least one row, got an empty frame")
        if spec.sample_size is not None and spec.sample_size < len(frame):
            drawn = frame.sample(n=spec.sample_size, random_state=spec.seed)
        else:
            drawn = frame
        return drawn.reset_index(drop=True)

    @staticmethod
    def _label(statistic: Statistic) -> str:
        label = str(getattr(statistic, "__name__", ""))
        return label if label else "statistic"

    @staticmethod
    def _resampling(spec: MeasurementSpec) -> str:
        return spec.bootstrap if spec.null is None else f"{spec.bootstrap}+{spec.null}"

    @staticmethod
    def _column(frame: pd.DataFrame, column: str | None, role: str) -> str:
        """A declared column must be present in the drawn frame."""
        assert column is not None  # MeasurementSpec guarantees the mode's column
        if column not in frame.columns:
            available = ", ".join(map(str, frame.columns))
            raise ValueError(f"{role} {column!r} is not a column (have: {available})")
        return column

    @staticmethod
    def _evaluate(statistic: Statistic, frame: pd.DataFrame, context: str) -> float:
        """Run the statistic and fail loud on a raised or non-finite result."""
        try:
            value = float(statistic(frame))
        except Exception as error:
            logger.exception("%s failed on a %d-row draw", context, len(frame))
            raise ValueError(
                f"{context}: statistic raised {type(error).__name__}: {error}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"{context}: statistic returned non-finite value {value!r}")
        return value

    def _interval(
        self,
        draw: pd.DataFrame,
        statistic: Statistic,
        spec: MeasurementSpec,
    ) -> tuple[float, float]:
        """Percentile bootstrap interval; never clamped to contain the point.

        A bootstrap that lies entirely above (or below) the observed statistic
        is a real signal of resample bias. Clamping it to contain the point
        would report an interval the data never produced and hide the bias;
        the caller records ``bootstrap_biased`` instead.
        """
        generator = np.random.default_rng([spec.seed, _BOOTSTRAP_STREAM])
        draws = np.empty(spec.iterations)
        if spec.bootstrap == "iid":
            self._iid_draws(draw, statistic, spec, generator, draws)
        else:
            self._block_draws(draw, statistic, spec, generator, draws)
        tail = (1.0 - spec.confidence) / 2.0 * 100.0
        low, high = np.percentile(draws, [tail, 100.0 - tail])
        return float(low), float(high)

    def _iid_draws(
        self,
        draw: pd.DataFrame,
        statistic: Statistic,
        spec: MeasurementSpec,
        generator: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        """Rows resampled with replacement, treating the draw as independent."""
        rows = len(draw)
        for iteration in range(spec.iterations):
            positions = generator.integers(0, rows, rows)
            out[iteration] = self._evaluate(
                statistic, draw.iloc[positions], f"iid bootstrap draw {iteration}"
            )

    def _block_draws(
        self,
        draw: pd.DataFrame,
        statistic: Statistic,
        spec: MeasurementSpec,
        generator: np.random.Generator,
        out: np.ndarray,
    ) -> None:
        """Whole blocks resampled with replacement (all rows of each block)."""
        positions = self._block_positions(draw, spec.block_column, "bootstrap")
        blocks = len(positions)
        for iteration in range(spec.iterations):
            chosen = generator.integers(0, blocks, blocks).tolist()
            index = np.concatenate([positions[block] for block in chosen])
            out[iteration] = self._evaluate(
                statistic, draw.iloc[index], f"block bootstrap draw {iteration}"
            )

    def _null_threshold(
        self, draw: pd.DataFrame, statistic: Statistic, spec: MeasurementSpec
    ) -> float | None:
        """The ``confidence``-quantile of the no-effect null at the draw's n."""
        if spec.null is None:
            return None
        column = self._column(draw, spec.null_column, "null_column")
        generator = np.random.default_rng([spec.seed, _NULL_STREAM])
        values = draw[column].to_numpy()
        if spec.null == "shuffle":
            permutations: Iterable[np.ndarray] = (
                generator.permutation(values) for _ in range(spec.iterations)
            )
        else:
            blocks = self._block_positions(draw, spec.block_column, "block_shuffle")
            chunks = [values[positions] for positions in blocks]
            permutations = (
                np.concatenate(
                    [chunks[block] for block in generator.permutation(len(chunks)).tolist()]
                )
                for _ in range(spec.iterations)
            )
        out = np.empty(spec.iterations)
        for iteration, permuted in enumerate(permutations):
            permuted_frame = draw.copy()
            permuted_frame[column] = permuted
            out[iteration] = self._evaluate(
                statistic, permuted_frame, f"{spec.null} null draw {iteration}"
            )
        return float(np.percentile(out, spec.confidence * 100.0))

    def _block_positions(
        self, frame: pd.DataFrame, column: str | None, role: str
    ) -> list[np.ndarray]:
        """Row positions of every block; a single block cannot be resampled."""
        resolved = self._column(frame, column, "block_column")
        labels, uniques = pd.factorize(frame[resolved], sort=False)
        if len(uniques) < 2:
            raise ValueError(
                f"{role} needs at least two blocks in {resolved!r}, got {len(uniques)}"
            )
        return [np.flatnonzero(labels == index) for index in range(len(uniques))]
