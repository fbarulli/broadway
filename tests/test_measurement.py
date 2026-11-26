"""Measurement: value, interval, null, n and seed come from ONE deterministic draw."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from broadway.reports.decision import DecisionRule, Evidence, SegmentEvidence
from broadway.reports.measurement import Measurement, MeasurementEstimator, MeasurementSpec


def _evidence() -> Evidence:
    return Evidence(
        baseline="no-effect null at the measured n",
        segments=[SegmentEvidence(segment="weekday", value=0.4, sample_size=100)],
        sensitivity={"half_sample": 0.41},
        falsifier="the interval no longer clears the null band",
        provenance={"artifact": "synthetic", "seed": "7"},
    )


def _rule(**overrides: object) -> DecisionRule:
    fields: dict[str, object] = {
        "metric": "mean_y",
        "direction": "max",
        "min_effect": 0.01,
        "min_sample_size": 100,
        "action": "act on the mean",
        "owner": "measurement",
    }
    fields.update(overrides)
    return DecisionRule(**fields)


def _mean(frame: pd.DataFrame) -> float:
    return float(frame["y"].mean())


def _abs_correlation(frame: pd.DataFrame) -> float:
    return abs(float(np.corrcoef(frame["x"], frame["y"])[0, 1]))


def _max_block_mean(frame: pd.DataFrame) -> float:
    return float(frame.groupby("block")["y"].mean().max())


def _independent_columns(rows: int) -> pd.DataFrame:
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {"x": generator.normal(size=rows), "y": generator.normal(size=rows)}
    )


def test_one_draw_supplies_value_interval_and_null() -> None:
    """The published value is the subsample's, not the full frame's."""
    frame = pd.DataFrame({"y": np.arange(1_000, dtype=float)})
    spec = MeasurementSpec(
        bootstrap="iid",
        null="shuffle",
        null_column="y",
        sample_size=100,
        iterations=200,
        seed=11,
    )

    measured = MeasurementEstimator().estimate(frame, _mean, spec, name="mean_y")

    drawn_mean = float(frame.sample(n=100, random_state=11)["y"].mean())
    assert measured.sample_size == 100
    assert measured.seed == 11
    assert measured.value == pytest.approx(drawn_mean)
    assert measured.interval_low <= measured.value <= measured.interval_high
    # A mean is invariant under permutation, so its null band must equal the
    # same draw's mean — the null cannot have been measured on another n.
    assert measured.null_threshold == pytest.approx(measured.value)


def test_same_seed_is_byte_identical_and_a_new_seed_draws_again() -> None:
    frame = _independent_columns(1_500)
    spec = MeasurementSpec(
        bootstrap="iid", null="shuffle", null_column="y", sample_size=300, iterations=150, seed=3
    )

    first = MeasurementEstimator().estimate(frame, _mean, spec)
    again = MeasurementEstimator().estimate(frame, _mean, spec)
    other = MeasurementEstimator().estimate(frame, _mean, spec.model_copy(update={"seed": 4}))

    assert first.model_dump_json() == again.model_dump_json()
    assert other.sample_size == first.sample_size
    assert other.value != first.value


def test_block_bootstrap_widens_the_interval_on_clustered_rows() -> None:
    generator = np.random.default_rng(0)
    blocked = np.repeat(np.arange(40), 25)
    frame = pd.DataFrame(
        {
            "block": blocked,
            "y": generator.normal(size=40)[blocked] + 0.1 * generator.normal(size=1_000),
        }
    )
    iid = MeasurementSpec(bootstrap="iid", iterations=300, seed=5)
    block = MeasurementSpec(bootstrap="block", block_column="block", iterations=300, seed=5)

    flat = MeasurementEstimator().estimate(frame, _mean, iid)
    clustered = MeasurementEstimator().estimate(frame, _mean, block)

    assert clustered.value == flat.value  # the CI mode never moves the point
    assert clustered.interval_high - clustered.interval_low > 2 * (
        flat.interval_high - flat.interval_low
    )
    assert clustered.resampling == "block"


def test_permutation_null_band_grows_as_the_sample_shrinks() -> None:
    frame = _independent_columns(2_000)

    def measured(sample_size: int) -> Measurement:
        spec = MeasurementSpec(
            bootstrap="iid",
            null="shuffle",
            null_column="y",
            sample_size=sample_size,
            iterations=400,
            seed=7,
        )
        return MeasurementEstimator().estimate(frame, _abs_correlation, spec)

    large, small = measured(2_000), measured(128)

    assert small.sample_size == 128
    assert small.null_threshold is not None and large.null_threshold is not None
    assert small.null_threshold > 2 * large.null_threshold


def test_block_shuffle_moves_whole_blocks_not_rows() -> None:
    generator = np.random.default_rng(0)
    blocked = np.repeat(np.arange(20), 10)
    frame = pd.DataFrame(
        {"block": blocked, "y": 10.0 * blocked + 0.5 * generator.normal(size=200)}
    )
    rows = MeasurementSpec(
        bootstrap="iid", null="shuffle", null_column="y", iterations=200, seed=9
    )
    blocks = MeasurementSpec(
        bootstrap="iid",
        null="block_shuffle",
        block_column="block",
        null_column="y",
        iterations=200,
        seed=9,
    )

    by_row = MeasurementEstimator().estimate(frame, _max_block_mean, rows)
    by_block = MeasurementEstimator().estimate(frame, _max_block_mean, blocks)

    # Whole blocks move together: the best block mean is the observed one on
    # every permutation, while row shuffling mixes blocks down to ~the mean.
    assert by_block.null_threshold == pytest.approx(by_block.value)
    assert by_row.null_threshold is not None
    assert by_row.null_threshold < by_block.value - 10.0
    assert by_block.resampling == "iid+block_shuffle"


def test_a_decision_cannot_disagree_with_its_measurement() -> None:
    frame = pd.DataFrame({"y": np.linspace(0.0, 1.0, 400)})
    spec = MeasurementSpec(
        bootstrap="iid", null="shuffle", null_column="y", iterations=200, seed=2
    )
    measured = MeasurementEstimator().estimate(frame, _mean, spec, name="mean_y")
    rule = _rule(min_effect=0.001)

    decision = measured.to_decision(rule, question="Is the mean above its null?", evidence=_evidence())

    assert decision.value == measured.value
    assert decision.interval_low == measured.interval_low
    assert decision.interval_high == measured.interval_high
    assert decision.sample_size == measured.sample_size
    assert decision.null_threshold == measured.null_threshold
    assert decision.threshold == max(rule.min_effect, measured.null_threshold)

    without_null = MeasurementEstimator().estimate(
        frame, _mean, spec.model_copy(update={"null": None, "null_column": None}), name="mean_y"
    )
    plain = without_null.to_decision(rule, question="q", evidence=_evidence())
    assert plain.null_threshold is None
    assert plain.threshold == rule.min_effect


@pytest.mark.parametrize(
    "fields",
    [
        {"bootstrap": "block"},
        {"bootstrap": "iid", "block_column": "block"},
        {"bootstrap": "iid", "null": "shuffle"},
        {"bootstrap": "iid", "null_column": "y"},
        {"bootstrap": "iid", "null": "block_shuffle", "null_column": "y"},
        {"bootstrap": "iid", "sample_size": 0},
        {"bootstrap": "iid", "iterations": 1},
        {"bootstrap": "iid", "confidence": 1.0},
    ],
)
def test_spec_rejects_modes_without_their_columns(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MeasurementSpec(seed=1, **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"value": 2.0, "interval_low": 0.0, "interval_high": 1.0},
        {"value": 0.5, "interval_low": 1.0, "interval_high": 0.0},
        {"value": float("nan"), "interval_low": 0.0, "interval_high": 1.0},
        {"value": 0.5, "interval_low": float("-inf"), "interval_high": 1.0},
        {"sample_size": 0},
    ],
)
def test_a_record_rejects_impossible_numbers(fields: dict[str, object]) -> None:
    numbers: dict[str, object] = {
        "statistic": "mean_y",
        "value": 0.5,
        "interval_low": 0.0,
        "interval_high": 1.0,
        "sample_size": 100,
        "seed": 1,
        "resampling": "iid",
        "iterations": 10,
        "confidence": 0.95,
    }
    numbers.update(fields)
    with pytest.raises(ValidationError):
        Measurement(**numbers)


def test_min_direction_cannot_consume_an_upper_tail_null() -> None:
    frame = pd.DataFrame({"y": np.linspace(0.0, 1.0, 200)})
    spec = MeasurementSpec(
        bootstrap="iid", null="shuffle", null_column="y", iterations=50, seed=4
    )
    measured = MeasurementEstimator().estimate(frame, _mean, spec)

    with pytest.raises(ValueError, match="upper-tail null"):
        measured.to_decision(
            _rule(direction="min"), question="q", evidence=_evidence()
        )


@pytest.mark.parametrize(
    ("frame", "spec", "statistic", "message"),
    [
        (
            pd.DataFrame({"y": []}),
            MeasurementSpec(bootstrap="iid", iterations=10, seed=1),
            _mean,
            "at least one row",
        ),
        (
            pd.DataFrame({"y": [1.0, 2.0, 3.0]}),
            MeasurementSpec(
                bootstrap="block", block_column="day", iterations=10, seed=1
            ),
            _mean,
            "not a column",
        ),
        (
            pd.DataFrame({"y": [1.0, 2.0, 3.0], "block": [0, 0, 0]}),
            MeasurementSpec(
                bootstrap="block", block_column="block", iterations=10, seed=1
            ),
            _mean,
            "at least two blocks",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}),
            MeasurementSpec(
                bootstrap="iid",
                null="shuffle",
                null_column="missing",
                iterations=10,
                seed=1,
            ),
            _mean,
            "not a column",
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "block": [0, 0, 0]}),
            MeasurementSpec(
                bootstrap="iid",
                null="block_shuffle",
                block_column="block",
                null_column="y",
                iterations=10,
                seed=1,
            ),
            _mean,
            "at least two blocks",
        ),
    ],
)
def test_estimator_fails_loud_on_invalid_input(
    frame: pd.DataFrame,
    spec: MeasurementSpec,
    statistic,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MeasurementEstimator().estimate(frame, statistic, spec)


def test_estimator_rejects_a_non_frame() -> None:
    spec = MeasurementSpec(bootstrap="iid", iterations=10, seed=1)
    with pytest.raises(TypeError, match="DataFrame"):
        MeasurementEstimator().estimate([1.0, 2.0], _mean, spec)


def test_estimator_wraps_a_raising_or_non_finite_statistic() -> None:
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    spec = MeasurementSpec(bootstrap="iid", iterations=10, seed=1)

    def explodes(_: pd.DataFrame) -> float:
        raise RuntimeError("statistic bug")

    def never_finite(_: pd.DataFrame) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="statistic raised RuntimeError"):
        MeasurementEstimator().estimate(frame, explodes, spec)
    with pytest.raises(ValueError, match="non-finite"):
        MeasurementEstimator().estimate(frame, never_finite, spec)
