from __future__ import annotations

import numpy as np
import pytest

from broadway.baseline.contracts import load_result
from broadway.baseline.module import load_persisted
from broadway.causal.contracts import load_design
from broadway.config.loader import load_config
from broadway.evaluate.contracts import EvaluationResult
from broadway.evaluate.metrics import compute_metrics
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction
from broadway.stats.plan import load_plan
from broadway.training.contracts import TrainingResult
from broadway.training.optuna import run_study


def test_load_result_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "b.json"
    bad.write_text("not json")
    with pytest.raises(Exception):
        load_result(bad)

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(Exception):
        load_result(empty)


def test_load_plan_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "p.json"
    bad.write_text("not json")
    with pytest.raises(Exception):
        load_plan(bad)

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(Exception):
        load_plan(empty)


def test_load_design_malformed_raises(tmp_path) -> None:
    bad = tmp_path / "d.json"
    bad.write_text("not json")
    with pytest.raises(Exception):
        load_design(bad)

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(Exception):
        load_design(empty)


def test_training_result_malformed_raises() -> None:
    with pytest.raises(Exception):
        TrainingResult.model_validate_json("{}")


def test_evaluation_result_malformed_raises() -> None:
    with pytest.raises(Exception):
        EvaluationResult.model_validate_json("{}")


def test_load_persisted_missing_is_none(tmp_path) -> None:
    cfg = load_config("baseline", dataset="test", analysis="test")
    cfg = cfg.model_copy(
        update={"baseline": cfg.baseline.model_copy(update={"output_dir": str(tmp_path / "nope")})}
    )
    assert load_persisted(cfg) is None


def test_compute_metrics_nan_raises() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_compute_metrics_inf_raises() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, np.inf]), np.array([1.0, 2.0]))


def test_optuna_all_invalid_raises() -> None:
    with pytest.raises(ValueError):
        run_study(lambda params: float("nan"), {"x": [0, 1]}, n_trials=2)


def _audit(rows_in: int, rows_out: int, unexplained: int) -> TransformAudit:
    dropped = rows_in - rows_out
    return TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=dropped,
        rows_dropped_unexplained=unexplained,
        reasons=["unexpected row loss: 40 rows"],
        columns_before=["a", "b"],
        columns_after=["a", "b"],
        columns_added=[],
        columns_removed=[],
    )


def test_enforce_drop_fraction_raises() -> None:
    audit = _audit(rows_in=100, rows_out=60, unexplained=40)
    with pytest.raises(ValueError):
        enforce_drop_fraction(audit, 0.1)
    enforce_drop_fraction(audit, 0.5)


def test_enforce_drop_fraction_zero_rows_ok() -> None:
    audit = _audit(rows_in=0, rows_out=0, unexplained=0)
    enforce_drop_fraction(audit, 0.0)
