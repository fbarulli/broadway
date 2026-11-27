"""Public behavior of the EDA decisions: correction, test choice, verdicts, artifacts.

Every number below is hermetic (seeded synthetic frames): the correction on
known p-values, the per-dtype test choice, a significant-but-trivial association
that must not act, an outlier share against the configured maximum, and the set
of files the step publishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import EdaStep, TaskType
from broadway.eda import module
from broadway.eda.correction import MultipleTestingCorrection
from broadway.eda.correlation import CorrelationAnalyzer
from broadway.eda.models import QualityResult
from broadway.eda.outliers import OutlierDecisionAnalyzer
from broadway.eda.quality import QualityAnalyzer
from broadway.eda.segments import SegmentPlan
from broadway.eda.tests import FeatureTestAnalyzer

_PROVENANCE = {"artifact": "synthetic", "rows": "n/a", "seed": "0"}


def _plan(frame: pd.DataFrame, datetime_column: str | None = None) -> SegmentPlan:
    return SegmentPlan.build(frame, datetime_column, 20)


def _step(**overrides: object) -> EdaStep:
    cfg = load_config("eda", dataset="test", experiment="baseline")
    assert cfg.eda is not None
    return cfg.eda.model_copy(update=overrides)


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "group": rng.choice(["a", "b", "c"], n),
            "target": rng.normal(size=n),
        }
    )


def test_benjamini_hochberg_matches_the_published_reference_values() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]

    q_values = MultipleTestingCorrection("bh").adjust(p_values)

    assert q_values == pytest.approx(
        [0.010, 0.040, 0.084, 0.084, 0.084, 0.100, 0.1057, 0.216, 0.216, 0.216], abs=1e-4
    )
    assert MultipleTestingCorrection("none").adjust([0.01, 0.2]) == [0.01, 0.2]


def test_test_choice_follows_dtype_and_shape() -> None:
    step = _step(permutations=20, max_features=3)
    frame = _frame(600)
    frame["signal"] = frame["target"] * 2.0 + frame["signal"] * 0.1

    plan = _plan(frame)
    regression = FeatureTestAnalyzer(step, TaskType.REGRESSION).analyze(
        frame, "target", _PROVENANCE, plan
    )
    chosen = {row.feature: row.test for row in regression.features}
    assert chosen["signal"] == "spearman"
    assert chosen["group"] in {"anova", "kruskal"}

    classification = frame.assign(
        flag=(frame["target"] > frame["target"].median()).astype(int)
    )
    binary = FeatureTestAnalyzer(step, TaskType.CLASSIFICATION).analyze(
        classification, "flag", _PROVENANCE, _plan(classification)
    )
    binary_choice = {row.feature: row.test for row in binary.features}
    assert binary_choice["signal"] in {"welch", "mann_whitney"}
    assert binary_choice["group"] == "chi2"


def test_categorical_test_survives_a_top_code_in_one_target_class() -> None:
    """Regression: reserve the full contingency shape before reshape.

    The largest feature code can co-occur with only one target class (a city
    whose last value has no bad reviews), which made ``bincount`` return fewer
    cells than the declared shape and the reshape fail.
    """
    step = _step(permutations=20, max_features=2)
    frame = pd.DataFrame({"city": ["a", "b", "c", "c"], "flag": [0, 0, 0, 1]})
    bundle = FeatureTestAnalyzer(step, TaskType.CLASSIFICATION).analyze(
        frame, "flag", _PROVENANCE, _plan(frame)
    )
    assert [row.feature for row in bundle.features] == ["city"]


def test_time_and_single_group_columns_are_skipped_not_tested() -> None:
    step = _step(permutations=20)
    frame = _frame(300).assign(
        when=pd.to_datetime(["2024-01-01"] * 300),
        elapsed=pd.to_timedelta(["1h"] * 300),
        constant="a",
        ready=np.arange(300) % 2 == 0,
    )

    bundle = FeatureTestAnalyzer(step, TaskType.REGRESSION).analyze(
        frame, "target", _PROVENANCE, _plan(frame)
    )

    tested = {row.feature for row in bundle.features}
    assert {"signal", "noise", "group", "ready"} <= tested
    assert {"when", "elapsed", "constant"} <= set(bundle.skipped)
    assert next(row for row in bundle.features if row.feature == "ready").test in {
        "anova",
        "kruskal",
    }


def test_a_frame_with_no_testable_feature_records_why() -> None:
    step = _step(permutations=20)
    frame = pd.DataFrame(
        {"when": pd.to_datetime(["2024-01-01", "2024-01-02"]), "target": [1.0, 2.0]}
    )

    bundle = FeatureTestAnalyzer(step, TaskType.REGRESSION).analyze(
        frame, "target", _PROVENANCE, _plan(frame)
    )
    correlation = CorrelationAnalyzer(step).analyze(
        frame.assign(label=["a", "b"]), "label"
    )

    assert bundle.features == [] and bundle.summary is None
    numeric_target = CorrelationAnalyzer(step).analyze(frame, "target")

    assert bundle.reason is not None and "no testable feature column" in bundle.reason
    assert correlation.ranking == []
    assert correlation.reason is not None and "not numeric" in correlation.reason
    assert numeric_target.reason is not None and "no numeric feature" in numeric_target.reason


def test_a_significant_but_trivial_association_is_no_action() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    frame = pd.DataFrame(
        {"weak": rng.normal(size=n), "target": rng.normal(size=n)}
    )
    frame["target"] = frame["target"] + 0.08 * frame["weak"]

    step = _step(min_effect=0.3, permutations=20)
    bundle = FeatureTestAnalyzer(step, TaskType.REGRESSION).analyze(
        frame, "target", _PROVENANCE, _plan(frame)
    )

    row = bundle.features[0]
    assert row.q_value < 0.05
    assert row.significant is True
    assert row.effect < 0.3
    assert row.verdict == "no_action"
    assert bundle.summary is not None and bundle.summary.verdict == "no_action"


def test_outlier_share_against_the_configured_maximum() -> None:
    rng = np.random.default_rng(11)
    n = 1000
    frame = pd.DataFrame(
        {
            "extreme": np.where(np.arange(n) < n // 5, 100.0, rng.normal(size=n)),
            "clean": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )
    quality: QualityResult = QualityAnalyzer(1.5, 3.0, 0.2).analyze(
        frame, TaskType.REGRESSION, "target"
    )

    result = OutlierDecisionAnalyzer(
        _step(outlier_max_share=0.05, permutations=20)
    ).analyze(quality, frame, _PROVENANCE, _plan(frame))

    shares = {item.column: item for item in result.columns}
    assert shares["extreme"].iqr_share == pytest.approx(0.2)
    assert shares["extreme"].n == n
    assert shares["extreme"].verdict == "act"
    assert shares["clean"].verdict == "no_action"
    assert result.decisions[0].null_threshold is not None


def test_correlation_ranks_the_target_and_keeps_pairwise_counts() -> None:
    rng = np.random.default_rng(3)
    n = 500
    frame = pd.DataFrame(
        {
            "driving": rng.normal(size=n),
            "unrelated": rng.normal(size=n),
            "constant": np.full(n, 5.0),
            "sparse": [1.0] + [np.nan] * (n - 1),
            "target": rng.normal(size=n),
        }
    )
    frame["target"] = frame["target"] + 3.0 * frame["driving"]
    frame.loc[::10, "unrelated"] = np.nan

    result = CorrelationAnalyzer(_step(max_features=3)).analyze(frame, "target")

    ranked = [item.feature for item in result.ranking if item.feature in {"driving", "unrelated"}]
    assert ranked == ["driving", "unrelated"]
    assert result.ranking[0].value == pytest.approx(0.95, abs=0.05)
    degenerate = {item.feature: item for item in result.ranking}
    assert degenerate["constant"].value == 0.0 and degenerate["constant"].p_value == 1.0
    assert degenerate["sparse"].n == 1 and degenerate["sparse"].value == 0.0
    pair = next(pair for pair in result.pairs if {pair.left, pair.right} == {"unrelated", "target"})
    assert pair.n == n - len(frame.loc[::10])
    assert pair.pearson is not None and pair.spearman_q is not None
    constant_pair = next(
        pair for pair in result.pairs if {pair.left, pair.right} == {"constant", "driving"}
    )
    assert constant_pair.pearson is None and constant_pair.spearman is None
    assert len(result.pearson) == len(result.spearman) == len(result.columns)

    pearson = CorrelationAnalyzer(
        _step(max_features=3, correlation_method="pearson")
    ).analyze(frame, "target")
    assert pearson.method == "pearson" and pearson.ranking[0].method == "pearson"


def test_eda_step_publishes_every_decision_artifact(tmp_path: Path) -> None:
    cfg = load_config("eda", dataset="test", experiment="baseline")
    assert cfg.eda is not None and cfg.etl is not None
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)}),
            "eda": cfg.eda.model_copy(
                update={
                    "sample_size": 0,
                    "output_dir": str(tmp_path / "reports"),
                    "output_file": "eda.html",
                    "permutations": 20,
                    "max_features": 2,
                    "min_sample_size": 50,
                }
            ),
        }
    )
    frame = _frame(400)
    frame["target"] = frame["target"] + 2.0 * frame["signal"]
    frame["extreme"] = np.where(np.arange(400) < 80, 100.0, frame["noise"])
    frame["when"] = pd.date_range("2024-01-01", periods=400, freq="h")
    processed = tmp_path / cfg.environment.processed_subdir
    processed.mkdir(parents=True)
    frame.to_parquet(processed / cfg.etl.train_file, index=False)
    frame.iloc[:50].to_parquet(processed / cfg.etl.val_file, index=False)
    frame.to_parquet(processed / "test_canonical.parquet", index=False)

    module.run(cfg)

    reports = tmp_path / "reports"
    for name in (
        "eda.html",
        "eda_effects.csv",
        "eda_correlations.csv",
        "eda_outliers.csv",
        "eda_effects.png",
        "eda_correlations.png",
        "eda_outliers.png",
        "eda_decision.json",
        "eda_feature_decisions.json",
        "eda_comparison.csv",
    ):
        artifact = reports / name
        assert artifact.exists(), name
        assert artifact.stat().st_size > 0, name

    effects = pd.read_csv(reports / "eda_effects.csv")
    assert list(effects.columns[:4]) == ["feature", "feature_kind", "target_kind", "test"]
    assert effects.loc[effects["feature"] == "signal", "verdict"].item() == "act"
    decision = json.loads((reports / "eda_decision.json").read_text(encoding="utf-8"))
    assert decision["verdict"] == "act"
    assert decision["metric"] == "association_signal"
    assert decision["evidence"]["provenance"]["rows"] == "450"
    sidecar = json.loads(
        (reports / "eda_feature_decisions.json").read_text(encoding="utf-8")
    )
    assert sidecar["summary"]["metric"] == "association_signal"
    assert sidecar["comparison"] is not None
    assert sidecar["outliers"]
    html = (reports / "eda.html").read_text(encoding="utf-8")
    assert "Decisions" in html and "Correlation" in html
    assert html.count("data:image/png;base64,") >= 4

    figure = (reports / "eda_effects.png").read_bytes()
    module.run(cfg)
    assert (reports / "eda_effects.png").read_bytes() == figure
