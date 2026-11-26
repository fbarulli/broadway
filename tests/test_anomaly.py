"""Anomaly detection: PyOD strategies plus the statistical MAD/IQR baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.unsupervised.anomaly import (
    AnomalyConfig,
    AnomalyDetection,
    DetectorKind,
)


def _synthetic() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(300, 3))
    outliers = rng.normal(loc=9.0, scale=0.4, size=(20, 3))
    features = pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c"])
    # A constant column has zero MAD and zero IQR; the baseline keeps its
    # score at 0.0 for it instead of dividing by zero.
    features["constant"] = 1.0
    labels = np.array([0] * 300 + [1] * 20)
    return features, labels


def _config() -> AnomalyConfig:
    return AnomalyConfig(
        detectors=[DetectorKind.IFOREST, DetectorKind.MAD, DetectorKind.IQR],
        baseline=DetectorKind.MAD,
        contamination=0.05,
        # Deliberately out of order: the config sorts it ascending, which is
        # the sensitivity table's canonical order.
        sensitivity_contaminations=[0.05, 0.01],
        random_state=7,
    )


def test_anomaly_flags_synthetic_outlier_cluster_and_orders_sensitivity() -> None:
    features, labels = _synthetic()
    config = _config()
    assert config.sensitivity_contaminations == [0.01, 0.05]
    report = AnomalyDetection(config).detect(features, labels)

    assert report.n_obs == 320
    outcomes = {outcome.detector: outcome for outcome in report.outcomes}
    assert set(outcomes) == {DetectorKind.IFOREST, DetectorKind.MAD, DetectorKind.IQR}
    iforest = outcomes[DetectorKind.IFOREST]
    assert iforest.n_flagged == 16  # round(0.05 * 320)
    assert len(iforest.scores) == 320 == len(iforest.flags)
    assert iforest.quality is not None
    assert iforest.quality.n_anomalies == 20
    assert iforest.quality.recall > 0.5
    assert iforest.quality.roc_auc > 0.9
    assert sum(outcomes[DetectorKind.MAD].flags) > 10
    assert sum(outcomes[DetectorKind.IQR].flags) > 10

    assert [(row.detector, row.contamination) for row in report.sensitivity] == [
        (DetectorKind.IFOREST, 0.01),
        (DetectorKind.IFOREST, 0.05),
        (DetectorKind.MAD, 0.01),
        (DetectorKind.MAD, 0.05),
        (DetectorKind.IQR, 0.01),
        (DetectorKind.IQR, 0.05),
    ]
    for row in report.sensitivity:
        assert row.n_flagged == max(1, round(row.contamination * report.n_obs))
        assert 0 <= row.both_flagged <= row.n_flagged
        assert 0 <= row.agreement <= report.n_obs

    unlabeled = AnomalyDetection(config).detect(features)
    assert all(outcome.quality is None for outcome in unlabeled.outcomes)

    with pytest.raises(ValueError, match="NaN"):
        AnomalyDetection(config).detect(features.assign(a=np.nan))
    with pytest.raises(ValueError, match="baseline"):
        AnomalyConfig(
            detectors=[DetectorKind.IFOREST],
            baseline=DetectorKind.MAD,
            contamination=0.05,
            sensitivity_contaminations=[0.01],
        )


@pytest.mark.parametrize(
    "kind",
    [
        DetectorKind.IFOREST,
        DetectorKind.LOF,
        DetectorKind.KNN,
        DetectorKind.HBOS,
        DetectorKind.ECOD,
        DetectorKind.COPOD,
    ],
)
def test_every_pyod_strategy_scores_the_same_frame(kind: DetectorKind) -> None:
    features, _ = _synthetic()
    config = AnomalyConfig(
        detectors=[kind, DetectorKind.MAD],
        baseline=DetectorKind.MAD,
        contamination=0.05,
        sensitivity_contaminations=[0.05],
        n_neighbors=5,
        random_state=7,
    )
    report = AnomalyDetection(config).detect(features.head(80))
    outcome = report.outcomes[0]
    assert outcome.detector is kind
    assert outcome.n_flagged == 4  # round(0.05 * 80)
    assert np.all(np.isfinite(outcome.scores))
