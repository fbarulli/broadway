"""Unsupervised anomaly detection: a Strategy over PyOD plus a statistical baseline.

Every configured detector is one interchangeable scoring Strategy behind the
same interface — ``fit`` the feature matrix, read one anomaly score per row,
flag the most anomalous share. PyOD supplies the model-based detectors
(IForest, LOF, KNN, HBOS, ECOD, COPOD; installed through the ``anomaly``
extra) and ``broadway.stats.robust`` supplies the statistical MAD/IQR baseline,
so the two families are compared on the same rows and the same units.

The report carries the per-row score and flag for each detector, precision /
recall / ROC-AUC when truth labels are supplied, and a contamination-
sensitivity table: every detector re-thresholded at each configured
contamination, with its agreement count against the baseline. Nothing here
gates a pipeline step — it is a measurement surface.
"""

from __future__ import annotations

import importlib
import logging
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from broadway.evaluate.metrics import METRIC_DECIMALS
from broadway.observability.profiling import timed
from broadway.stats.robust import iqr_excursions, mad_zscores

logger = logging.getLogger(__name__)


class DetectorKind(StrEnum):
    """The detector vocabulary; the name is the config value."""

    IFOREST = "iforest"
    LOF = "lof"
    KNN = "knn"
    HBOS = "hbos"
    ECOD = "ecod"
    COPOD = "copod"
    MAD = "mad"
    IQR = "iqr"


# The model-based family is resolved lazily (importing pyod pulls numba), so
# the platform only pays the import when a PyOD strategy is requested.
_PYOD_CLASSES: dict[DetectorKind, tuple[str, str]] = {
    DetectorKind.IFOREST: ("pyod.models.iforest", "IForest"),
    DetectorKind.LOF: ("pyod.models.lof", "LOF"),
    DetectorKind.KNN: ("pyod.models.knn", "KNN"),
    DetectorKind.HBOS: ("pyod.models.hbos", "HBOS"),
    DetectorKind.ECOD: ("pyod.models.ecod", "ECOD"),
    DetectorKind.COPOD: ("pyod.models.copod", "COPOD"),
}
_STATISTICAL: frozenset[DetectorKind] = frozenset({DetectorKind.MAD, DetectorKind.IQR})


class AnomalyConfig(BaseModel):
    """Boundary-validated anomaly-detection knobs.

    ``contamination`` is the share flagged in the per-row output; the
    ``sensitivity_contaminations`` grid drives the sensitivity table (kept
    sorted ascending so the table has one canonical order). ``mad_threshold``
    and ``iqr_multiplier`` are the statistical baseline's own cutoffs; the
    sensitivity table re-thresholds every detector by rank instead, so the
    families are compared at the same flagged share.
    """

    model_config = ConfigDict(extra="forbid")

    detectors: list[DetectorKind] = Field(min_length=1)
    baseline: DetectorKind = DetectorKind.MAD
    contamination: float = Field(gt=0.0, lt=0.5)
    sensitivity_contaminations: list[float] = Field(min_length=1)
    mad_threshold: float = Field(default=3.5, gt=0.0)
    iqr_multiplier: float = Field(default=1.5, gt=0.0)
    n_neighbors: int = Field(default=20, ge=1)
    random_state: int = 42

    @field_validator("sensitivity_contaminations")
    @classmethod
    def _sorted_contaminations(cls, values: list[float]) -> list[float]:
        if any(not 0.0 < value < 0.5 for value in values):
            raise ValueError(f"contamination values must be in (0, 0.5); got {values}")
        if len(set(values)) != len(values):
            raise ValueError(f"contamination values have duplicates: {values}")
        return sorted(values)

    @model_validator(mode="after")
    def _baseline_is_configured(self) -> AnomalyConfig:
        if self.baseline not in _STATISTICAL:
            raise ValueError(
                f"baseline must be a statistical detector (mad/iqr), got {self.baseline!r}"
            )
        if self.baseline not in self.detectors:
            raise ValueError(
                f"baseline {self.baseline!r} must be one of the configured detectors "
                f"{[kind.value for kind in self.detectors]}"
            )
        return self


class AnomalyQuality(BaseModel):
    """Labeled quality of one detector's flags (only when labels exist)."""

    model_config = ConfigDict(extra="forbid")

    n_labels: int
    n_anomalies: int
    n_flagged: int
    precision: float
    recall: float
    roc_auc: float


class DetectorOutcome(BaseModel):
    """One detector's per-row score and flag over the scored frame."""

    model_config = ConfigDict(extra="forbid")

    detector: DetectorKind
    n_flagged: int
    scores: list[float]
    flags: list[bool]
    quality: AnomalyQuality | None = None


class SensitivityRow(BaseModel):
    """One detector re-thresholded at one contamination, vs the baseline."""

    model_config = ConfigDict(extra="forbid")

    detector: DetectorKind
    contamination: float
    n_flagged: int
    agreement: int
    both_flagged: int
    agreement_rate: float


class AnomalyReport(BaseModel):
    """The detection result: per-row outcomes plus the sensitivity table."""

    model_config = ConfigDict(extra="forbid")

    n_obs: int
    contamination: float
    baseline: DetectorKind
    outcomes: list[DetectorOutcome]
    sensitivity: list[SensitivityRow]


def _top_fraction_flags(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Flag the ``contamination`` share of rows with the highest scores."""
    count = max(1, round(contamination * scores.size))
    flags = np.zeros(scores.size, dtype=bool)
    flags[np.argsort(-scores, kind="stable")[:count]] = True
    return flags


class AnomalyDetection:
    """Score a feature frame with each configured detector and report flags."""

    def __init__(self, config: AnomalyConfig) -> None:
        self._config = config

    def detect(self, features: pd.DataFrame, labels: np.ndarray | None = None) -> AnomalyReport:
        matrix = self._matrix(features)
        with timed() as elapsed:
            scores = {kind: self._scores(kind, matrix) for kind in self._config.detectors}
            flags = {
                kind: self._native_flags(kind, scores[kind]) for kind in self._config.detectors
            }
        outcomes = [
            DetectorOutcome(
                detector=kind,
                n_flagged=int(flags[kind].sum()),
                scores=[float(value) for value in scores[kind]],
                flags=[bool(value) for value in flags[kind]],
                quality=self._quality(scores[kind], flags[kind], labels),
            )
            for kind in self._config.detectors
        ]
        sensitivity = self._sensitivity(scores)
        logger.info(
            "anomaly: scored %d rows with %s at contamination %.3f in %.3fs",
            matrix.shape[0],
            [kind.value for kind in self._config.detectors],
            self._config.contamination,
            elapsed(),
        )
        return AnomalyReport(
            n_obs=int(matrix.shape[0]),
            contamination=self._config.contamination,
            baseline=self._config.baseline,
            outcomes=outcomes,
            sensitivity=sensitivity,
        )

    @staticmethod
    def _matrix(features: pd.DataFrame) -> np.ndarray:
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("anomaly detection needs a non-empty numeric feature frame")
        matrix = features.to_numpy(dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "anomaly features contain NaN or infinite values; impute or drop "
                "those rows before scoring"
            )
        return matrix

    def _build_pyod_detector(self, kind: DetectorKind) -> Any:
        module_name, class_name = _PYOD_CLASSES[kind]
        detector_class = getattr(importlib.import_module(module_name), class_name)
        kwargs: dict[str, Any] = {"contamination": self._config.contamination}
        if kind in {DetectorKind.LOF, DetectorKind.KNN}:
            kwargs["n_neighbors"] = self._config.n_neighbors
        if kind is DetectorKind.IFOREST:
            kwargs["random_state"] = self._config.random_state
        return detector_class(**kwargs)

    def _scores(self, kind: DetectorKind, matrix: np.ndarray) -> np.ndarray:
        if kind is DetectorKind.MAD:
            columns = [mad_zscores(column) for column in matrix.T]
            return np.max(np.abs(np.column_stack(columns)), axis=1)
        if kind is DetectorKind.IQR:
            columns = [iqr_excursions(column) for column in matrix.T]
            return np.max(np.column_stack(columns), axis=1)
        detector = self._build_pyod_detector(kind)
        detector.fit(matrix)
        return np.asarray(detector.decision_function(matrix), dtype=float)

    def _native_flags(self, kind: DetectorKind, scores: np.ndarray) -> np.ndarray:
        if kind is DetectorKind.MAD:
            return scores > self._config.mad_threshold
        if kind is DetectorKind.IQR:
            return scores > self._config.iqr_multiplier
        return _top_fraction_flags(scores, self._config.contamination)

    def _quality(
        self,
        scores: np.ndarray,
        flags: np.ndarray,
        labels: np.ndarray | None,
    ) -> AnomalyQuality | None:
        if labels is None:
            return None
        truth = np.asarray(labels)
        if truth.shape != scores.shape:
            raise ValueError(
                f"labels must carry one entry per row, got {truth.shape} for {scores.size} rows"
            )
        if truth.dtype != bool:
            truth = truth.astype(float)
            if not np.all((truth == 0.0) | (truth == 1.0)):
                raise ValueError("labels must hold binary truth values (0/1 or bool)")
        anomalies = int(truth.sum())
        if anomalies in {0, truth.size}:
            raise ValueError(
                "roc_auc needs both anomaly and normal labels; labels are single-class"
            )
        return AnomalyQuality(
            n_labels=int(truth.size),
            n_anomalies=anomalies,
            n_flagged=int(flags.sum()),
            precision=round(float(precision_score(truth, flags, zero_division=0)), METRIC_DECIMALS),
            recall=round(float(recall_score(truth, flags, zero_division=0)), METRIC_DECIMALS),
            roc_auc=round(float(roc_auc_score(truth, scores)), METRIC_DECIMALS),
        )

    def _sensitivity(self, scores: dict[DetectorKind, np.ndarray]) -> list[SensitivityRow]:
        rows: list[SensitivityRow] = []
        for kind in self._config.detectors:
            for contamination in self._config.sensitivity_contaminations:
                flags = _top_fraction_flags(scores[kind], contamination)
                baseline = _top_fraction_flags(scores[self._config.baseline], contamination)
                agreement = int(np.count_nonzero(flags == baseline))
                rows.append(
                    SensitivityRow(
                        detector=kind,
                        contamination=contamination,
                        n_flagged=int(flags.sum()),
                        agreement=agreement,
                        both_flagged=int(np.count_nonzero(flags & baseline)),
                        agreement_rate=round(agreement / scores[kind].size, METRIC_DECIMALS),
                    )
                )
        return rows
