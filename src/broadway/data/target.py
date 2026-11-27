"""The modelling target after the split: binary classifications read as integers.

The ETL leaves a 0/1 target as ``float64`` because the null-target rows are
dropped after the numeric coercion pass, so the surviving column never gets the
integer cast. Modelling reads the split frames through this owner and casts the
binary target to ``int64``; the persisted source artifacts keep their declared
dtype (data is immutable — the cast is a read-side model contract).
"""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import PipelineConfig, TaskType


class TargetCaster:
    """Cast a binary classification target to ``int64`` on a split frame."""

    def __init__(self, task: TaskType, target: str) -> None:
        self._task = task
        self._target = target

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> TargetCaster:
        if cfg.dataset is None:
            raise ValueError("the target cast requires a dataset config")
        return cls(cfg.dataset.task, cfg.dataset.target)

    def cast(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with the binary target as int (regression unchanged)."""
        if self._task is not TaskType.CLASSIFICATION:
            return frame
        if self._target not in frame.columns:
            raise ValueError(f"target column '{self._target}' is missing from the frame")
        values = frame[self._target]
        if values.isna().any():
            raise ValueError(
                f"target column '{self._target}' carries nulls — drop them at the "
                "ETL boundary before casting the split"
            )
        observed = set(values.unique().tolist())
        if not observed <= {0, 1}:
            raise ValueError(
                f"target column '{self._target}' is not binary 0/1 "
                f"(observed: {sorted(observed)[:5]})"
            )
        return frame.assign(**{self._target: values.astype("int64")})
