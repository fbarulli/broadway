"""Apply a dataset contract's declared column renames to a raw frame.

A raw source can carry labels the contract cannot use — a trailing space in a
CSV header, a vendor spelling. ``DatasetContract.column_renames`` declares the
raw label -> contract label mapping once; every reader that materializes the
raw frame (the loader, the profiler, the sampler) applies it through this one
class, so downstream steps always see the columns the contract declares.
"""

from __future__ import annotations

import pandas as pd

from broadway.config.schema import DatasetContract


class ColumnNormalizer:
    """The single applier of a contract's read-time column renames."""

    def __init__(self, renames: dict[str, str]) -> None:
        self._renames = dict(renames)
        self._sources = {target: source for source, target in self._renames.items()}

    @classmethod
    def from_contract(cls, dataset: DatasetContract) -> ColumnNormalizer:
        """Build the normalizer a contract's frames must pass through."""
        return cls(dataset.column_renames)

    @property
    def renames(self) -> dict[str, str]:
        return dict(self._renames)

    def source_name(self, contract_name: str) -> str:
        """The raw label a contract column is read from (itself when not renamed).

        Needed by readers that prune columns before the frame is normalized —
        a projection must name the file's labels, not the contract's.
        """
        return self._sources.get(contract_name, contract_name)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` with the declared renames applied (identity when none).

        A declared source label that is absent fails loud: a silent no-op would
        leave the frame on raw names and every downstream contract check would
        be reading a different schema than the one declared.
        """
        if not self._renames:
            return frame
        missing = sorted(name for name in self._renames if name not in frame.columns)
        if missing:
            raise KeyError(
                f"dataset contract declares renames for absent source column(s) "
                f"{missing}; frame carries {list(frame.columns)}"
            )
        return frame.rename(columns=self._renames)
