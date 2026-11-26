"""The column plan owned by one named-sample definition.

A sample definition declares its own projected surface (``columns``) and its
own derived columns (``derived``). That declaration materialises as its own
``ColumnPlan`` instance — never an experiment's resolved plan — so
``ColumnPlan.apply`` stays the single place a sample column is selected,
derived, added, or dropped, exactly as it is for an experiment.
"""

from __future__ import annotations

import pandas as pd

from broadway.features.builders import BUILDERS, builder_dtype
from broadway.features.plan import ColumnOp, ColumnPlan
from broadway.lineage.models import DerivedSpec, SampleSpec


class SampleColumnPlanFactory:
    """Build the ``ColumnPlan`` for one sample definition."""

    @classmethod
    def build(cls, spec: SampleSpec, frame: pd.DataFrame) -> ColumnPlan:
        """Materialise ``spec`` against the sample's source ``frame``.

        Declared columns are sources in declaration order; derived columns
        follow in declaration order. An undeclared projection selects every
        source column, matching a sample that filters rather than projects.
        """
        selected = cls._selected(spec, frame)
        ops = [cls._source(frame, name) for name in selected]
        ops.extend(cls._derived(feature) for feature in (spec.derived or []))
        return ColumnPlan(
            target=None,
            datetime_column=None,
            columns=tuple(ops),
            claimed=frozenset(selected),
        )

    @staticmethod
    def _selected(spec: SampleSpec, frame: pd.DataFrame) -> list[str]:
        """The declared projection, failing loud on a column the source lacks."""
        selected = list(spec.columns) if spec.columns is not None else list(frame.columns)
        missing = [name for name in selected if name not in frame.columns]
        if missing:
            raise ValueError(
                f"sample '{spec.name}': declared column(s) missing from source: {missing}"
            )
        return selected

    @staticmethod
    def _source(frame: pd.DataFrame, name: str) -> ColumnOp:
        return ColumnOp(op="source", name=name, dtype=str(frame[name].dtype))

    @staticmethod
    def _derived(feature: DerivedSpec) -> ColumnOp:
        """One declared derivation; its role columns become the op's sources.

        An empty role mapping is legitimate: the builder resolves its own
        inputs and falls back to its own defaults.
        """
        if feature.formula not in BUILDERS:
            raise ValueError(
                f"unknown derived formula '{feature.formula}' "
                f"(registry: {sorted(BUILDERS)})"
            )
        return ColumnOp(
            op="derive",
            name=feature.name,
            dtype=builder_dtype(feature.formula),
            sources=tuple(dict.fromkeys(feature.columns.values())),
            func=feature.formula,
            params={"columns": dict(feature.columns)},
        )
