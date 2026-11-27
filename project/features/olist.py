"""Feature plan for olist — every FEATURE-role column, leakage excluded by role.

The dataset contract is the keep-set source, so the plan cannot silently drift
into a post-purchase column: only ``ColumnRole.FEATURE`` entries enter, numeric
ones as numeric sources and categorical ones through the existing frequency
encoder. The context columns (delivery_days, late, review_score, timestamps)
and every id are ``role: ignore`` in ``project/config/dataset/olist.yaml`` and
therefore never appear here.

``feature_plan: project.features.olist:plan`` in the experiment config is the
only reference; this module is the single keep-set.
"""

from __future__ import annotations

from broadway.config.schema import ColumnRole, DatasetContract
from broadway.contracts.pandera import is_numeric_dtype
from broadway.features.plan import ColumnOp, ColumnPlan


def _frequency(source: str) -> ColumnOp:
    """One categorical source as a frequency-encoded numeric column."""
    return ColumnOp(
        op="encode",
        name=f"{source}_frequency_enc",
        dtype="float64",
        sources=(source,),
        encoder="frequency",
        params={},
    )


def plan(dataset: DatasetContract) -> ColumnPlan:
    """Build the Olist keep-set from the contract's feature roles."""
    numeric: list[str] = []
    categorical: list[str] = []
    for name, column in dataset.columns.items():
        if column.role is not ColumnRole.FEATURE:
            continue
        (numeric if is_numeric_dtype(column.dtype) else categorical).append(name)
    return ColumnPlan.from_declaration(
        dataset,
        include=numeric,
        encodings=[_frequency(name) for name in categorical],
    )
