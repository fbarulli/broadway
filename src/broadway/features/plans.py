"""Code-owned column plans for the shipped demo experiments.

Every experiment names one of these callables in ``feature_plan`` as
``module:callable``; the loader resolves it against the dataset contract. The
plan is the single keep-set — there is no parallel YAML column list.
"""

from __future__ import annotations

from broadway.config.schema import ColumnRole, DatasetContract, DerivedFeature
from broadway.features.plan import ColumnOp, ColumnPlan


def _encoded(
    name: str, sources: tuple[str, ...], encoder: str, **params: object
) -> ColumnOp:
    """One encoder output declared in the plan (read-side execution)."""
    return ColumnOp(
        op="encode",
        name=name,
        dtype="float64",
        sources=sources,
        encoder=encoder,
        params=dict(params),
    )


def baseline(dataset: DatasetContract) -> ColumnPlan:
    """Demo baseline: the two declared numeric placeholders."""
    return ColumnPlan.from_declaration(dataset, include=["feature_1", "feature_2"])


def engineered(dataset: DatasetContract) -> ColumnPlan:
    """Demo engineered: source copy plus a frequency encoding."""
    return ColumnPlan.from_declaration(
        dataset,
        include=["feature_1", "feature_2"],
        derived=[
            DerivedFeature(
                name="engineered_feature_1", func="source_copy", source="feature_2"
            )
        ],
        encodings=[
            _encoded("feature_3_freq_enc", ("feature_3",), "frequency")
        ],
    )


def hyperopt(dataset: DatasetContract) -> ColumnPlan:
    """Demo HPO surface: the placeholders plus a frequency encoding."""
    return ColumnPlan.from_declaration(
        dataset,
        include=["feature_1", "feature_2"],
        encodings=[
            _encoded("feature_3_freq_enc", ("feature_3",), "frequency")
        ],
    )


def contract_features(dataset: DatasetContract) -> ColumnPlan:
    """Every contract column declared with the ``feature`` role, in contract order.

    The generic surface for callers (and fixtures) that have no engineered
    features: the contract itself is the keep-set.
    """
    return ColumnPlan.from_declaration(
        dataset,
        include=[
            name
            for name, column in dataset.columns.items()
            if column.role == ColumnRole.FEATURE
        ],
    )


_NUMERIC_DTYPE_PREFIXES = ("int", "uint", "float", "bool")


def contract_numeric_features(dataset: DatasetContract) -> ColumnPlan:
    """Numeric feature-role contract columns — the categorical-free surface."""
    return ColumnPlan.from_declaration(
        dataset,
        include=[
            name
            for name, column in dataset.columns.items()
            if column.role == ColumnRole.FEATURE
            and column.dtype.startswith(_NUMERIC_DTYPE_PREFIXES)
        ],
    )
