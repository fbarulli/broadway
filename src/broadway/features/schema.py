"""Feature definitions and schema generation.

``FeatureSpec`` is the generic feature contract; ``build_engineered_schema``
generates a Pandera schema from a registry of specs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandera as pa

from broadway.contracts.pandera import pandera_dtype


@dataclass(frozen=True)
class FeatureSpec:
    """Structure-only contract for a single engineered feature."""

    name: str
    dtype: str
    nullable: bool = False


def build_engineered_schema(specs: dict[str, FeatureSpec]) -> pa.DataFrameSchema:
    """Build a structure-only Pandera schema from a ``FeatureSpec`` registry."""
    return pa.DataFrameSchema(
        {
            spec.name: pa.Column(
                pandera_dtype(spec.dtype),
                nullable=spec.nullable,
            )
            for spec in specs.values()
        }
    )
