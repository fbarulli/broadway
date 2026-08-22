"""Schema-module registry: ``schema_contract`` value -> column set (Decision 6)."""

from __future__ import annotations

from broadway.config.schema import DatasetContract
from broadway.schemas.joined import joined_schema_columns


def schema_columns(schema_contract: str, dataset: DatasetContract) -> frozenset[str]:
    """Resolve the bound schema module's column set; unknown values fail loud."""
    if schema_contract == "raw":
        return frozenset(dataset.columns)
    if schema_contract == "joined":
        return joined_schema_columns(dataset)
    raise ValueError(
        f"unknown schema_contract '{schema_contract}' (supported: raw, joined)"
    )
