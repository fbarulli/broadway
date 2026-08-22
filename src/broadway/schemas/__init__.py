"""Schema-module registry: ``schema_contract`` value -> column set (Decision 6).

Each loader path names the schema module describing what it ACTUALLY emits;
the config-load cross-check validates recipe columns against the module the
experiment's ``data_source.schema_contract`` points to. Unknown contracts fail
loud — a typo must never silently bind to nothing.
"""

from __future__ import annotations

from collections.abc import Callable

from broadway.config.schema import DatasetContract
from broadway.schemas.joined import joined_schema_columns

_SCHEMA_MODULES: dict[str, Callable[[DatasetContract], frozenset[str]]] = {
    "raw": lambda dataset: frozenset(dataset.columns),
    "joined": joined_schema_columns,
}


def schema_columns(schema_contract: str, dataset: DatasetContract) -> frozenset[str]:
    """Resolve the bound schema module's column set; unknown values fail loud."""
    resolver = _SCHEMA_MODULES.get(schema_contract)
    if resolver is None:
        raise ValueError(
            f"unknown schema_contract '{schema_contract}' "
            f"(supported: {sorted(_SCHEMA_MODULES)})"
        )
    return resolver(dataset)
