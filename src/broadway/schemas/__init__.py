"""Schema-module registry: ``schema_contract`` value -> column set (Decision 6).

Each loader path names the schema module describing what it ACTUALLY emits;
the config-load cross-check validates recipe columns against the module the
experiment's ``data_source.schema_contract`` points to. Unknown contracts fail
loud — a typo must never silently bind to nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from broadway.config.schema import DatasetContract
from broadway.schemas.joined import joined_schema_columns

if TYPE_CHECKING:
    from broadway.features.plan import ColumnPlan


def _engineered_schema_columns(
    dataset: DatasetContract, plan: ColumnPlan | None
) -> frozenset[str]:
    """The experiment's resolved plan is the engineered feature surface."""
    if plan is None:
        raise ValueError(
            "schema_contract 'engineered' requires the experiment's resolved "
            "column plan — the declared surface resolves through "
            "ColumnPlan.from_config(dataset, experiment)"
        )
    return frozenset(plan.names)


_SCHEMA_MODULES: dict[str, Callable[..., frozenset[str]]] = {
    "raw": lambda dataset, plan=None: frozenset(dataset.columns),
    "joined": lambda dataset, plan=None: joined_schema_columns(dataset),
    "engineered": _engineered_schema_columns,
}


def schema_columns(
    schema_contract: str,
    dataset: DatasetContract,
    plan: ColumnPlan | None = None,
) -> frozenset[str]:
    """Resolve the bound schema module's column set; unknown values fail loud.

    ``raw``/``joined`` ignore ``plan``; ``engineered`` requires the experiment's
    resolved plan and raises when it is None — without it there is no declared
    surface.
    """
    resolver = _SCHEMA_MODULES.get(schema_contract)
    if resolver is None:
        raise ValueError(
            f"unknown schema_contract '{schema_contract}' "
            f"(supported: {sorted(_SCHEMA_MODULES)})"
        )
    return resolver(dataset, plan)
