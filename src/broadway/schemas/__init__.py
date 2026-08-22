"""Schema-module registry: ``schema_contract`` value -> column set (Decision 6).

Each loader path names the schema module describing what it ACTUALLY emits;
the config-load cross-check validates recipe columns against the module the
experiment's ``data_source.schema_contract`` points to. Unknown contracts fail
loud — a typo must never silently bind to nothing. ``engineered`` (Decision 5
repoint mapping) names the features-step output surface: the config-declared
include + derived + encoding columns (``build_generic_feature_specs``).
"""

from __future__ import annotations

from collections.abc import Callable

from broadway.config.schema import DatasetContract, FeatureConfig
from broadway.schemas.joined import joined_schema_columns


def _engineered_schema_columns(dataset: DatasetContract, features: FeatureConfig) -> frozenset[str]:
    """Engineered-surface column set: include + derived + encoding outputs
    (the generic engineered schema, Decision 5 repoint mapping). Lazy import
    keeps the module graph acyclic."""
    from broadway.features.generic import build_generic_feature_specs

    return frozenset(build_generic_feature_specs(dataset, features))


_SCHEMA_MODULES: dict[str, Callable[..., frozenset[str]]] = {
    "raw": lambda dataset, features=None: frozenset(dataset.columns),
    "joined": lambda dataset, features=None: joined_schema_columns(dataset),
    "engineered": _engineered_schema_columns,
}


def schema_columns(
    schema_contract: str,
    dataset: DatasetContract,
    features: FeatureConfig | None = None,
) -> frozenset[str]:
    """Resolve the bound schema module's column set; unknown values fail loud."""
    resolver = _SCHEMA_MODULES.get(schema_contract)
    if resolver is None:
        raise ValueError(
            f"unknown schema_contract '{schema_contract}' "
            f"(supported: {sorted(_SCHEMA_MODULES)})"
        )
    if schema_contract == "engineered" and features is None:
        raise ValueError(
            "schema_contract 'engineered' requires the experiment's features config"
        )
    return resolver(dataset, features)
