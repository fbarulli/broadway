"""Schema-module registry: ``schema_contract`` value -> column set (Decision 6).

Each loader path names the schema module describing what it ACTUALLY emits;
the config-load cross-check validates recipe columns against the module the
experiment's ``data_source.schema_contract`` points to. Unknown contracts fail
loud — a typo must never silently bind to nothing.
"""

from __future__ import annotations

from collections.abc import Callable

from broadway.config.schema import DatasetContract, FeatureConfig
from broadway.schemas.joined import joined_schema_columns


def _engineered_schema_columns(
    dataset: DatasetContract, features: FeatureConfig | None
) -> frozenset[str]:
    """Features-step output surface (Decision 6 extension: the contract may
    name features-step output). Lazy import — the builders registry must not
    sit on the package-import path of every schemas consumer."""
    from broadway.features.generic import build_generic_feature_specs

    if features is None:
        raise ValueError(
            "schema_contract 'engineered' requires the experiment's features "
            "config — the declared surface resolves through "
            "build_generic_feature_specs(dataset, features)"
        )
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
    """Resolve the bound schema module's column set; unknown values fail loud.

    ``raw``/``joined`` ignore ``features``; ``engineered`` resolves through the
    experiment's features config and raises when ``features`` is None — without
    it there is no declared surface.
    """
    resolver = _SCHEMA_MODULES.get(schema_contract)
    if resolver is None:
        raise ValueError(
            f"unknown schema_contract '{schema_contract}' "
            f"(supported: {sorted(_SCHEMA_MODULES)})"
        )
    return resolver(dataset, features)
