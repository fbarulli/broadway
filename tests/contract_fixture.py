"""Derive synthetic fixture shapes from the demo dataset contract.

Column names come from ``configs/dataset/test.yaml`` (loaded via
``DatasetContract``), never from literals — rename a demo column in the
config and contract-bound tests follow automatically ("derive, don't
maintain"). Tests that build frames with values tied to assertions use the
*name* helpers here and keep their own value shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from broadway.config.schema import ColumnRole, DatasetContract


def feature_columns(contract: DatasetContract) -> list[str]:
    """Feature column names in contract order (excludes target/datetime)."""
    return [n for n, c in contract.columns.items() if c.role == ColumnRole.FEATURE]


def target_column(contract: DatasetContract) -> str:
    return contract.target


def categorical_column(contract: DatasetContract) -> str | None:
    """The object-typed feature column (the analysis group column in the demo)."""
    return next(
        (
            n
            for n, c in contract.columns.items()
            if c.role == ColumnRole.FEATURE and c.dtype == "object"
        ),
        None,
    )


def numeric_feature_columns(contract: DatasetContract) -> list[str]:
    """Numeric feature columns (excludes the object/group column and target)."""
    return [
        n
        for n, c in contract.columns.items()
        if c.role == ColumnRole.FEATURE and c.dtype != "object"
    ]


def frame_slots(contract: DatasetContract) -> tuple[list[str], str]:
    """(feature names in contract order, target name).

    Tests that build frames with assertion-tied values key the value lists by
    feature *position* (slots[0], slots[1], ...) instead of literal names, so
    a demo rename flows through automatically.
    """
    return feature_columns(contract), contract.target


def make_contract_frame(
    contract: DatasetContract, n: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Deterministic synthetic frame matching the contract's names/dtypes.

    The target is a noisy linear combination of the first two numeric
    features, so a linear model fits with low error (used by pipeline
    integration tests).
    """
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    numerics = numeric_feature_columns(contract)
    for i, name in enumerate(feature_columns(contract)):
        if contract.columns[name].dtype == "object":
            data[name] = rng.choice(["A", "B", "C", "D"], n)
        elif i < len(numerics):
            data[name] = np.arange(1, n + 1) * (i + 1) % 200
        else:
            data[name] = rng.integers(1, 200, n)
    signal = np.zeros(n)
    for i, name in enumerate(numerics[:2]):
        signal += (i + 2) * data[name]
    noise = rng.normal(0.0, 5.0, n)
    data[contract.target] = np.maximum(signal + noise, 1.0).astype(int)
    return pd.DataFrame(data)
