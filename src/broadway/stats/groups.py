"""Declared-group construction shared by stats, timeline, and baseline surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_declared_groups(
    df: pd.DataFrame,
    source_group_column: str,
    group_values: list[str],
    target: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build one target array per DECLARED group value, plus the absent list.

    Every entry in ``group_values`` appears in the returned mapping, in declared
    order; a value with no usable observations — either no matching rows in
    ``df[source_group_column]`` or rows whose ``target`` is all-NaN — maps to a
    size-0 array. The returned absent list is sorted.

    Invariant: a non-empty absent list MUST be treated as error by the caller;
    this function itself never raises. Stats and baseline raise the single
    standard vocabulary "declared groups absent from data: [...]" while the
    walkthrough keeps the include-all arrays for its own loud downstream
    guards (validate_groups on size-0 entries).
    """
    groups: dict[str, np.ndarray] = {
        g: df[df[source_group_column] == g][target].dropna().to_numpy()
        for g in group_values
    }
    absent = sorted(g for g, values in groups.items() if values.size == 0)
    return groups, absent
