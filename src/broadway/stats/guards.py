"""Group validity guards — fail loudly on empty / non-finite / zero-variance groups."""

from __future__ import annotations

import numpy as np


def validate_groups(groups: dict[str, np.ndarray]) -> list[str]:
    if len(groups) < 2:
        raise ValueError(f"at least two groups required, got {len(groups)}")
    warnings: list[str] = []
    all_zero_variance = True
    for name, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            raise ValueError(f"group '{name}' is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"group '{name}' contains non-finite values")
        if arr.size < 2:
            raise ValueError(f"group '{name}' has fewer than 2 observations")
        variance = float(arr.var(ddof=1))
        if variance == 0:
            warnings.append(f"group '{name}' has zero variance")
        else:
            all_zero_variance = False
    if all_zero_variance:
        raise ValueError("all groups have zero variance — no variation to compare")
    return warnings
