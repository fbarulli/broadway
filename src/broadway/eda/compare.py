"""Before/after ETL side-by-side comparison."""

from __future__ import annotations

import pandas as pd


def compare(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_dropped": len(before) - len(after),
        "cols_before": len(before.columns),
        "cols_after": len(after.columns),
        "nulls_before": int(before.isna().sum().sum()),
        "nulls_after": int(after.isna().sum().sum()),
    }
