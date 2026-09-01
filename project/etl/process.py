"""Dataset-agnostic ingest contract helpers.

Generic select/clean/validate primitives (``select_and_clean_columns`` and
``validate_contract_schema``) exercised by project/tests/test_ingest_contract.py
and project/tests/test_ingest_surface.py. The dataset-specific ingest pipeline
and its per-dataset configuration were removed with the legacy dataset layer.
"""

import logging

import pandera as pa
import pandas as pd

from broadway.config.schema import DatasetContract
from broadway.contracts.pandera import build_raw_schema, pandera_dtype

logger = logging.getLogger(__name__)

StageLedger = list[tuple[str, int]]


def select_and_clean_columns(
    df: pd.DataFrame,
    contract: DatasetContract,
    ledger: StageLedger | None = None,
) -> pd.DataFrame:
    cols = list(contract.columns.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"raw data missing required contract column(s): {sorted(missing)}")
    extra = [c for c in df.columns if c not in cols]
    if extra:
        logger.info(f"dropping extra column(s) not in contract: {sorted(extra)}")
    n_before = len(df)
    df = df[cols].dropna()
    logger.info(f"After dropna: {len(df)} ({n_before - len(df)} dropped)")
    if ledger is not None:
        ledger.append(("dropna", len(df)))
    n_duplicates_before = len(df)
    df = df.drop_duplicates()
    logger.info(f"After drop_duplicates: {len(df)} ({n_duplicates_before - len(df)} dropped)")
    if ledger is not None:
        ledger.append(("duplicates", len(df)))
    return df


def validate_contract_schema(df: pd.DataFrame, contract: DatasetContract) -> None:
    for name, col in contract.columns.items():
        if not col.dtype.strip():
            raise ValueError(f"column '{name}' is missing a dtype in the dataset contract")
        if pandera_dtype(col.dtype) is pa.Object:
            raise ValueError(f"column '{name}' has unsupported dtype '{col.dtype}'")
    build_raw_schema(contract).validate(df)
